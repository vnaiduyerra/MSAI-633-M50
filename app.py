"""
Cricket Rules RAG Chatbot
=========================
Retrieval-Augmented Generation chatbot that answers questions about the Laws of
Cricket using only the document corpus in /data/cricket_rules.md.

Pipeline stages (per MSAI 633 project spec):
  1. Corpus        - cricket_rules.md (Laws of Cricket, MCC)
  2. Chunking      - structure-aware (split on headings) + fixed-size with overlap
  3. Embedding     - sentence-transformers/all-MiniLM-L6-v2 (384-dim, L2-normalized)
  4. Vector store  - FAISS (IndexFlatIP for cosine similarity)
  5. Retrieval     - Dense (FAISS) + BM25 + hybrid fusion (RRF)
  6. Reranking     - cross-encoder/ms-marco-MiniLM-L-6-v2
  7. Generation    - google/flan-t5-base (local, since HF Inference API is DNS-blocked)
  8. Refusal       - explicit "I don't have that information" when context is insufficient
  9. Citations     - source section headings attached to every factual answer

Run:  .venv/bin/python app.py
"""

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import re
import textwrap
from collections import Counter
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ──────────────────────────────────────────────────────────────────────
# 1. CORPUS
# ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
CORPUS_FILE = DATA_DIR / "cricket_rules.md"

def load_corpus():
    """Return list of {section, text} documents from the markdown corpus."""
    text = CORPUS_FILE.read_text(encoding="utf-8")
    sections = []
    current_heading = "Introduction"
    current_lines = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append({"section": current_heading, "text": "\n".join(current_lines).strip()})
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append({"section": current_heading, "text": "\n".join(current_lines).strip()})
    return sections

# ──────────────────────────────────────────────────────────────────────
# 2. CHUNKING
# ──────────────────────────────────────────────────────────────────────
def structure_aware_chunk(docs):
    """Split on headings; each heading block becomes a chunk carrying its heading path."""
    chunks = []
    for doc in docs:
        paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
        for para in paragraphs:
            chunks.append({"text": para, "section": doc["section"]})
    return chunks

def fixed_size_chunk(docs, chunk_size=150, overlap=30):
    """Fixed-size chunks with overlap, in words."""
    chunks = []
    for doc in docs:
        words = doc["text"].split()
        step = chunk_size - overlap
        for i in range(0, max(len(words), 1), step):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunks.append({
                    "text": " ".join(chunk_words),
                    "section": doc["section"]
                })
            if i + chunk_size >= len(words):
                break
    return chunks

# ──────────────────────────────────────────────────────────────────────
# 3. EMBEDDING
# ──────────────────────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.array(embeddings, dtype="float32")

# ──────────────────────────────────────────────────────────────────────
# 4. VECTOR STORE (FAISS)
# ──────────────────────────────────────────────────────────────────────
def build_faiss_index(embeddings):
    """Build FAISS inner-product index (equivalent to cosine for L2-normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# ──────────────────────────────────────────────────────────────────────
# 5. RETRIEVAL: Dense + BM25 + Hybrid (RRF)
# ──────────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def build_bm25(chunks):
    """Simple BM25 index."""
    tokenized = [tokenize(c["text"]) for c in chunks]
    N = len(tokenized)
    avgdl = sum(len(t) for t in tokenized) / max(N, 1)
    df = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1
    return {"tokenized": tokenized, "N": N, "avgdl": avgdl, "df": dict(df)}

def bm25_score(bm25_idx, chunk_idx, query, k1=1.5, b=0.75):
    tokens = bm25_idx["tokenized"][chunk_idx]
    N = bm25_idx["N"]
    avgdl = bm25_idx["avgdl"]
    df = bm25_idx["df"]
    score = 0.0
    for term in query:
        if term not in df:
            continue
        idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
        tf = tokens.count(term)
        denom = tf + k1 * (1 - b + b * len(tokens) / max(avgdl, 1))
        score += idf * (tf * (k1 + 1)) / max(denom, 1e-9)
    return score

def dense_retrieve(index, query_embedding, k=20):
    scores, indices = index.search(query_embedding.reshape(1, -1), k)
    return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

def bm25_retrieve(bm25_idx, query_terms, k=20):
    scores = []
    for i in range(bm25_idx["N"]):
        s = bm25_score(bm25_idx, i, query_terms)
        scores.append((i, s))
    scores.sort(key=lambda x: -x[1])
    return scores[:k]

def hybrid_retrieve(index, embeddings, bm25_idx, query, k=20):
    """Hybrid fusion using Reciprocal Rank Fusion (RRF)."""
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    dense_results = dict(dense_retrieve(index, q_emb, k))
    bm25_results = dict(bm25_retrieve(bm25_idx, tokenize(query), k))
    # RRF
    rrf = {}
    rrf_k = 60
    all_ids = set(dense_results) | set(bm25_results)
    for idx in all_ids:
        rank = 0.0
        if idx in dense_results:
            rank += 1.0 / (rrf_k + list(dense_results).index(idx) + 1)
        if idx in bm25_results:
            rank += 1.0 / (rrf_k + list(bm25_results).index(idx) + 1)
        rrf[idx] = rank
    sorted_results = sorted(rrf.items(), key=lambda x: -x[1])
    return sorted_results[:k]

# ──────────────────────────────────────────────────────────────────────
# 6. RERANKING (Cross-Encoder)
# ──────────────────────────────────────────────────────────────────────
print("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, chunks, retrieved_indices, top_k=5):
    pairs = [(query, chunks[i]["text"]) for i in retrieved_indices]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(retrieved_indices, scores), key=lambda x: -x[1])
    return ranked[:top_k]

# ──────────────────────────────────────────────────────────────────────
# 7. GENERATION (flan-t5-base)
# ──────────────────────────────────────────────────────────────────────
print("Loading generation model (flan-t5-small)...")
from transformers import T5ForConditionalGeneration, T5Tokenizer
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

def generate_answer(query, context_text, source_sections):
    """Generate an answer grounded in the retrieved context."""
    context = context_text[:3000]  # cap context length
    prompt = (
        f"Answer the question using only the context provided from the Laws of Cricket. "
        f"If the context does not contain the answer, say 'I don't have that information.' "
        f"Do not guess or use outside knowledge.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    inputs = t5_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = t5_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Check for refusal signals in generated answer
    refusal_phrases = ["i don't have that information", "not mentioned", "not found", "context does not"]
    if any(p in answer.lower() for p in refusal_phrases):
        answer = "I don't have that information in the provided corpus. Please rephrase your question or ask about a different cricket rule."
    return answer, source_sections

# ──────────────────────────────────────────────────────────────────────
# BUILD INDEX AT STARTUP
# ──────────────────────────────────────────────────────────────────────
print("Building corpus and indexes...")
docs = load_corpus()
chunks = structure_aware_chunk(docs)
print(f"  Corpus: {len(docs)} sections, {len(chunks)} chunks")

embeddings = embed_chunks(chunks)
faiss_index = build_faiss_index(embeddings)
bm25_index = build_bm25(chunks)
print(f"  FAISS index: {faiss_index.ntotal} vectors, {embeddings.shape[1]} dims")
print("Ready!")

# ──────────────────────────────────────────────────────────────────────
# CHATBOT LOGIC
# ──────────────────────────────────────────────────────────────────────
def cricket_rag_chat(question, history):
    """
    Main chatbot function. Returns (updated_history, retrieval_trace).
    """
    history = history or []

    # Retrieve
    retrieved = hybrid_retrieve(faiss_index, embeddings, bm25_index, question, k=20)
    retrieved_indices = [i for i, _ in retrieved]

    # Rerank
    reranked = rerank(question, chunks, retrieved_indices, top_k=5)
    top_indices = [i for i, _ in reranked]
    top_scores = {i: s for i, s in reranked}

    # Build context from top chunks
    context_parts = []
    source_sections = []
    for idx in top_indices:
        chunk = chunks[idx]
        context_parts.append(chunk["text"])
        if chunk["section"] not in source_sections:
            source_sections.append(chunk["section"])
    context_text = "\n\n".join(context_parts)

    # Check if we have any meaningful context (low scores = likely unanswerable)
    if not reranked or max(top_scores.values()) < 0.5:
        answer = ("I don't have that information in the provided corpus. "
                  "Please rephrase your question or ask about a different cricket rule.")
        sources = "N/A"
    else:
        answer, sources_list = generate_answer(question, context_text, source_sections)
        sources = "\n".join(f"- {s}" for s in sources_list)

    # Build retrieval trace
    trace_lines = []
    for rank, (idx, score) in enumerate(reranked):
        chunk = chunks[idx]
        trace_lines.append(
            f"{rank+1}. [{chunk['section']}] (score: {score:.3f})\n"
            f"   {chunk['text'][:200]}..."
        )
    trace = "\n\n".join(trace_lines)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    return history, trace, sources

# ──────────────────────────────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────────────────────────────
def create_ui():
    with gr.Blocks(title="Cricket Rules RAG Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏏 Cricket Rules RAG Chatbot")
        gr.Markdown(
            "Ask questions about the Laws of Cricket. Answers are grounded in the MCC rulebook "
            "corpus with source citations. If the corpus doesn't contain an answer, the bot will "
            "tell you so."
        )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                question_input = gr.Textbox(
                    placeholder="Ask a cricket rules question...",
                    label="Your Question",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=1):
                gr.Markdown("### 📚 Sources")
                sources_output = gr.Textbox(label="Cited Sections", lines=4, interactive=False)
                gr.Markdown("### 🔍 Retrieval Trace (Top 5 after reranking)")
                trace_output = gr.Textbox(label="Retrieval Trace", lines=15, interactive=False)

        def respond(question, history):
            history, trace, sources = cricket_rag_chat(question, history)
            return history, trace, sources, ""

        submit_btn.click(respond, [question_input, chatbot], [chatbot, trace_output, sources_output, question_input])
        question_input.submit(respond, [question_input, chatbot], [chatbot, trace_output, sources_output, question_input])
        clear_btn.click(lambda: ([], "", "", ""), outputs=[chatbot, trace_output, sources_output, question_input])

    return demo

# ──────────────────────────────────────────────────────────────────────
# LAUNCH
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
