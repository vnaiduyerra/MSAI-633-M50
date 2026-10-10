"""Stage 3 - two chunking strategies over the saved normalized documents.

fixed      naive sliding window of FIXED_WORDS words with FIXED_OVERLAP overlap over the
           flattened page text (headings and tables are just more words).
structure  one chunk per heading section, split at paragraph boundaries when a section is
           longer than STRUCT_MAX_WORDS; the heading path is carried into every chunk; tables
           are kept whole (long tables are split by rows with the header row repeated).

Chunk IDs are deterministic: <strategy>-<doc_id>-<index>-<sha1(text)[:8]>, so re-running
chunking on unchanged text yields identical IDs.
"""

import hashlib
import json

import numpy as np

from . import config

STRATEGIES = ("fixed", "structure")


def chunk_id(strategy, doc_id, idx, text):
    return f"{strategy}-{doc_id}-{idx:04d}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:8]}"


def _base(doc, strategy):
    return {
        "strategy": strategy,
        "doc_id": doc["doc_id"],
        "source_url": doc["source_url"],
        "title": doc["title"],
        "page_type": doc["page_type"],
        "effective_date": doc.get("effective_date", ""),
        "modified_date": doc.get("modified_date", ""),
        "crawl_timestamp": doc["crawl_timestamp"],
    }


def _finish(chunks, doc, strategy):
    out = []
    for i, (text, heading, is_table) in enumerate(chunks):
        c = _base(doc, strategy)
        c.update({"chunk_id": chunk_id(strategy, doc["doc_id"], i, text), "text": text,
                  "heading_path": heading, "is_table": is_table, "n_words": len(text.split())})
        out.append(c)
    return out


def fixed_chunks(doc, size=config.FIXED_WORDS, overlap=config.FIXED_OVERLAP):
    words = []
    for s in doc["sections"]:
        words += " ".join(s["heading_path"][-1:]).split()
        for b in s["blocks"]:
            words += b["text"].split()
    step = max(1, size - overlap)
    pieces = []
    for start in range(0, max(len(words), 1), step):
        window = words[start:start + size]
        if window:
            pieces.append((" ".join(window), "", False))
        if start + size >= len(words):
            break
    return _finish(pieces, doc, "fixed")


def _split_table(md, max_rows):
    lines = md.split("\n")
    header, body = lines[:2], lines[2:]
    if len(body) <= max_rows:
        return [md]
    return ["\n".join(header + body[i:i + max_rows]) for i in range(0, len(body), max_rows)]


def structure_chunks(doc, max_words=config.STRUCT_MAX_WORDS, max_rows=config.TABLE_MAX_ROWS):
    pieces = []
    for s in doc["sections"]:
        path = [doc["title"]] + [h for h in s["heading_path"] if h != doc["title"]]
        heading = " > ".join(path)
        prefix = heading + "\n"
        buf, buf_words = [], 0

        def flush():
            nonlocal buf, buf_words
            if buf:
                pieces.append((prefix + "\n".join(buf), heading, False))
            buf, buf_words = [], 0

        for b in s["blocks"]:
            if b["type"] == "table":
                flush()
                for part in _split_table(b["text"], max_rows):
                    pieces.append((prefix + part, heading, True))
                continue
            n = len(b["text"].split())
            if buf and buf_words + n > max_words:
                flush()
            if n > max_words:  # a single very long paragraph: window it inside the section
                words = b["text"].split()
                for i in range(0, len(words), max_words - 30):
                    pieces.append((prefix + " ".join(words[i:i + max_words]), heading, False))
                    if i + max_words >= len(words):
                        break
                continue
            buf.append(b["text"])
            buf_words += n
        flush()
    return _finish(pieces, doc, "structure")


def chunk_docs(docs, strategy):
    fn = {"fixed": fixed_chunks, "structure": structure_chunks}[strategy]
    return [c for d in docs for c in fn(d)]


def save_chunks(chunks, strategy):
    path = config.chunks_file(strategy)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return path


def load_chunks(strategy):
    path = config.chunks_file(strategy)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def token_counter():
    """Count tokens with the embedding model's tokenizer (falls back to words offline)."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
        return (lambda text: len(tok.encode(text, add_special_tokens=False))), config.EMBED_MODEL
    except Exception:
        return (lambda text: len(text.split())), "whitespace words (tokenizer unavailable)"


def length_stats(chunks, count):
    lens = np.array([count(c["text"]) for c in chunks]) if chunks else np.array([0])
    return {
        "chunks": len(chunks),
        "docs": len({c["doc_id"] for c in chunks}),
        "table_chunks": sum(c.get("is_table", False) for c in chunks),
        "mean": round(float(lens.mean()), 1),
        "p5": int(np.percentile(lens, 5)),
        "p50": int(np.percentile(lens, 50)),
        "p95": int(np.percentile(lens, 95)),
        "max": int(lens.max()),
        "over_512": int((lens > 512).sum()),
    }
