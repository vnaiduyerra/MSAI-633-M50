---
title: Cricket Rules RAG Chatbot
emoji: 🏏
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
---

# MSAI-633-M50 — Cricket Rules RAG Chatbot

A retrieval-augmented chatbot that answers questions about the **Laws of Cricket** using only pages
crawled from the public MCC site, cites a source link for every factual claim, refuses when its sources
don't cover a question, and refers high-stakes questions to a person. Built to the MSAI 633 residency spec:
crawl → extract → two chunkers → embeddings → vector DB → re-chunk migration → dense/BM25/hybrid
retrieval → cross-encoder reranking → grounded generation, plus a full evaluation harness.

| Stage | Implementation |
|---|---|
| Crawl | `rag/crawl.py` — robots.txt enforced, ≥1 s/request, contact User-Agent, raw bytes + SHA-256 manifest |
| Extract | `rag/extract.py` — boilerplate removal, tables → Markdown, PDF text, dedupe, dates, PII redaction |
| Chunk | `rag/chunking.py` — fixed-size with overlap and structure-aware (heading path), deterministic IDs |
| Embed | `BAAI/bge-small-en-v1.5`, L2-normalized, cosine |
| Store | FAISS `IndexIDMap2(IndexFlatIP)` + JSON metadata (`rag/store.py`) — filter on `page_type`, delete by id / doc / strategy |
| Retrieve | dense, BM25, hybrid RRF over a pool of 20 (`rag/retrieval.py`) |
| Rerank | `BAAI/bge-reranker-base` → top 5 |
| Generate | `Qwen/Qwen2.5-1.5B-Instruct` with citation check, 25-word quote limit, refusal gate (`rag/pipeline.py`) |
| UI | Gradio (`app.py`) — chat history, clickable citations, retrieval trace with pre/post-rerank scores |

Model choices are explained in `reports/00_proposal.md`. Swap models with `RAG_EMBED_MODEL`,
`RAG_RERANK_MODEL`, `RAG_LLM` (e.g. `RAG_LLM=Qwen/Qwen2.5-3B-Instruct` on a GPU).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v      # offline tests, no downloads
```
Or open [`run_in_colab.ipynb`](https://colab.research.google.com/github/vnaiduyerra/MSAI-633-M50/blob/claude/amazing-fermat-bu68c9/run_in_colab.ipynb)
in Colab (**Runtime → Change runtime type → T4 GPU**, then run the cells in order).

## Reproduce every reported number
```bash
export CRAWL_CONTACT_EMAIL=you@ucumberlands.edu
python -m rag.cli crawl                         # data/raw/ + manifest.jsonl (≈ 7 min for 400 pages)
python -m rag.cli extract                       # data/processed/docs.jsonl + ingestion table
python -m rag.cli chunk                         # both strategies + token-length table
python -m rag.cli index --strategy fixed        # initial index
python -m rag.cli index --strategy structure    # re-chunk migration: deletes stale chunks, orphan check
python -m rag.cli verify-metric                 # store ranking vs brute-force cosine
# write eval/golden_set.jsonl by hand (see eval/README.md), then:
python -m rag.cli golden --assign-split         # validate against the corpus + stratified 60/40 split
python -m rag.cli eval --split dev              # B0, B1, B2, A1, A2, B3 on dev
python -m rag.cli injection-probe               # planted-page prompt-injection test
python -m rag.cli failures --config B3          # export failures, then hand-label them
python -m rag.cli eval --split test             # ONCE, after freezing the configuration
python -m rag.cli report                        # regenerate tables and fill reports/*.md
python app.py                                   # chatbot at http://localhost:7861 (+ gradio.live link)
```
Every table in `reports/` is generated from these outputs; nothing is typed in by hand.
After a successful run, `pip freeze > requirements.lock.txt` records the exact package versions used
(the Colab notebook does this); commit that file alongside the results.

## Repository layout
```
app.py                  Gradio chatbot
crawl_config.json       crawl scope (seeds, allowed prefixes, exclusions, page types)
rag/                    pipeline package (crawl, extract, chunking, embed, store, indexer,
                        retrieval, generate, safety, pipeline, golden, evaluate, stats, reports, cli)
data/planted/           prompt-injection test page (added to the corpus as page_type=planted_test)
data/processed/         normalized documents and chunk files (committed after the crawl)
eval/                   golden set, results, human labels, failure labels
reports/00…07_*.md      project reports; reports/generated/ holds the generated tables
tests/                  offline unit tests with fake models
run_in_colab.ipynb      Colab runner
```

## Deploy to the Hugging Face Space
The YAML header above is the Space config. Commit `data/processed/` (so the Space can index without
crawling), then push to https://huggingface.co/spaces/vnaiduyerra/Rag_Model. Use **CPU basic** hardware
(not ZeroGPU); on CPU set the Space variable `RAG_LLM=Qwen/Qwen2.5-0.5B-Instruct` for faster replies.

## Known limitations
- PDF tables come out as plain text (pypdf has no table detection); HTML tables are preserved.
- Answer correctness is first scored by an automatic word-overlap check; humans should review and record
  overrides in `eval/human_labels.csv`.
- Follow-up questions are handled by prepending the previous question for retrieval, not by an LLM rewrite.
- Small open models can still misread a source; the citation and quote checks catch format problems,
  not every factual error.
- The crawl scope in `crawl_config.json` was written without being able to test it against lords.org;
  confirm robots.txt and page counts on the first run.

## AI usage
Claude Code (Anthropic) wrote the pipeline code, tests, report scaffolding and Colab notebook in this
branch (commits authored "Claude"). No golden-set items, gold answers or reported numbers were produced
by AI: the golden set must be written and verified by the team, and every number comes from running the
commands above. **Team to add:** any other AI tools used, how, and on which artifacts.

## Team workflow (spec §6)
`main` is protected; work on feature branches merged through reviewed pull requests; each member commits
under their own account across more than one stage; track owners on the GitHub Projects board; `.env`
and keys are git-ignored.
