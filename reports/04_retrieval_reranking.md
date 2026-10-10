# 04 · Retrieval and reranking

Command: `python -m rag.cli eval --split dev` (runs all configs; tables below are generated).

- **Dense**: bge-small query embedding (with the bge query instruction) against the FAISS index, optional
  `page_type` metadata filter (`law`, `spirit`, `pdf`, `news`, `general`).
- **BM25**: Okapi (k1 = 1.5, b = 0.75) over the same chunks, same filter.
- **Hybrid**: Reciprocal Rank Fusion of the two top-20 lists (k = 60).
- **Pool**: 20 candidates. **Reranker**: bge-reranker-base scores all 20; the top 5 go to the generator.
  Relevance for metrics = chunk's `source_url` is one of the item's gold URLs.

## Results (dev)
<!-- BEGIN:retrieval_dev -->
<!-- END:retrieval_dev -->

> **Team to complete:** reranker lift = (post − pre) for R@5 / MRR / nDCG@10 on B3; added latency =
> rerank p50; Recall@20 of the pool as the ceiling; whether metadata filtering changed anything (try
> `python -m rag.cli ask "..." --page-type pdf`).

## Results (test, run once)
<!-- BEGIN:retrieval_test -->
<!-- END:retrieval_test -->
