# 06 · Final performance

Final configuration (frozen): **B3** = structure-aware chunks · hybrid dense + BM25 (RRF) · pool 20 ·
bge-reranker-base → top 5 · Qwen2.5-1.5B-Instruct with grounding checks.

Commands: `python -m rag.cli eval --split dev`, then once at the end
`python -m rag.cli eval --split test`, then `python -m rag.cli report`.

> **Team to state here:** "The test split was run once, on <date>, after the configuration was frozen"
> (see `eval/results/test_split_runs.json`), or explain why a re-run was needed.

## Baselines (dev)
B0 closed-book · B1 BM25 only · B2 dense + fixed chunks · B3 final.
<!-- BEGIN:baselines_dev -->
<!-- END:baselines_dev -->

## Baselines (test)
<!-- BEGIN:baselines_test -->
<!-- END:baselines_test -->

## Ablation (one change at a time, B2 → B3)
<!-- BEGIN:ablation_test -->
<!-- END:ablation_test -->

Δ is the change in per-item pass rate; the CI is a 10,000-sample paired bootstrap; "fixed"/"broken" are
the discordant pairs used by the exact McNemar test. With ~16 test items a difference is only a finding
if its CI excludes 0.

## Error analysis
`python -m rag.cli failures --config B3 --split dev` exports failed items to `eval/failure_labels.csv`;
each needs a hand label (retrieval_miss, rank_failure, chunk_boundary, extraction_failure, stale_content,
reranker_demotion, generator_ignored_context, invalid_citation, over_refusal, hallucination).
<!-- BEGIN:failure_analysis -->
<!-- END:failure_analysis -->

> **Team to complete:** at least 15 labeled failures (add B2/B1 failures if B3 has fewer), with 3–4
> worked examples and what you would change next.

## Latency and cost
Latency p50/p95 are in the baseline tables (hardware in `eval/results/environment.json`). All models are
open-weights and run locally, so the API cost per query is $0; report the compute time instead.
