# 02 · Chunking and re-chunk migration

Commands: `python -m rag.cli chunk`, `python -m rag.cli index --strategy fixed`,
`python -m rag.cli index --strategy structure` (this second call *is* the migration),
`python -m rag.cli verify-metric`.

## Strategies
| | Fixed-size (naive) | Structure-aware |
|---|---|---|
| Unit | 180-word window, 40-word overlap over the flattened page | one heading section; split at paragraph boundaries above 300 words |
| Headings | just more words in the stream | full heading path (`Title > Law 41 > 41.3 > 41.3.1`) prefixed to every chunk |
| Tables | can be cut mid-row | whole table per chunk; long tables split by rows with the header repeated |

## Chunk counts and token-length distributions
<!-- BEGIN:chunking_stats -->
<!-- END:chunking_stats -->

## Embedding and metric
Embeddings are L2-normalized and FAISS uses inner product (`IndexFlatIP`), which equals cosine
similarity only for unit vectors - so `build_index` asserts unit norms before inserting, and
`verify-metric` compares the store's top-10 with a brute-force cosine ranking.

## Re-chunk migration (from saved normalized text, no re-crawl)
The live index holds one strategy. Switching deletes every chunk whose metadata `strategy` is the old one,
removes any leftover id not in the new chunk file, upserts the new chunks, then checks for orphans
(store ids not in the chunk file), missing chunks, and ids of the other strategy.
<!-- BEGIN:rechunk_migration -->
<!-- END:rechunk_migration -->

## Effect on the evaluation set
The B2 → A1 row of the ablation in `06_final_performance.md` is exactly this change (same retriever,
fixed → structure chunks).
<!-- BEGIN:ablation_dev -->
<!-- END:ablation_dev -->
