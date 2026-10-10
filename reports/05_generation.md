# 05 · Generation, grounding and refusal

Generator: `Qwen/Qwen2.5-1.5B-Instruct`, greedy decoding, chat template, max 320 new tokens.

## Grounding rules enforced in code (`rag/pipeline.py`, `rag/generate.py`, `rag/safety.py`)
1. Only the top-5 reranked chunks are shown to the model, numbered `[1]…[5]`, with an instruction to cite
   a number after every factual sentence and to reply with a fixed refusal sentence otherwise.
2. **Refusal gate before generation**: if the best rerank score is below `RAG_MIN_RERANK_SCORE`
   (tuned on dev) the bot refuses without calling the model.
3. **Citation check after generation**: invalid citation numbers are removed and any paragraph without a
   valid citation is dropped; if nothing cited remains, the answer becomes a refusal.
4. **Quote limit**: any run of more than 25 consecutive words matching a source is cut to 25 words + "…".
5. Every citation is rendered as a clickable link with crawl date and effective/updated date.

## Results (dev)
<!-- BEGIN:generation_dev -->
<!-- END:generation_dev -->

## Results (test, run once)
<!-- BEGIN:generation_test -->
<!-- END:generation_test -->

> **Team to complete:** the chosen `RAG_MIN_RERANK_SCORE` and the dev evidence for it (correct-refusal
> vs over-refusal trade-off); two example answers (one good, one refusal) copied from the app.
