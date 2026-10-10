# 03 · Golden evaluation set

File: `eval/golden_set.jsonl` · validate and split: `python -m rag.cli golden --assign-split`

> **Team to complete:** who wrote which items, on which day (must be Friday, before the chatbot
> worked), who verified each gold answer against the live page, and how ambiguous / adversarial items
> were designed (e.g. "LBW" vs "leg before wicket", "what's the rule for bouncers in Tests?" when the
> Laws don't cover playing conditions, an injection attempt inside a question).

## Composition and split
<!-- BEGIN:golden_set_summary -->
<!-- END:golden_set_summary -->

The validator rejects the set unless every answerable item has gold URLs that exist in the corpus and an
evidence quote of ≤25 words found verbatim on that page, every unanswerable item records what was
searched for, every item names a human verifier, and the category minimums are met. The split is
stratified by category with a fixed seed; tuning used dev only, and the CLI records the single test run
in `eval/results/test_split_runs.json`.
