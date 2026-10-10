# Evaluation

| File | Written by | Purpose |
|---|---|---|
| `golden_set.jsonl` | **the team, by hand** | ≥40 items with gold answers, URLs and ≤25-word evidence quotes (start from `golden_set.template.jsonl`) |
| `results/<config>_<split>.jsonl` | `rag.cli eval` | one line per item: answer, citations, pool/ranked URLs, metrics, timings |
| `results/test_split_runs.json` | `rag.cli eval --split test` | proof of when the test split was run (it refuses a second run) |
| `human_labels.csv` | team (optional) | `config,id,human_correct` overrides for the automatic correctness check |
| `failure_labels.csv` | `rag.cli failures`, then the team | hand-labeled failure types (≥15 required) |

Categories and minimums: single_hop 15, multi_hop 6, comparative 4, temporal 4, unanswerable 8, ambiguous 3.
`python -m rag.cli golden --assign-split` validates the set against the corpus (quotes must appear
verbatim on the cited page) and assigns a stratified 60/40 dev/test split.

Answer correctness is first scored automatically (≥60% of the gold answer's content words present and
every number matching). Check those verdicts by hand and record disagreements in `human_labels.csv`;
the report tables use the human label whenever one exists.
