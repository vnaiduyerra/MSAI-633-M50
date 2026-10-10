"""Turns saved run artifacts into Markdown tables and splices them into reports/*.md.

Every table is computed from files the pipeline wrote (manifest, docs, chunk files, index
state, eval results). Nothing is typed in by hand, so a table is either real or absent.
"""

import json
import math
import re
from collections import Counter

from . import config, evaluate, golden
from .pipeline import ABLATION_PATH

MARKER_RX = re.compile(r"(<!-- BEGIN:(?P<name>[\w-]+) -->)(.*?)(<!-- END:(?P=name) -->)", re.S)


def fmt(x, digits=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return "–" if math.isnan(x) else f"{x:.{digits}f}"
    return str(x)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write(name, md):
    config.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (config.GENERATED_DIR / f"{name}.md").write_text(md.strip() + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- per-stage tables
def ingestion_report(extraction_stats):
    manifest = config.RAW_DIR / "manifest.jsonl"
    recs = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()] if manifest.exists() else []
    crawl_log = config.RAW_DIR / "crawl_log.jsonl"
    events = Counter(json.loads(l)["event"] for l in crawl_log.read_text().splitlines()
                     if l.strip()) if crawl_log.exists() else Counter()
    ctypes = Counter(r["content_type"] or "unknown" for r in recs)
    rows = [["raw documents saved", len(recs)], ["raw bytes", sum(r["bytes"] for r in recs)]]
    rows += [[f"content type: {k}", v] for k, v in sorted(ctypes.items())]
    rows += [[f"skipped: {k}", v] for k, v in sorted(events.items())]
    rows += [[k, v] for k, v in sorted(extraction_stats.items())]
    if recs:
        rows += [["first fetch", min(r["fetched_at"] for r in recs)], ["last fetch", max(r["fetched_at"] for r in recs)]]
    write("ingestion_stats", table(["Measure", "Value"], rows))


def chunking_report(stats_by_strategy, tokenizer_name):
    keys = ["chunks", "docs", "table_chunks", "mean", "p5", "p50", "p95", "max", "over_512"]
    rows = [[s] + [st[k] for k in keys] for s, st in stats_by_strategy.items()]
    md = f"Token counts use the `{tokenizer_name}` tokenizer.\n\n" + table(["Strategy"] + keys, rows)
    write("chunking_stats", md)


def migration_report():
    if not config.INDEX_STATE_FILE.exists():
        return
    state = json.loads(config.INDEX_STATE_FILE.read_text())
    rows = [[m["at"][:19], m["previous_strategy"] or "(empty)", m["strategy"], m["deleted_stale_chunks"],
             m["deleted_leftover_chunks"], m["inserted_or_updated"], m["store_count"], m["n_orphans"],
             m["n_missing"], m["other_strategy_ids"]] for m in state.get("migrations", [])]
    emb = state.get("embedding", {})
    md = (f"Embedding: `{emb.get('model')}` revision `{emb.get('revision')}`, dimension {emb.get('dimension')}, "
          f"L2-normalized: {emb.get('l2_normalized')}, metric: {emb.get('distance_metric')}; "
          f"vector store: FAISS inner product on unit vectors (= cosine).\n\n")
    md += table(["When (UTC)", "From", "To", "Stale deleted", "Leftover deleted", "Inserted", "Store count",
                 "Orphans", "Missing", "Other-strategy IDs"], rows)
    if "metric_check" in state:
        md += f"\n\nStore vs brute-force cosine top-10 overlap: **{fmt(state['metric_check'])}** (1.000 = identical)."
    write("rechunk_migration", md)


def golden_report(items, problems):
    cats = list(golden.QUOTAS)
    rows = []
    for c in cats:
        group = [i for i in items if i["category"] == c]
        rows.append([c, golden.QUOTAS[c], len(group), sum(i.get("split") == "dev" for i in group),
                     sum(i.get("split") == "test" for i in group), sum(bool(i.get("verified_by")) for i in group)])
    md = table(["Category", "Required", "Items", "Dev", "Test", "Human-verified"], rows)
    md += f"\n\nValidation problems: **{len(problems)}**"
    if problems:
        md += "\n\n" + "\n".join(f"- {p}" for p in problems[:50])
    write("golden_set_summary", md)


def _load(name, split):
    return evaluate.load_results(name, split)


def eval_reports(split):
    runs = {n: _load(n, split) for n in evaluate.CONFIGS}
    runs = {n: r for n, r in runs.items() if r}
    if not runs:
        return
    sums = {n: evaluate.summarize(r) for n, r in runs.items()}
    tag = f"split = **{split}**"

    # retrieval
    rows = []
    for n, s in sums.items():
        if n == "B0":
            continue
        rows.append([n, s["pre_recall@5"], s["pre_recall@20"], s["pre_mrr"], s["pre_ndcg@10"],
                     s["post_recall@5"], s["post_mrr"], s["post_ndcg@10"], s["retrieve_p50_s"], s["rerank_p50_s"]])
    md = (f"{tag}. *pre* = retrieval order of the 20-candidate pool, *post* = after reranking "
          f"(identical when the config has no reranker). Recall@20 of the pool is the ceiling for the reranker.\n\n")
    md += table(["Config", "R@5 pre", "R@20 pool", "MRR pre", "nDCG@10 pre", "R@5 post", "MRR post",
                 "nDCG@10 post", "retrieve p50 s", "rerank p50 s"], rows)
    write(f"retrieval_{split}", md)

    # generation / refusal
    rows = [[n, s["n_items"], s["answer_correct"], s["citation_valid"], s["citation_precision"],
             s["correct_refusal_rate"], s["over_refusal_rate"], s["generate_p50_s"]] for n, s in sums.items()]
    write(f"generation_{split}", f"{tag}\n\n" + table(
        ["Config", "Items", "Answer correct", "Citation valid", "Citation precision", "Correct refusal",
         "Over-refusal", "generate p50 s"], rows))

    # baselines + B0 leakage + latency/cost
    rows = [[n, s["pass_rate"], f"[{fmt(s['pass_ci95'][0])}, {fmt(s['pass_ci95'][1])}]", s["answer_correct"],
             s["correct_refusal_rate"], s["over_refusal_rate"], s["latency_p50_s"], s["latency_p95_s"],
             s["cost_per_query_usd"]] for n, s in sums.items() if n in ("B0", "B1", "B2", "B3")]
    md = f"{tag}\n\n" + table(["Config", "Pass rate", "95% CI", "Answer correct", "Correct refusal",
                               "Over-refusal", "p50 s", "p95 s", "API $/query"], rows)
    if "B0" in runs:
        md += "\n\n**On the B0-incorrect subset** (answerable items the closed-book model got wrong):\n\n"
        md += table(["Config", "Items", "Answer correct"],
                    [[n, *evaluate.b0_incorrect_subset(runs["B0"], r).values()] for n, r in runs.items() if n != "B0"])
    write(f"baselines_{split}", md)

    # ablation
    rows = []
    for a, b in zip(ABLATION_PATH, ABLATION_PATH[1:]):
        if a in runs and b in runs:
            c = evaluate.compare(runs[a], runs[b])
            rows.append([f"{a} → {b}", c["n"], c["pass_a"], c["pass_b"], c["diff"],
                         f"[{fmt(c['ci95'][0])}, {fmt(c['ci95'][1])}]", c["a_fail_b_pass"], c["a_pass_b_fail"],
                         c["mcnemar_p"]])
    if rows:
        md = (f"{tag}. Each step changes one factor: B2 (dense, fixed chunks) → A1 structure-aware chunks → "
              f"A2 hybrid BM25+dense (RRF) → B3 + cross-encoder rerank.\n\n")
        md += table(["Step", "n", "Pass before", "Pass after", "Δ", "95% bootstrap CI", "fixed", "broken",
                     "McNemar p"], rows)
        write(f"ablation_{split}", md)


def failure_report():
    path = config.EVAL_DIR / "failure_labels.csv"
    if not path.exists():
        return
    import csv

    rows = [r for r in csv.DictReader(path.open()) if r.get("failure_label", "").strip()]
    counts = Counter(r["failure_label"].strip() for r in rows)
    md = f"{len(rows)} hand-labeled failures.\n\n" + table(["Failure type", "Count"], sorted(counts.items(), key=lambda x: -x[1]))
    write("failure_analysis", md)


def fill_reports(log=print):
    """Replace every <!-- BEGIN:name --> ... <!-- END:name --> block with reports/generated/name.md."""
    for report in sorted(config.REPORTS_DIR.glob("*.md")):
        text = report.read_text(encoding="utf-8")

        def sub(m):
            gen = config.GENERATED_DIR / f"{m.group('name')}.md"
            body = gen.read_text(encoding="utf-8").strip() if gen.exists() else \
                f"_Not generated yet - see the README for the command that produces `{m.group('name')}`._"
            return f"{m.group(1)}\n{body}\n{m.group(4)}"

        new = MARKER_RX.sub(sub, text)
        if new != text:
            report.write_text(new, encoding="utf-8")
            log(f"updated reports/{report.name}")
