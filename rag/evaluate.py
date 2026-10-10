"""Evaluation harness: per-item runs, retrieval/answer/refusal/latency metrics, comparisons."""

import json
import math
import platform
import re
from datetime import datetime, timezone

import numpy as np

from . import config, golden, stats
from .pipeline import ABLATION_PATH, CONFIGS

TEST_LOCK = config.RESULTS_DIR / "test_split_runs.json"


# ----------------------------------------------------------------------------- retrieval metrics
def _urls(cands):
    return [c.chunk["source_url"] for c in cands]


def recall_at_k(urls, gold, k):
    gold = set(gold)
    return len(gold & set(urls[:k])) / len(gold) if gold else float("nan")


def mrr(urls, gold, k=None):
    for r, u in enumerate(urls[:k] if k else urls, start=1):
        if u in gold:
            return 1.0 / r
    return 0.0


def ndcg_at_k(urls, gold, k=10, n_relevant=None):
    gains = [1.0 if u in gold else 0.0 for u in urls[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_n = min(k, n_relevant if n_relevant is not None else max(len(set(gold)), sum(gains)))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(int(ideal_n)))
    return dcg / idcg if idcg else 0.0


# ----------------------------------------------------------------------------- answer metrics
STOP = set("a an the of to in on at for and or is are was were be by with from as that this it its "
           "which who what when where how than then there their".split())


def content_tokens(s):
    return [t for t in re.findall(r"[a-z0-9.]+", s.lower()) if t not in STOP]


def auto_correct(answer, gold_answer, threshold=0.6):
    """Automatic proxy for correctness: >= 60% of the gold answer's content words appear in the
    answer and every number in the gold answer appears. Humans can override per item."""
    gold = content_tokens(gold_answer)
    if not gold:
        return False
    ans = set(content_tokens(answer))
    nums = [t for t in gold if re.fullmatch(r"\d+(\.\d+)?", t)]
    if any(n not in ans for n in nums):
        return False
    return sum(t in ans for t in gold) / len(gold) >= threshold


def evidence_in_chunk(evidence, chunk_text):
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()
    t = norm(chunk_text)
    return any(norm(e["quote"]) in t for e in evidence if e.get("quote"))


def score_item(item, ans, n_relevant_by_url):
    beh = golden.expected_behavior(item)
    gold_urls = item.get("gold_urls", [])
    rec = {
        "id": item["id"], "category": item["category"], "split": item.get("split"),
        "expected": beh, "question": item["question"], "gold_answer": item.get("gold_answer", ""),
        "answer": ans.text, "refused": ans.refused, "refusal_reason": ans.refusal_reason,
        "referral": ans.referral, "cited_urls": [c["url"] for c in ans.citations],
        "pool_urls": _urls(ans.pool), "ranked_urls": _urls(ans.ranked),
        "pool_chunk_ids": [c.chunk_id for c in ans.pool], "ranked_chunk_ids": [c.chunk_id for c in ans.ranked],
        "blocked": [b[0] for b in ans.blocked], "quote_cuts": ans.quote_cuts,
        "timings": ans.timings, "human_correct": None, "failure_label": "",
    }
    if beh == "answer" and gold_urls:
        n_rel = sum(n_relevant_by_url.get(u, 0) for u in gold_urls)
        for prefix, urls in (("pre", rec["pool_urls"]), ("post", rec["ranked_urls"])):
            rec[f"{prefix}_recall@5"] = recall_at_k(urls, gold_urls, 5)
            rec[f"{prefix}_recall@20"] = recall_at_k(urls, gold_urls, 20)
            rec[f"{prefix}_mrr"] = mrr(urls, gold_urls)
            rec[f"{prefix}_ndcg@10"] = ndcg_at_k(urls, gold_urls, 10, n_rel or None)
        rec["auto_correct"] = (not ans.refused) and auto_correct(ans.text, item.get("gold_answer", ""))
        if ans.citations:
            ok = [c["url"] in gold_urls or evidence_in_chunk(item.get("evidence", []),
                                                             next((x["text"] for x in ans.context
                                                                   if x["chunk_id"] == c["chunk_id"]), ""))
                  for c in ans.citations]
            rec["citation_valid"] = all(ok)
            rec["citation_precision"] = sum(ok) / len(ok)
    elif beh == "clarify":
        rec["auto_correct"] = ans.refused or "?" in ans.text
    else:
        rec["auto_correct"] = ans.refused
    return rec


def item_pass(rec):
    correct = rec["human_correct"] if rec.get("human_correct") is not None else rec["auto_correct"]
    return bool(correct)


# ----------------------------------------------------------------------------- running
def select_items(items, split):
    return [it for it in items if split == "all" or it.get("split") == split]


def guard_test_split(config_names, force=False):
    TEST_LOCK.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(TEST_LOCK.read_text()) if TEST_LOCK.exists() else []
    already = sorted({c for entry in log for c in entry["configs"]} & set(config_names))
    if already and not force:
        raise SystemExit(f"The test split was already run for {already}. The spec allows one test run "
                         f"at the end; pass --force-test only if that run genuinely failed, and say so "
                         f"in the report.")
    log.append({"at": datetime.now(timezone.utc).isoformat(), "configs": config_names, "forced": force})
    TEST_LOCK.write_text(json.dumps(log, indent=2))


def run_config(name, items, build_pipeline, log=print):
    pipe, n_relevant_by_url = build_pipeline(CONFIGS[name])
    out = []
    for n, it in enumerate(items, start=1):
        ans = pipe.answer(it["question"])
        rec = score_item(it, ans, n_relevant_by_url)
        rec["config"] = name
        out.append(rec)
        log(f"  [{name} {n}/{len(items)}] {it['id']} pass={item_pass(rec)} refused={ans.refused} "
            f"{ans.timings.get('total', 0):.1f}s")
    return out


def results_path(name, split):
    return config.RESULTS_DIR / f"{name}_{split}.jsonl"


def save_results(recs, name, split):
    p = results_path(name, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def load_results(name, split):
    p = results_path(name, split)
    if not p.exists():
        return None
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = config.EVAL_DIR / "human_labels.csv"   # optional human overrides: config,id,human_correct
    if labels.exists():
        import csv

        over = {(r["config"], r["id"]): r["human_correct"].strip().lower() in ("1", "true", "yes")
                for r in csv.DictReader(labels.open()) if r.get("human_correct", "").strip()}
        for r in recs:
            if (name, r["id"]) in over:
                r["human_correct"] = over[(name, r["id"])]
    return recs


# ----------------------------------------------------------------------------- summaries
def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def summarize(recs):
    ans = [r for r in recs if r["expected"] == "answer"]
    unans = [r for r in recs if r["expected"] == "refuse"]
    totals = [r["timings"].get("total", 0.0) for r in recs]
    s = {"n_items": len(recs), "n_answerable": len(ans), "n_unanswerable": len(unans)}
    for prefix in ("pre", "post"):
        for m in ("recall@5", "recall@20", "mrr", "ndcg@10"):
            s[f"{prefix}_{m}"] = _mean([r.get(f"{prefix}_{m}") for r in ans])
    s["answer_correct"] = _mean([item_pass(r) for r in ans])
    s["citation_valid"] = _mean([r["citation_valid"] for r in ans if "citation_valid" in r])
    s["citation_precision"] = _mean([r["citation_precision"] for r in ans if "citation_precision" in r])
    s["correct_refusal_rate"] = _mean([r["refused"] for r in unans])
    s["over_refusal_rate"] = _mean([r["refused"] for r in ans])
    s["pass_rate"] = _mean([item_pass(r) for r in recs])
    s["latency_p50_s"] = float(np.percentile(totals, 50)) if totals else float("nan")
    s["latency_p95_s"] = float(np.percentile(totals, 95)) if totals else float("nan")
    for stage in ("retrieve", "rerank", "generate"):
        vals = [r["timings"][stage] for r in recs if stage in r["timings"]]
        s[f"{stage}_p50_s"] = float(np.percentile(vals, 50)) if vals else float("nan")
    s["cost_per_query_usd"] = 0.0  # local open-weights inference, no API fees; see compute time
    s["pass_ci95"] = stats.bootstrap_mean_ci([item_pass(r) for r in recs])[1:]
    return s


def compare(recs_a, recs_b):
    """Paired comparison on items present in both runs."""
    a = {r["id"]: r for r in recs_a}
    b = {r["id"]: r for r in recs_b}
    ids = sorted(set(a) & set(b))
    pa = [item_pass(a[i]) for i in ids]
    pb = [item_pass(b[i]) for i in ids]
    diff, lo, hi = stats.bootstrap_diff_ci(pa, pb)
    b01, b10, p = stats.mcnemar_exact(pa, pb)
    return {"n": len(ids), "pass_a": _mean(pa), "pass_b": _mean(pb), "diff": diff, "ci95": (lo, hi),
            "a_fail_b_pass": b01, "a_pass_b_fail": b10, "mcnemar_p": p}


def b0_incorrect_subset(b0_recs, recs):
    """Headline metric restricted to answerable items the closed-book model got wrong."""
    wrong = {r["id"] for r in b0_recs if r["expected"] == "answer" and not item_pass(r)}
    sub = [r for r in recs if r["id"] in wrong]
    return {"n": len(sub), "answer_correct": _mean([item_pass(r) for r in sub])}


def environment_info():
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        info["device"] = "unknown"
    return info


__all__ = ["ABLATION_PATH", "CONFIGS"]
