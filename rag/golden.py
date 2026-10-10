"""Golden evaluation set: schema, validation against the corpus, stratified dev/test split.

One JSON object per line in eval/golden_set.jsonl:

{"id": "SH01", "category": "single_hop", "question": "...", "gold_answer": "...",
 "gold_urls": ["https://..."], "evidence": [{"url": "https://...", "quote": "<= 25 words"}],
 "searched_for": "", "expected_behavior": "answer", "verified_by": "Full Name",
 "verified_on": "2026-10-10", "split": "dev", "notes": ""}

Unanswerable items leave gold_answer/gold_urls/evidence empty and record what was searched
for in searched_for. expected_behavior is answer | refuse | clarify (ambiguous items).
"""

import json
import random
import re
from collections import Counter, defaultdict

from . import config

GOLDEN_FILE = config.EVAL_DIR / "golden_set.jsonl"
QUOTAS = {"single_hop": 15, "multi_hop": 6, "comparative": 4, "temporal": 4,
          "unanswerable": 8, "ambiguous": 3}
MIN_TOTAL = 40


def load_golden(path=GOLDEN_FILE):
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() and not line.lstrip().startswith("//"):
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON ({exc})") from None
    return items


def expected_behavior(item):
    if item.get("expected_behavior"):
        return item["expected_behavior"]
    return "refuse" if item["category"] == "unanswerable" else "answer"


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def validate(items, docs=None):
    """Return a list of problems (empty list = valid). docs enables quote-in-page checks."""
    problems = []
    text_by_url = {}
    if docs:
        for d in docs:
            text_by_url[d["source_url"]] = _norm(" ".join(b["text"] for s in d["sections"] for b in s["blocks"]))
    ids = Counter(i.get("id") for i in items)
    for dup, n in ids.items():
        if n > 1:
            problems.append(f"duplicate id {dup}")
    cats = Counter(i.get("category") for i in items)
    for cat, need in QUOTAS.items():
        if cats.get(cat, 0) < need:
            problems.append(f"category {cat}: {cats.get(cat, 0)} items, need >= {need}")
    if len(items) < MIN_TOTAL:
        problems.append(f"{len(items)} items, need >= {MIN_TOTAL}")
    for it in items:
        iid = it.get("id", "?")
        if str(iid).upper().startswith("EXAMPLE"):
            problems.append(f"{iid}: template example still present - delete it")
        if it.get("category") not in QUOTAS:
            problems.append(f"{iid}: unknown category {it.get('category')!r}")
        if not it.get("question"):
            problems.append(f"{iid}: missing question")
        if not it.get("verified_by"):
            problems.append(f"{iid}: verified_by is empty - a human must check the gold answer against the page")
        beh = expected_behavior(it)
        if beh not in ("answer", "refuse", "clarify"):
            problems.append(f"{iid}: expected_behavior must be answer/refuse/clarify")
        if it.get("category") == "unanswerable" or beh == "refuse":
            if not it.get("searched_for"):
                problems.append(f"{iid}: unanswerable item must record what was searched for")
            continue
        if beh == "answer":
            if not it.get("gold_answer"):
                problems.append(f"{iid}: missing gold_answer")
            if not it.get("gold_urls"):
                problems.append(f"{iid}: missing gold_urls")
            if not it.get("evidence"):
                problems.append(f"{iid}: missing evidence quote")
        if it.get("category") == "multi_hop" and len(set(it.get("gold_urls", []))) < 2:
            problems.append(f"{iid}: multi_hop needs >= 2 different gold pages")
        for ev in it.get("evidence", []):
            q = ev.get("quote", "")
            if len(q.split()) > config.MAX_QUOTE_WORDS:
                problems.append(f"{iid}: evidence quote is {len(q.split())} words (max 25)")
            if ev.get("url") not in it.get("gold_urls", []):
                problems.append(f"{iid}: evidence url not in gold_urls: {ev.get('url')}")
            if text_by_url:
                page = text_by_url.get(ev.get("url"))
                if page is None:
                    problems.append(f"{iid}: evidence url not in the corpus: {ev.get('url')}")
                elif _norm(q) not in page:
                    problems.append(f"{iid}: evidence quote not found verbatim on {ev.get('url')}")
    return problems


def assign_split(items, dev_frac=0.6, seed=633, overwrite=False):
    """Stratified 60/40 dev/test split by category (deterministic)."""
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    rng = random.Random(seed)
    for cat, group in sorted(by_cat.items()):
        if not overwrite and all(it.get("split") in ("dev", "test") for it in group):
            continue
        group = sorted(group, key=lambda it: it["id"])
        rng.shuffle(group)
        n_dev = round(len(group) * dev_frac)
        for n, it in enumerate(group):
            it["split"] = "dev" if n < n_dev else "test"
    return items


def save_golden(items, path=GOLDEN_FILE):
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
