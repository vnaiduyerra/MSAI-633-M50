"""End-to-end question answering, shared by the Gradio app and the evaluation harness.

A PipelineConfig switches one factor at a time, which is how the baselines (B0-B3) and the
ablation steps are produced from the same code path.
"""

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from . import config, safety
from .generate import build_messages, enforce_citations, is_refusal
from .retrieval import rerank


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    retrieval: str = "hybrid"        # none | bm25 | dense | hybrid
    strategy: str = "structure"      # fixed | structure
    rerank: bool = True
    pool_k: int = config.POOL_K
    top_k: int = config.TOP_K
    min_rerank_score: float = config.MIN_RERANK_SCORE
    injection_filter: bool = True
    quote_limit: int = config.MAX_QUOTE_WORDS


CONFIGS = {
    "B0": PipelineConfig("B0", retrieval="none", rerank=False),
    "B1": PipelineConfig("B1", retrieval="bm25", strategy="structure", rerank=False),
    "B2": PipelineConfig("B2", retrieval="dense", strategy="fixed", rerank=False),
    # one-factor-at-a-time path from B2 to B3
    "A1": PipelineConfig("A1", retrieval="dense", strategy="structure", rerank=False),
    "A2": PipelineConfig("A2", retrieval="hybrid", strategy="structure", rerank=False),
    "B3": PipelineConfig("B3", retrieval="hybrid", strategy="structure", rerank=True),
}
ABLATION_PATH = ["B2", "A1", "A2", "B3"]
FOLLOWUP_RX = re.compile(r"(?i)\b(it|its|that|this|those|these|they|them|he|she|what about|and if|same)\b")


@dataclass
class Answer:
    question: str
    retrieval_query: str
    text: str
    refused: bool
    refusal_reason: str = ""
    citations: list = field(default_factory=list)   # [{n, url, title, heading, crawled, effective, stale}]
    referral: str = ""
    pool: list = field(default_factory=list)        # Candidate objects (pre-rerank order)
    ranked: list = field(default_factory=list)      # Candidate objects (final order)
    context: list = field(default_factory=list)     # chunks actually given to the model
    blocked: list = field(default_factory=list)     # (chunk_id, matched text) withheld by the injection filter
    quote_cuts: int = 0
    dropped_paragraphs: int = 0
    raw_model_output: str = ""
    timings: dict = field(default_factory=dict)


def _age_days(ts):
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return None


def retrieval_query(question, history):
    """Follow-ups ("what about in T20?") borrow the previous user question for retrieval."""
    prev = [t["content"] for t in (history or []) if t["role"] == "user"]
    if prev and (len(question.split()) <= 8 or FOLLOWUP_RX.search(question)):
        return f"{prev[-1]} {question}"
    return question


class RAGPipeline:
    def __init__(self, cfg, retriever=None, reranker=None, generator=None):
        self.cfg = cfg
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def answer(self, question, history=None, page_type=None):
        cfg = self.cfg
        t_start = time.perf_counter()
        timings = {}
        rq = retrieval_query(question, history)
        result = Answer(question=question, retrieval_query=rq, text="", refused=False,
                        referral=safety.referral_for(question) or "")

        if cfg.retrieval == "none":
            t0 = time.perf_counter()
            out = self.generator.chat(build_messages(question, [], history, closed_book=True))
            timings["generate"] = time.perf_counter() - t0
            result.raw_model_output = out
            result.text, result.refused = out, is_refusal(out)
            if result.refused:
                result.refusal_reason = "model"
            timings["total"] = time.perf_counter() - t_start
            result.timings = timings
            return result

        t0 = time.perf_counter()
        pool = self.retriever.retrieve(rq, cfg.retrieval, cfg.pool_k, page_type)
        timings["retrieve"] = time.perf_counter() - t0
        ranked = pool
        if cfg.rerank and pool:
            ranked, timings["rerank"] = rerank(self.reranker, rq, list(pool))
        result.pool, result.ranked = pool, ranked

        top = []
        for c in ranked:
            hit = safety.find_injection(c.chunk["text"]) if cfg.injection_filter else None
            if hit:
                c.blocked = hit
                result.blocked.append((c.chunk_id, hit))
                continue
            top.append(c)
            if len(top) == cfg.top_k:
                break
        result.context = [c.chunk for c in top]

        if not top:
            return self._refuse(result, "no usable sources retrieved", timings, t_start)
        if cfg.rerank and max(c.rerank_score for c in top) < cfg.min_rerank_score:
            return self._refuse(result, f"best rerank score below {cfg.min_rerank_score}", timings, t_start)

        t0 = time.perf_counter()
        out = self.generator.chat(build_messages(question, result.context, history))
        timings["generate"] = time.perf_counter() - t0
        result.raw_model_output = out
        if is_refusal(out):
            return self._refuse(result, "model said the sources do not answer it", timings, t_start)

        text, cited, dropped = enforce_citations(out, len(result.context))
        result.dropped_paragraphs = dropped
        if not cited:
            return self._refuse(result, "answer had no valid citation", timings, t_start)
        text, result.quote_cuts = safety.enforce_quote_limit(text, [c["text"] for c in result.context],
                                                             cfg.quote_limit)
        result.text = text
        for n in cited:
            c = result.context[n - 1]
            age = _age_days(c.get("crawl_timestamp", ""))
            result.citations.append({
                "n": n, "url": c["source_url"], "title": c["title"], "heading": c.get("heading_path", ""),
                "chunk_id": c["chunk_id"], "crawled": (c.get("crawl_timestamp") or "")[:10],
                "effective": c.get("effective_date", ""), "modified": (c.get("modified_date") or "")[:10],
                "stale": age is not None and age > config.STALE_AFTER_DAYS,
            })
        timings["total"] = time.perf_counter() - t_start
        result.timings = timings
        return result

    def _refuse(self, result, reason, timings, t_start):
        result.text, result.refused, result.refusal_reason = config.REFUSAL_TEXT, True, reason
        timings["total"] = time.perf_counter() - t_start
        result.timings = timings
        return result


def config_dict(cfg):
    return asdict(cfg)
