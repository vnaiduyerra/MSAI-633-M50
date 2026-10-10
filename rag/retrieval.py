"""Stage 7 - dense, BM25 and hybrid (Reciprocal Rank Fusion) retrieval; stage 8 - reranking."""

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import config

STOPWORDS = set("""a an and are as at be by for from has have how i if in into is it its of on or
that the their there these this to was were what when where which who why will with does do can
""".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)   # term -> [(chunk_index, tf)]
        self.lengths = []
        for n, c in enumerate(chunks):
            toks = tokenize(c["text"])
            self.lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((n, tf))
        self.N = len(chunks)
        self.avgdl = sum(self.lengths) / max(self.N, 1)

    def search(self, query, k, page_type=None):
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.N - len(plist) + 0.5) / (len(plist) + 0.5))
            for n, tf in plist:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[n] / max(self.avgdl, 1e-9))
                scores[n] += idf * tf * (self.k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        if page_type:
            ranked = [(n, s) for n, s in ranked if self.chunks[n]["page_type"] == page_type]
        return [(self.chunks[n]["chunk_id"], s) for n, s in ranked[:k]]


def rrf_fuse(ranked_lists, k=config.RRF_K):
    fused = defaultdict(float)
    for lst in ranked_lists:
        for rank, (cid, _) in enumerate(lst, start=1):
            fused[cid] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


@dataclass
class Candidate:
    chunk: dict
    retrieval_rank: int
    retrieval_score: float
    dense_score: float = None
    bm25_score: float = None
    rerank_score: float = None
    blocked: str = None  # set when the injection filter withholds this chunk

    @property
    def chunk_id(self):
        return self.chunk["chunk_id"]


@dataclass
class Retrieved:
    pool: list                    # candidates in retrieval order (pre-rerank)
    ranked: list                  # candidates in final order (post-rerank if enabled)
    timings: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, chunks, store=None, embedder=None):
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.bm25 = BM25(chunks)
        self.store = store
        self.embedder = embedder

    def dense(self, query, k, page_type=None):
        where = {"page_type": page_type} if page_type else None
        return self.store.query(self.embedder.encode_query(query), k, where)

    def retrieve(self, query, mode, k=config.POOL_K, page_type=None):
        dense = self.dense(query, k, page_type) if mode in ("dense", "hybrid") else []
        sparse = self.bm25.search(query, k, page_type) if mode in ("bm25", "hybrid") else []
        if mode == "dense":
            ranked = dense
        elif mode == "bm25":
            ranked = sparse
        elif mode == "hybrid":
            ranked = rrf_fuse([dense, sparse])[:k]
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        d, s = dict(dense), dict(sparse)
        return [Candidate(self.by_id[cid], r, score, d.get(cid), s.get(cid))
                for r, (cid, score) in enumerate(ranked, start=1) if cid in self.by_id]


class Reranker:
    def __init__(self, name=config.RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.name = name
        self.model = CrossEncoder(name, max_length=512)

    def score(self, query, texts):
        if not texts:
            return []
        import numpy as np

        raw = np.asarray(self.model.predict([(query, t) for t in texts]), dtype="float64").reshape(-1)
        if raw.min() < 0 or raw.max() > 1:   # logits -> probabilities so the threshold is model-agnostic
            raw = 1 / (1 + np.exp(-raw))
        return raw.tolist()


def rerank(reranker, query, pool):
    t0 = time.perf_counter()
    scores = reranker.score(query, [c.chunk["text"] for c in pool])
    for c, s in zip(pool, scores):
        c.rerank_score = float(s)
    ranked = sorted(pool, key=lambda c: -c.rerank_score)
    return ranked, time.perf_counter() - t0
