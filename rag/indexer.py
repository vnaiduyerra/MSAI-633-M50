"""Indexing and the stage-6 re-chunk migration.

The live index holds exactly one chunking strategy. Switching strategy re-chunks from the
saved normalized text (data/processed/docs.jsonl - never a re-crawl), deletes every chunk of
the previous strategy, inserts the new chunks, and then proves that no orphans remain:
every ID in the store must be an ID in the current chunk file and vice versa.
"""

import json
from datetime import datetime, timezone

import numpy as np

from . import chunking, config
from .store import chunk_metadata


def read_state(path=config.INDEX_STATE_FILE):
    return json.loads(path.read_text()) if path.exists() else {}


def write_state(state, path=config.INDEX_STATE_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def cached_embeddings(chunks, embedder, cache_dir=config.INDEX_DIR):
    """Embed chunks, reusing vectors cached by chunk ID (IDs include a content hash)."""
    safe = embedder.name.replace("/", "__")
    cache_file = cache_dir / f"emb_cache_{safe}.npz"
    cache = {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        cache = dict(zip(z["ids"].tolist(), z["vecs"]))
    todo = [c for c in chunks if c["chunk_id"] not in cache]
    if todo:
        vecs = embedder.encode_docs([c["text"] for c in todo])
        cache.update(zip([c["chunk_id"] for c in todo], vecs))
        cache_dir.mkdir(parents=True, exist_ok=True)
        ids = list(cache)
        np.savez(cache_file, ids=np.array(ids), vecs=np.stack([cache[i] for i in ids]))
    return np.stack([cache[c["chunk_id"]] for c in chunks]) if chunks else np.zeros((0, 1))


def orphan_check(store, chunks):
    expected = {c["chunk_id"] for c in chunks}
    actual = set(store.all_ids())
    strategy = chunks[0]["strategy"] if chunks else None
    return {
        "store_count": len(actual),
        "expected_count": len(expected),
        "orphans_in_store": sorted(actual - expected)[:20],
        "n_orphans": len(actual - expected),
        "n_missing": len(expected - actual),
        "other_strategy_ids": sum(1 for i in actual if strategy and not i.startswith(strategy + "-")),
    }


def build_index(store, embedder, strategy, chunks=None, state_path=config.INDEX_STATE_FILE, log=print):
    chunks = chunks if chunks is not None else chunking.load_chunks(strategy)
    state = read_state(state_path)
    previous = state.get("strategy")
    deleted_stale = 0
    if previous and previous != strategy:
        deleted_stale = store.delete(where={"strategy": previous})
        log(f"Deleted {deleted_stale} stale '{previous}' chunks")
    # Drop anything in the store that is not part of the new chunk set (e.g. changed text).
    expected = {c["chunk_id"] for c in chunks}
    leftovers = [i for i in store.all_ids() if i not in expected]
    deleted_leftover = store.delete(ids=leftovers) if leftovers else 0

    vecs = cached_embeddings(chunks, embedder)
    if len(chunks):
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), "embeddings must be L2-normalized for cosine/IP"
    store.upsert([c["chunk_id"] for c in chunks], vecs, [chunk_metadata(c) for c in chunks],
                 [c["text"] for c in chunks])
    check = orphan_check(store, chunks)
    report = {
        "at": datetime.now(timezone.utc).isoformat(),
        "previous_strategy": previous,
        "strategy": strategy,
        "deleted_stale_chunks": deleted_stale,
        "deleted_leftover_chunks": deleted_leftover,
        "inserted_or_updated": len(chunks),
        "embedding": embedder.info(),
        "store_space": getattr(store, "space", "?"),
        **check,
    }
    state.update({"strategy": strategy, "embedding": embedder.info(), "count": check["store_count"]})
    state.setdefault("migrations", []).append(report)
    write_state(state, state_path)
    log(f"Index now holds {check['store_count']} '{strategy}' chunks; orphans={check['n_orphans']}, "
        f"missing={check['n_missing']}")
    return report


def verify_metric(store, embedder, chunks, queries, k=10):
    """Compare the store's ranking with a brute-force cosine ranking for a few queries."""
    vecs = cached_embeddings(chunks, embedder)
    ids = [c["chunk_id"] for c in chunks]
    agree = []
    for q in queries:
        qv = embedder.encode_query(q)
        brute = [ids[i] for i in np.argsort(-(vecs @ qv))[:k]]
        got = [i for i, _ in store.query(qv, k)]
        agree.append(len(set(brute) & set(got)) / k)
    return float(np.mean(agree)) if agree else float("nan")


def delete_document(store, doc_id):
    """Working delete path: remove every chunk of one document."""
    return store.delete(where={"doc_id": doc_id})
