"""Stage 5 - vector database.

FaissStore is the real store (persistent, exact cosine search, metadata filtering, delete by id
or by metadata). NumpyStore implements the same interface in memory for the offline unit tests.
"""

import numpy as np

META_KEYS = ("strategy", "doc_id", "source_url", "title", "page_type", "heading_path",
             "effective_date", "modified_date", "crawl_timestamp", "is_table")


def chunk_metadata(c):
    meta = {}
    for k in META_KEYS:
        v = c.get(k, "")
        meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return meta


def _where_ok(meta, where):
    return not where or all(meta.get(k) == v for k, v in where.items())


class NumpyStore:
    space = "cosine"

    def __init__(self):
        self.ids, self.vecs, self.metas, self.docs = [], [], [], []

    def upsert(self, ids, embeddings, metadatas, documents):
        pos = {i: n for n, i in enumerate(self.ids)}
        for i, e, m, d in zip(ids, embeddings, metadatas, documents):
            if i in pos:
                n = pos[i]
                self.vecs[n], self.metas[n], self.docs[n] = np.asarray(e, dtype="float32"), m, d
            else:
                pos[i] = len(self.ids)
                self.ids.append(i)
                self.vecs.append(np.asarray(e, dtype="float32"))
                self.metas.append(m)
                self.docs.append(d)

    def query(self, embedding, k, where=None):
        q = np.asarray(embedding, dtype="float32")
        hits = []
        for i, v, m in zip(self.ids, self.vecs, self.metas):
            if _where_ok(m, where):
                sim = float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q) + 1e-12))
                hits.append((i, sim))
        hits.sort(key=lambda x: -x[1])
        return hits[:k]

    def delete(self, ids=None, where=None):
        drop = set(ids or [])
        keep = [n for n, (i, m) in enumerate(zip(self.ids, self.metas))
                if not (i in drop or (where and _where_ok(m, where)))]
        removed = len(self.ids) - len(keep)
        self.ids = [self.ids[n] for n in keep]
        self.vecs = [self.vecs[n] for n in keep]
        self.metas = [self.metas[n] for n in keep]
        self.docs = [self.docs[n] for n in keep]
        return removed

    def all_ids(self, where=None):
        return [i for i, m in zip(self.ids, self.metas) if _where_ok(m, where)]

    def count(self):
        return len(self.ids)


class FaissStore:
    """Exact inner-product search (IndexIDMap2 over IndexFlatIP) on L2-normalized vectors, i.e. cosine.

    String chunk IDs map to int64 FAISS IDs; metadata lives in a JSON sidecar next to the index so
    filtering (page_type, strategy, doc_id) and deletes by id or metadata work.
    """

    space = "cosine"

    def __init__(self, path):
        import json
        from pathlib import Path

        import faiss

        self.faiss, self.json = faiss, json
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_file, self.meta_file = self.dir / "index.faiss", self.dir / "meta.json"
        self.index, self.int_of, self.meta, self.next_id = None, {}, {}, 0
        if self.index_file.exists() and self.meta_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            saved = json.loads(self.meta_file.read_text())
            self.int_of, self.meta, self.next_id = saved["int_of"], saved["meta"], saved["next_id"]
        self.str_of = {v: k for k, v in self.int_of.items()}

    def _save(self):
        if self.index is not None:
            self.faiss.write_index(self.index, str(self.index_file))
        self.meta_file.write_text(self.json.dumps({"int_of": self.int_of, "meta": self.meta, "next_id": self.next_id}))

    def _remove(self, str_ids):
        ints = [self.int_of.pop(i) for i in str_ids if i in self.int_of]
        for n in ints:
            sid = self.str_of.pop(n)
            self.meta.pop(sid, None)
        if ints and self.index is not None:
            self.index.remove_ids(np.asarray(ints, dtype="int64"))
        return len(ints)

    def upsert(self, ids, embeddings, metadatas, documents=None):
        vecs = np.ascontiguousarray(np.asarray(embeddings, dtype="float32"))
        if len(ids) == 0:
            return
        if self.index is None:
            self.index = self.faiss.IndexIDMap2(self.faiss.IndexFlatIP(vecs.shape[1]))
        self._remove([i for i in ids if i in self.int_of])
        ints = np.arange(self.next_id, self.next_id + len(ids), dtype="int64")
        self.next_id += len(ids)
        self.index.add_with_ids(vecs, ints)
        for sid, n, m in zip(ids, ints.tolist(), metadatas):
            self.int_of[sid], self.str_of[n], self.meta[sid] = n, sid, m
        self._save()

    def query(self, embedding, k, where=None):
        n = self.count()
        if n == 0:
            return []
        q = np.asarray(embedding, dtype="float32").reshape(1, -1)
        scores, ints = self.index.search(q, n if where else min(k, n))  # exact search; filter afterwards
        hits = []
        for s, i in zip(scores[0], ints[0]):
            sid = self.str_of.get(int(i))
            if sid is not None and _where_ok(self.meta[sid], where):
                hits.append((sid, float(s)))
                if len(hits) == k:
                    break
        return hits

    def delete(self, ids=None, where=None):
        drop = set(ids or [])
        if where:
            drop |= {sid for sid, m in self.meta.items() if _where_ok(m, where)}
        removed = self._remove(list(drop))
        self._save()
        return removed

    def all_ids(self, where=None):
        return [sid for sid, m in self.meta.items() if _where_ok(m, where)]

    def count(self):
        return len(self.meta)
