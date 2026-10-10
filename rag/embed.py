"""Stage 4 - embeddings (sentence-transformers, L2-normalized, cosine metric)."""

import numpy as np

from . import config


def model_revision(name):
    try:
        from huggingface_hub import model_info

        return model_info(name).sha or "unknown"
    except Exception:
        return "unknown"


class Embedder:
    normalized = True
    metric = "cosine"

    def __init__(self, name=config.EMBED_MODEL, query_prefix=config.EMBED_QUERY_PREFIX):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.query_prefix = query_prefix if "bge" in name.lower() else ""
        self.model = SentenceTransformer(name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.revision = model_revision(name)

    def encode_docs(self, texts, batch_size=64):
        return np.asarray(self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                            show_progress_bar=len(texts) > 200), dtype="float32")

    def encode_query(self, text):
        return np.asarray(self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0],
                          dtype="float32")

    def info(self):
        return {"model": self.name, "revision": self.revision, "dimension": self.dim,
                "l2_normalized": self.normalized, "distance_metric": self.metric,
                "query_prefix": self.query_prefix}
