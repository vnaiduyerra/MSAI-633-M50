"""Central settings. Every value can be overridden with an environment variable."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                  # raw bytes + manifest (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"      # normalized docs + chunk files
INDEX_DIR = DATA_DIR / "index"              # vector DB + embedding cache (git-ignored)
PLANTED_DIR = DATA_DIR / "planted"          # prompt-injection test page
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
REPORTS_DIR = ROOT / "reports"
GENERATED_DIR = REPORTS_DIR / "generated"
CRAWL_CONFIG = ROOT / "crawl_config.json"

DOCS_FILE = PROCESSED_DIR / "docs.jsonl"
INDEX_STATE_FILE = INDEX_DIR / "index_state.json"


def chunks_file(strategy: str) -> Path:
    return PROCESSED_DIR / f"chunks_{strategy}.jsonl"


def _env(name, default):
    return os.environ.get(name, default)


# Hugging Face models (see reports/00_proposal.md for why these were chosen)
EMBED_MODEL = _env("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge v1.5 models retrieve better when the query (not the passage) carries this prefix
EMBED_QUERY_PREFIX = _env("RAG_EMBED_QUERY_PREFIX",
                          "Represent this sentence for searching relevant passages: ")
RERANK_MODEL = _env("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
LLM_MODEL = _env("RAG_LLM", "Qwen/Qwen2.5-1.5B-Instruct")

# Chunking (sizes in words; token distributions are measured and reported)
FIXED_WORDS = int(_env("RAG_FIXED_WORDS", 180))
FIXED_OVERLAP = int(_env("RAG_FIXED_OVERLAP", 40))
STRUCT_MAX_WORDS = int(_env("RAG_STRUCT_MAX_WORDS", 300))
TABLE_MAX_ROWS = int(_env("RAG_TABLE_MAX_ROWS", 40))

# Retrieval / reranking / generation
POOL_K = int(_env("RAG_POOL_K", 20))
TOP_K = int(_env("RAG_TOP_K", 5))
RRF_K = int(_env("RAG_RRF_K", 60))
MIN_RERANK_SCORE = float(_env("RAG_MIN_RERANK_SCORE", 0.10))  # tune on the dev split
MAX_QUOTE_WORDS = 25
MAX_NEW_TOKENS = int(_env("RAG_MAX_NEW_TOKENS", 320))
STALE_AFTER_DAYS = int(_env("RAG_STALE_AFTER_DAYS", 365))

REFUSAL_TEXT = "I don't have that information in my sources."
