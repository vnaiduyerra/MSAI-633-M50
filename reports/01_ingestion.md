# 01 · Ingestion (crawl + extraction)

Commands: `python -m rag.cli crawl --email <you@school.edu>` then `python -m rag.cli extract`.

## What the code does
- **Crawl** (`rag/crawl.py`): breadth-first from the seeds in `crawl_config.json`; every URL is checked
  against the host's robots.txt with `urllib.robotparser` before fetching; requests are spaced by
  `max(1 s, Crawl-delay)`; raw bytes are saved as `data/raw/<sha256>.<ext>` and a manifest line with URL,
  final URL, UTC timestamp, SHA-256, status, content type and Last-Modified is appended *before* any
  parsing; skipped URLs are logged with the reason (out of scope, robots, login-walled, HTTP error).
- **Extract** (`rag/extract.py`): main-content detection, removal of nav/header/footer/aside/forms and
  cookie/banner/menu elements, heading-path tracking (h1–h6), tables converted to Markdown tables and kept
  as single blocks, PDF text per page via pypdf, de-duplication on a hash of normalized text, page-type
  classification from URL rules, effective/modified date discovery (meta tags, `<time>`, "effective from …"
  phrases, Last-Modified), PII redaction.

## Numbers
<!-- BEGIN:ingestion_stats -->
<!-- END:ingestion_stats -->

## Requirements check
> **Team to complete** from the table above: ≥150 pages? ≥5 PDFs? ≥3 pages with tables? Which pages
> failed extraction and why? Name 2–3 example tables that survived intact.
