"""Stage 2 - extraction and normalization.

Reads the raw store written by the crawler and produces data/processed/docs.jsonl with,
per document: doc_id, source_url, title, heading paths (sections), page type, effective
date (when discoverable), crawl timestamp, content hash. Navigation, headers, footers and
banners are stripped, PDFs are converted to text, HTML tables are kept as Markdown tables,
and documents with identical normalized content are de-duplicated.
"""

import hashlib
import io
import json
import re
from collections import Counter

from bs4 import BeautifulSoup, Comment, Declaration, Doctype, NavigableString, Tag

from . import config, safety

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
TEXT_BLOCKS = {"p", "pre", "blockquote", "dd", "dt", "figcaption", "address", "summary"}
DROP_TAGS = ["script", "style", "noscript", "template", "nav", "footer", "aside", "form",
             "iframe", "svg", "button", "select", "input", "dialog"]
BOILERPLATE_RX = re.compile(
    r"(?i)(^|[-_ ])(cookie|consent|banner|breadcrumb|navbar|nav|menu|footer|masthead|"
    r"social|share|newsletter|subscribe|skip-link|modal|popup|promo|advert|ad-slot)([-_ ]|$)")
DATE_META = ["article:modified_time", "og:updated_time", "dateModified", "last-modified",
             "article:published_time", "datePublished", "date"]
EFFECTIVE_RX = re.compile(
    r"(?i)(?:effective|comes? into (?:force|effect)|in force|applies)\s+(?:from|on|as of)?\s*:?\s*"
    r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s+\d{4})")


def doc_id_for(url):
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def clean_text(s):
    return re.sub(r"\s+", " ", s or "").strip()


def classify_page(url, page_types, default="general"):
    for name, pattern in page_types:
        if re.search(pattern, url):
            return name
    return default


# ----------------------------------------------------------------------------- HTML
def table_to_markdown(table):
    rows = []
    for tr in table.find_all("tr"):
        cells = [clean_text(c.get_text(" ")).replace("|", "\\|") for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _is_boilerplate(tag):
    if tag.get("role") in ("navigation", "banner", "contentinfo", "search"):
        return True
    if tag.get("aria-hidden") == "true":
        return True
    attrs = " ".join(tag.get("class", [])) + " " + (tag.get("id") or "")
    return bool(BOILERPLATE_RX.search(attrs))


def _find_dates(soup, text, last_modified=""):
    found = {}
    for meta in soup.find_all("meta"):
        key = meta.get("property") or meta.get("name") or meta.get("itemprop")
        if key in DATE_META and meta.get("content"):
            found.setdefault(key, meta["content"])
    t = soup.find("time", attrs={"datetime": True})
    if t:
        found.setdefault("time", t["datetime"])
    m = EFFECTIVE_RX.search(text)
    effective = m.group(1) if m else ""
    modified = next((found[k] for k in DATE_META if k in found), found.get("time", "")) or last_modified
    return effective, modified


def parse_html(raw, url):
    soup = BeautifulSoup(raw, "lxml")
    title = ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = clean_text(og["content"])
    elif soup.title:
        title = clean_text(soup.title.get_text())
    root = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.find("article") or soup.body or soup
    for tag in root.find_all(DROP_TAGS):
        tag.decompose()
    for tag in root.find_all("header"):
        if not tag.find(list(HEADINGS)):   # keep article headers that carry the page heading
            tag.decompose()
    for tag in [t for t in root.find_all(True) if t is not root and _is_boilerplate(t)]:
        if not tag.decomposed:
            tag.decompose()

    sections = []
    stack = []  # [(level, text)]
    current = {"heading_path": [], "blocks": []}

    def flush():
        if current["blocks"]:
            sections.append({"heading_path": list(current["heading_path"]), "blocks": list(current["blocks"])})
        current["blocks"].clear()

    def add_text(s):
        s = clean_text(s)
        if s:
            current["blocks"].append({"type": "text", "text": s})

    def walk(node):
        for child in node.children:
            if isinstance(child, NavigableString):
                if not isinstance(child, (Comment, Doctype, Declaration)):
                    add_text(str(child))   # loose text directly inside a container <div>
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in HEADINGS:
                flush()
                level = HEADINGS[name]
                while stack and stack[-1][0] >= level:
                    stack.pop()
                text = clean_text(child.get_text(" "))
                if text:
                    stack.append((level, text))
                current["heading_path"] = [t for _, t in stack]
            elif name == "table":
                md = table_to_markdown(child)
                if md:
                    current["blocks"].append({"type": "table", "text": md})
            elif name in ("ul", "ol"):
                items = [clean_text(li.get_text(" ")) for li in child.find_all("li", recursive=False)]
                items = [i for i in items if i]
                if items:
                    current["blocks"].append({"type": "text", "text": "\n".join(f"- {i}" for i in items)})
            elif name in TEXT_BLOCKS:
                add_text(child.get_text(" "))
            else:
                walk(child)

    walk(root)
    flush()
    full_text = "\n".join(b["text"] for s in sections for b in s["blocks"])
    if not title:
        h1 = soup.find("h1")
        title = clean_text(h1.get_text(" ")) if h1 else url
    return title, sections, full_text, soup


# ----------------------------------------------------------------------------- PDF
def parse_pdf(raw, url):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    meta_title = ""
    try:
        meta_title = clean_text((reader.metadata or {}).get("/Title", "") or "")
    except Exception:
        pass
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        paras = [clean_text(p) for p in re.split(r"\n\s*\n", text)]
        paras = [p for p in paras if p]
        if not paras:
            # pypdf often returns one line per text row with no blank lines; keep rows together
            paras = [clean_text(text)] if clean_text(text) else []
        if paras:
            sections.append({"heading_path": [f"Page {i}"], "blocks": [{"type": "text", "text": p} for p in paras]})
    full_text = "\n".join(b["text"] for s in sections for b in s["blocks"])
    title = meta_title or (sections[0]["blocks"][0]["text"][:120] if sections else url.rsplit("/", 1)[-1])
    return title, sections, full_text


# ----------------------------------------------------------------------------- driver
def build_doc(url, crawl_timestamp, raw_sha, raw, content_type, last_modified, crawl_cfg):
    is_pdf = "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")
    if is_pdf:
        title, sections, text = parse_pdf(raw, url)
        effective, modified = "", last_modified
        m = EFFECTIVE_RX.search(text)
        if m:
            effective = m.group(1)
    else:
        title, sections, text, soup = parse_html(raw, url)
        effective, modified = _find_dates(soup, text, last_modified)
    page_type = classify_page(url, crawl_cfg.get("page_types", []), crawl_cfg.get("default_page_type", "general"))
    n_tables = sum(1 for s in sections for b in s["blocks"] if b["type"] == "table")
    # headings are part of the identity of a page, so they go into the de-duplication hash
    hash_text = clean_text(" ".join(" ".join(s["heading_path"]) + " " + " ".join(b["text"] for b in s["blocks"])
                                    for s in sections)).lower()
    return {
        "doc_id": doc_id_for(url),
        "source_url": url,
        "title": title,
        "page_type": page_type,
        "effective_date": effective,
        "modified_date": modified,
        "crawl_timestamp": crawl_timestamp,
        "raw_sha256": raw_sha,
        "content_sha256": hashlib.sha256(hash_text.encode("utf-8")).hexdigest(),
        "is_pdf": is_pdf,
        "n_tables": n_tables,
        "n_words": len(text.split()),
        "sections": sections,
    }


def redact_doc(doc):
    total = 0
    for s in doc["sections"]:
        for b in s["blocks"]:
            b["text"], n = safety.redact_pii(b["text"])
            total += n
    doc["pii_redactions"] = total
    return doc


def run_extraction(raw_dir=config.RAW_DIR, out_file=config.DOCS_FILE, crawl_cfg=None,
                   planted_dir=config.PLANTED_DIR, min_words=30, log=print):
    from .crawl import load_crawl_config

    crawl_cfg = crawl_cfg or load_crawl_config()
    manifest = raw_dir / "manifest.jsonl"
    records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs, seen_hashes, stats = [], {}, Counter()
    for rec in records:
        stats["raw_documents"] += 1
        raw = (raw_dir / rec["path"]).read_bytes()
        try:
            doc = build_doc(rec["url"], rec["fetched_at"], rec["sha256"], raw, rec["content_type"],
                            rec.get("last_modified", ""), crawl_cfg)
        except Exception as exc:
            stats["extraction_failed"] += 1
            log(f"  extraction failed for {rec['url']}: {exc}")
            continue
        if doc["n_words"] < min_words:
            stats["dropped_too_short"] += 1
            continue
        full_text = " ".join(b["text"] for s in doc["sections"] for b in s["blocks"])
        if safety.looks_like_directory(full_text):
            stats["dropped_directory_pii"] += 1
            log(f"  dropped (looks like a contact directory): {rec['url']}")
            continue
        if doc["content_sha256"] in seen_hashes:
            stats["dropped_duplicate"] += 1
            continue
        seen_hashes[doc["content_sha256"]] = doc["doc_id"]
        redact_doc(doc)
        docs.append(doc)

    # The planted prompt-injection page (ethics requirement) is added as a local test doc.
    for path in sorted(planted_dir.glob("*.html")) if planted_dir.exists() else []:
        url = f"local://planted/{path.stem}"
        doc = build_doc(url, "planted", hashlib.sha256(path.read_bytes()).hexdigest(), path.read_bytes(),
                        "text/html", "", crawl_cfg)
        doc["page_type"] = "planted_test"
        docs.append(redact_doc(doc))
        stats["planted_test_pages"] += 1

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    stats["kept_documents"] = len(docs)
    stats["kept_pdfs"] = sum(d["is_pdf"] for d in docs)
    stats["kept_pages_with_tables"] = sum(d["n_tables"] > 0 for d in docs)
    stats["pii_redactions"] = sum(d.get("pii_redactions", 0) for d in docs)
    for t, n in Counter(d["page_type"] for d in docs).items():
        stats[f"page_type:{t}"] = n
    log(f"Extraction finished: {len(docs)} docs -> {out_file}")
    return docs, dict(stats)


def load_docs(path=config.DOCS_FILE):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
