"""Stage 1 - polite crawler.

* robots.txt is fetched, parsed and enforced in code for every URL (urllib.robotparser)
* at most 1 request/second (or the site's Crawl-delay, whichever is slower)
* descriptive User-Agent that carries a contact e-mail
* raw response bytes are written to disk with URL, timestamp and SHA-256 *before*
  anything is parsed; link extraction happens afterwards from the saved bytes
* login-walled pages (401/403 or redirects to a login URL) and directory-style pages
  are skipped
"""

import hashlib
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup

from . import config

LOGIN_HINT = re.compile(r"(?i)/(login|log-in|signin|sign-in|sso|auth)(/|$|\?)")


def load_crawl_config(path=config.CRAWL_CONFIG):
    return json.loads(path.read_text(encoding="utf-8"))


def build_user_agent(cfg, contact_email):
    if not contact_email or "@" not in contact_email:
        raise ValueError("A contact e-mail is required for the User-Agent "
                         "(set CRAWL_CONTACT_EMAIL or pass --email).")
    return f"{cfg['user_agent_name']} (+{cfg.get('project_url', '')}; contact: {contact_email})"


def normalize_url(url):
    url, _ = urldefrag(url.strip())
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return None
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    query = f"?{p.query}" if p.query else ""
    return f"{p.scheme}://{p.netloc.lower()}{path}{query}"


def is_pdf_url(url):
    return bool(re.search(r"(?i)\.pdf(\?|$)", url))


class Crawler:
    def __init__(self, cfg, contact_email, out_dir=config.RAW_DIR, session=None, sleep=time.sleep,
                 clock=time.monotonic):
        import requests  # local import keeps the module importable for tests without network

        self.cfg = cfg
        self.user_agent = build_user_agent(cfg, contact_email)
        self.out_dir = out_dir
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self.sleep = sleep
        self.clock = clock
        self.min_delay = max(1.0, float(cfg.get("min_delay_seconds", 1.0)))
        self.timeout = cfg.get("timeout_seconds", 30)
        self.exclude = [re.compile(p) for p in cfg.get("exclude_patterns", [])]
        self.robots = {}
        self._last_request = None
        self.manifest_path = out_dir / "manifest.jsonl"
        self.log_path = out_dir / "crawl_log.jsonl"

    # ---------------------------------------------------------------- politeness
    def _throttle(self, host):
        delay = self.min_delay
        rp = self.robots.get(host)
        if rp is not None:
            crawl_delay = rp.crawl_delay(self.user_agent)
            if crawl_delay:
                delay = max(delay, float(crawl_delay))
        if self._last_request is not None:
            wait = delay - (self.clock() - self._last_request)
            if wait > 0:
                self.sleep(wait)
        self._last_request = self.clock()

    def _get(self, url):
        self._throttle(urlparse(url).netloc)
        return self.session.get(url, timeout=self.timeout, allow_redirects=True)

    def robots_for(self, url):
        p = urlparse(url)
        host = p.netloc
        if host not in self.robots:
            rp = robotparser.RobotFileParser()
            robots_url = f"{p.scheme}://{host}/robots.txt"
            try:
                resp = self._get(robots_url)
                if resp.status_code in (401, 403):
                    rp.disallow_all = True       # access-controlled robots.txt => stay out
                elif resp.status_code >= 400:
                    rp.parse([])                 # no robots.txt => everything allowed
                else:
                    rp.parse(resp.text.splitlines())
            except Exception as exc:             # unreachable robots.txt => do not crawl the host
                self._log("robots_error", robots_url, str(exc))
                rp.disallow_all = True
            self.robots[host] = rp
        return self.robots[host]

    def robots_allowed(self, url):
        return self.robots_for(url).can_fetch(self.user_agent, url)

    # ---------------------------------------------------------------- scope
    def in_scope(self, url):
        p = urlparse(url)
        host = p.netloc.lower()
        if host not in self.cfg["allowed_domains"]:
            return False
        if any(rx.search(url) for rx in self.exclude):
            return False
        if is_pdf_url(url) and self.cfg.get("allow_pdfs_anywhere_on_allowed_domains"):
            return True
        return any(p.path.startswith(pref) for pref in self.cfg["allowed_path_prefixes"])

    # ---------------------------------------------------------------- storage
    def _log(self, event, url, detail=""):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, "url": url, "detail": detail,
                                "at": datetime.now(timezone.utc).isoformat()}) + "\n")

    def save_raw(self, url, resp):
        content = resp.content
        sha = hashlib.sha256(content).hexdigest()
        ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        ext = ".pdf" if ("pdf" in ctype or is_pdf_url(url)) else ".html"
        path = self.out_dir / f"{sha}{ext}"
        if not path.exists():
            path.write_bytes(content)
        record = {
            "url": url,
            "final_url": normalize_url(resp.url) or url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "sha256": sha,
            "status": resp.status_code,
            "content_type": ctype,
            "bytes": len(content),
            "path": path.name,
            "last_modified": resp.headers.get("Last-Modified", ""),
        }
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record, path

    # ---------------------------------------------------------------- crawl loop
    def extract_links(self, base_url, raw_path):
        soup = BeautifulSoup(raw_path.read_bytes(), "lxml")
        links = []
        for a in soup.find_all("a", href=True):
            u = normalize_url(urljoin(base_url, a["href"]))
            if u:
                links.append(u)
        return links

    def run(self, max_pages=None, log=print):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        max_pages = max_pages or self.cfg.get("max_pages", 400)
        seen = set()
        if self.manifest_path.exists():  # resume: never re-fetch what is already on disk
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
                seen.add(json.loads(line)["url"])
        saved = len(seen)
        queue = deque(normalize_url(s) for s in self.cfg["seeds"])
        queued = set(queue)
        while queue and saved < max_pages:
            url = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            if not self.in_scope(url):
                self._log("out_of_scope", url)
                continue
            if not self.robots_allowed(url):
                self._log("robots_disallowed", url)
                continue
            try:
                resp = self._get(url)
            except Exception as exc:
                self._log("fetch_error", url, str(exc))
                continue
            final = normalize_url(resp.url) or url
            if resp.status_code in (401, 403) or LOGIN_HINT.search(final):
                self._log("login_walled", url, f"status={resp.status_code} final={final}")
                continue
            if resp.status_code != 200:
                self._log("http_error", url, str(resp.status_code))
                continue
            if final != url and (not self.in_scope(final) or not self.robots_allowed(final)):
                self._log("redirect_out_of_scope", url, final)
                continue
            record, path = self.save_raw(url, resp)
            saved += 1
            log(f"[{saved}/{max_pages}] {record['content_type'] or '?'} {url}")
            if path.suffix == ".html":
                for link in self.extract_links(final, path):
                    if link not in seen and link not in queued and self.in_scope(link):
                        queue.append(link)
                        queued.add(link)
        log(f"Crawl finished: {saved} documents in {self.out_dir}")
        return saved
