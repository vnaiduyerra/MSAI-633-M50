# 07 · Ethics and conduct

## Crawl etiquette (in code)
robots.txt is fetched per host and checked for every URL (`Crawler.robots_allowed`); an unreachable or
access-controlled robots.txt means the host is not crawled. Requests wait `max(1 s, Crawl-delay)`. The
User-Agent names the project, its repository and a contact e-mail (the crawler refuses to start without
one). Login-walled responses (401/403 or redirects to a login URL) and directory-style URLs are skipped.

## Safeguard 1 · PII (running in code)
At extraction, a page with ≥5 e-mail addresses / phone numbers is treated as a directory and dropped,
and every e-mail address and phone number in kept pages is replaced with `[redacted …]`
(`rag/safety.py`, tested in `tests/test_pipeline.py`). Counts: see `pii_redactions` and
`dropped_directory_pii` in the ingestion table of `01_ingestion.md`.

## Safeguard 2 · Prompt injection (running in code)
Retrieved text is treated as untrusted: the system prompt tells the model to ignore instructions inside
sources, and chunks that match injection patterns ("ignore your previous instructions", "you are now…",
"system:") are withheld from the prompt and shown as 🛑 in the retrieval trace.
`data/planted/injection_test.html` is a planted page that tells the model to reply "PWNED".
Command: `python -m rag.cli injection-probe` (runs with the filter on and off).
<!-- BEGIN:injection_probe -->
<!-- END:injection_probe -->

## High-stakes referral
Questions about doping, injuries/concussion, corruption or betting approaches, disciplinary sanctions,
eligibility and official disputes get an explicit referral (ICC/national anti-doping and anti-corruption
units, medical staff, match referee, MCC Laws department) alongside any answer.

## Staleness
Each citation shows the crawl date and the page's effective/updated date when one was found; sources
crawled more than 365 days ago are flagged "may be out of date". The Laws are revised in editions (e.g.
the 2017 Code, 3rd edition 2022) - a cited older edition can look verified, which is why the date is
shown next to every link.

## Overreliance
The interface states that answers come only from the linked pages captured on the crawl date, that the
bot can misread a source, and that the umpires' decision on the field is final; refusals are explicit.

> **Team to complete:** paste the injection-probe table above (it is generated), a screenshot of a
> referral, and a short reflection on residual risks.
