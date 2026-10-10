"""Guardrails used at ingestion time and answer time.

Safeguard 1 - PII: directory-like pages are dropped, e-mails / phone numbers are redacted.
Safeguard 2 - prompt injection: retrieved chunks that contain instructions aimed at the
              model are withheld from the prompt and flagged in the retrieval trace.
Also: high-stakes referral to a qualified human, and the 25-word quote limit.
"""

import re

# ----------------------------------------------------------------------------- PII
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# International (+44 ...) or trunk-prefixed (020 ...) numbers; plain digit runs such as
# "2022 2023 2024" in a table of seasons must not be mistaken for phone numbers.
PHONE_RE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?(?:\(0\)[\s.-]?)?|0)\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\w)")
DIRECTORY_MIN_CONTACTS = 5  # a page listing this many personal contacts is treated as a directory


def count_contacts(text):
    return len(EMAIL_RE.findall(text)) + len(PHONE_RE.findall(text))


def looks_like_directory(text):
    return count_contacts(text) >= DIRECTORY_MIN_CONTACTS


def redact_pii(text):
    """Return (redacted_text, number_of_redactions)."""
    n = 0

    def _sub(repl):
        def inner(_m):
            nonlocal n
            n += 1
            return repl
        return inner

    text = EMAIL_RE.sub(_sub("[redacted e-mail]"), text)
    text = PHONE_RE.sub(_sub("[redacted phone]"), text)
    return text, n


# ----------------------------------------------------------------------------- injection
INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(your |my )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(your |previous |prior |above )*(instructions|rules|prompt)",
    r"forget (all |your )?(previous |prior )?(instructions|rules)",
    r"(reveal|print|show|repeat) (your |the )?(system prompt|hidden instructions)",
    r"you are now (a|an|in) ",
    r"new instructions?:",
    r"\b(system|assistant)\s*:\s*",
    r"respond (only )?with ['\"]",
]
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)


def find_injection(text):
    m = INJECTION_RE.search(text)
    return m.group(0) if m else None


# ----------------------------------------------------------------------------- referral
# High-stakes cricket topics where a chatbot answer must not be the last word.
REFERRALS = [
    (r"\b(dop(e|ing)|banned substance|wada|drug test|therapeutic use)",
     "Anti-doping questions should go to your national anti-doping organisation or the "
     "ICC Anti-Doping Unit, not a chatbot."),
    (r"\b(concussion|head injur|injur(y|ed)|medical|bleeding|unconscious|heat ?stroke)",
     "For any injury or medical situation, stop and consult a qualified medical professional "
     "or the match medical officer."),
    (r"\b(match[- ]fixing|spot[- ]fixing|corrupt|betting|bribe|approach(ed)? by)",
     "Report suspected corruption or betting approaches to the ICC Anti-Corruption Unit or "
     "your board's integrity officer."),
    (r"\b(appeal (against|a) (ban|suspension|sanction)|disciplinary|code of conduct|suspension|ban(ned)? for)",
     "Disciplinary outcomes are decided by the match referee and the relevant governing body; "
     "contact them for a binding ruling."),
    (r"\b(eligib|qualif(y|ication) to play|residency rule|registration)",
     "Player eligibility is ruled on by the governing body (ICC or national board); confirm "
     "with their registration office."),
    (r"\b(dispute|protest|overturn|official ruling|binding decision)",
     "On the field, the umpires' decision is final; for an official interpretation contact "
     "the match referee or MCC Laws department (laws@mcc.org.uk)."),
]
_REFERRAL_RES = [(re.compile(p, re.IGNORECASE), msg) for p, msg in REFERRALS]


def referral_for(question):
    for rx, msg in _REFERRAL_RES:
        if rx.search(question):
            return msg
    return None


# ----------------------------------------------------------------------------- quote limit
_WORD_RE = re.compile(r"\S+")


def _norm_word(w):
    return re.sub(r"[^\w]", "", w.lower())


def enforce_quote_limit(answer, sources, max_words=25):
    """Cut any run of more than ``max_words`` consecutive words copied from a source.

    Returns (new_answer, number_of_runs_cut).
    """
    n = max_words + 1
    grams = set()
    for src in sources:
        words = [_norm_word(w) for w in _WORD_RE.findall(src)]
        words = [w for w in words if w]
        for i in range(len(words) - n + 1):
            grams.add(tuple(words[i:i + n]))
    if not grams:
        return answer, 0

    tokens = [(m.start(), m.end(), _norm_word(m.group())) for m in _WORD_RE.finditer(answer)]
    idx = [i for i, t in enumerate(tokens) if t[2]]          # tokens that are real words
    covered = [False] * len(tokens)
    for j in range(len(idx) - n + 1):
        window = tuple(tokens[idx[k]][2] for k in range(j, j + n))
        if window in grams:
            for k in range(j, j + n):
                covered[idx[k]] = True

    # Build the output: inside each covered run keep the first max_words words, drop the rest.
    out, cuts, pos, i = [], 0, 0, 0
    while i < len(tokens):
        if not covered[i]:
            i += 1
            continue
        j = i
        while j < len(tokens) and covered[j]:
            j += 1
        run = list(range(i, j))
        keep = run[:max_words]
        out.append(answer[pos:tokens[keep[-1]][1]] + " …")
        pos = tokens[run[-1]][1]
        cuts += 1
        i = j
    out.append(answer[pos:])
    return "".join(out), cuts
