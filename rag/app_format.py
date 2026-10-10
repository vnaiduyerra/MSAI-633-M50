"""Markdown rendering of answers and retrieval traces (used by the Gradio app and the CLI)."""

from . import config


def format_answer(ans):
    parts = [ans.text]
    if ans.refused:
        parts[0] = f"🚫 **{config.REFUSAL_TEXT}**\n\nTry rephrasing, or ask about a specific Law."
    if ans.citations:
        lines = []
        for c in ans.citations:
            date_bits = [f"crawled {c['crawled']}"] if c["crawled"] else []
            if c["effective"]:
                date_bits.append(f"effective {c['effective']}")
            elif c["modified"]:
                date_bits.append(f"page updated {c['modified']}")
            stale = " ⚠️ *may be out of date - check the live page*" if c["stale"] else ""
            label = c["heading"] or c["title"]
            lines.append(f"[{c['n']}] [{label}]({c['url']}) · {', '.join(date_bits)}{stale}")
        parts.append("**Sources**\n" + "\n".join(f"- {l}" for l in lines))
    if ans.referral:
        parts.append(f"🧑‍⚖️ **Please check with a person:** {ans.referral}")
    return "\n\n".join(parts)


def _fmt(x):
    return "–" if x is None else f"{x:.3f}"


def format_trace(ans):
    if not ans.pool:
        return "_No retrieval for this answer._"
    lines = [f"**Retrieval query:** {ans.retrieval_query}",
             f"**Timings:** " + ", ".join(f"{k} {v:.2f}s" for k, v in ans.timings.items()), ""]
    if ans.refusal_reason:
        lines.append(f"**Refusal reason:** {ans.refusal_reason}\n")
    used = {c["chunk_id"] for c in ans.context}
    lines.append("| Final rank | Pre-rerank rank | Retrieval score | Dense | BM25 | Rerank | Used | Source |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for n, c in enumerate(ans.ranked, start=1):
        status = "🛑 blocked (injection)" if c.blocked else ("✅" if c.chunk_id in used else "")
        src = f"[{(c.chunk.get('heading_path') or c.chunk['title'])[:60]}]({c.chunk['source_url']})"
        lines.append(f"| {n} | {c.retrieval_rank} | {_fmt(c.retrieval_score)} | {_fmt(c.dense_score)} | "
                     f"{_fmt(c.bm25_score)} | {_fmt(c.rerank_score)} | {status} | {src} |")
    lines.append("\n**Chunks given to the model**\n")
    for n, ch in enumerate(ans.context, start=1):
        snippet = ch["text"][:500].replace("\n", " ")
        lines.append(f"**[{n}]** `{ch['chunk_id']}`  \n{snippet}…\n")
    return "\n".join(lines)
