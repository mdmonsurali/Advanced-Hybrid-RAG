"""Render the orchestrator's SSE stream as Chainlit steps.

One parent step ("RAG pipeline") with a child step per pipeline event:

    3 · Understanding   4 · Transformation   5 · Routing   6 · Retrieval
    7 · Reranking       8 · Compression      9 · Generation  10 · Validation

A validation retry shows up as a second attempt ("↻ attempt 2 · …"). The
final `answer` event becomes the chat message with clickable [n] citations.
"""

from __future__ import annotations

import json

import chainlit as cl

from chat import backend
from chat.citations import source_elements, sources_footer

_TITLES = {
    "understanding": "3 · Understanding",
    "transformation": "4 · Transformation",
    "routing": "5 · Routing",
    "retrieved": "6 · Hybrid retrieval",
    "reranked": "7 · Reranking",
    "compressed": "8 · Compression",
    "generating": "9 · Generation",
    "validated": "10 · Validation",
}

_BADGES = {
    "accept": "✅ Grounded",
    "accept_low_confidence": "⚠️ Low confidence",
    "clarify": "❓ Needs clarification",
    "insufficient": "🔍 Not enough evidence",
    "blocked": "⛔ Blocked by safety check",
}


def _bullets(pairs: list[tuple[str, object]]) -> str:
    return "\n".join(f"- **{k}:** {v}" for k, v in pairs if v not in (None, "", [], {}))


def _render(event: str, d: dict) -> str:
    if event == "understanding":
        f = d.get("filters") or {}
        filters = ", ".join(f"{k}={v}" for k, v in f.items() if v)
        return _bullets([("Intent", d.get("intent")), ("Question", d.get("standalone_question")),
                         ("Entities", ", ".join(d.get("entities") or [])), ("Filters", filters),
                         ("Language", d.get("language"))])
    if event == "transformation":
        out = _bullets([("Rewritten", d.get("rewritten")),
                        ("Expansions", "; ".join(d.get("expansions") or []))])
        if d.get("sub_questions"):
            out += "\n- **Sub-questions:**\n" + "\n".join(f"  1. {q}" for q in d["sub_questions"])
        if d.get("hyde_passage"):
            out += f"\n- **HyDE passage:** _{d['hyde_passage']}_"
        return out
    if event == "routing":
        return _bullets([("Routes", " + ".join(d.get("routes") or [])),
                         ("Decided by", d.get("decided_by")), ("Reason", d.get("reason")),
                         ("MCP tool results", d.get("external_evidence") or None)])
    if event == "retrieved":
        rows = [f"| {r} | {d.get('per_route', {}).get(r, 0)} | {s} |"
                for r, s in (d.get("route_status") or {}).items()]
        table = "| Route | Candidates | Status |\n|---|---|---|\n" + "\n".join(rows) if rows else ""
        extra = " (broadened retry: filters dropped, more candidates)" if d.get("broaden") else ""
        return f"{d.get('candidates', 0)} candidates{extra}\n\n{table}"
    if event == "reranked":
        rows = [f"| {t.get('score')} | {t.get('route')} | {t.get('source')}"
                f"{' p.' + str(t['page']) if t.get('page') else ''} |" for t in d.get("top") or []]
        table = "| Score | Route | Source |\n|---|---|---|\n" + "\n".join(rows) if rows else ""
        return f"Kept **{d.get('kept', 0)}** passages\n\n{table}"
    if event == "compressed":
        return _bullets([("Evidence blocks", d.get("evidence")), ("Tokens", d.get("tokens")),
                         ("Dropped", d.get("dropped")), ("Merged", d.get("merged")),
                         ("Extractively compressed", d.get("extracted"))])
    if event == "generating":
        return "Direct answer (no retrieval)" if d.get("direct") else \
            f"Answering from {d.get('evidence', 0)} evidence blocks"
    if event == "validated":
        out = _bullets([("Decision", d.get("action")),
                        ("Groundedness", f"{round(100 * d.get('grounded_score', 0))}%"),
                        ("Citations valid", "yes" if d.get("citation_ok") else "no"),
                        ("PII redacted", d.get("pii_masked") or None),
                        ("Reason", d.get("reason"))])
        if d.get("unsupported_claims"):
            out += "\n- **Unsupported claims:**\n" + "\n".join(f"  - {c}" for c in d["unsupported_claims"])
        return out
    return f"```json\n{json.dumps(d, indent=2, ensure_ascii=False)}\n```"


def _footer(final: dict) -> str:
    v = final.get("verdict") or {}
    bits = [_BADGES.get(final.get("action"), final.get("action", ""))]
    if v.get("grounded_score") is not None and final.get("action") != "clarify":
        bits.append(f"groundedness {round(100 * v['grounded_score'])}%")
    if final.get("routes"):
        bits.append(" + ".join(final["routes"]))
    if v.get("pii_masked"):
        bits.append(f"{v['pii_masked']} PII redacted")
    if final.get("attempts", 1) > 1:
        bits.append(f"{final['attempts']} attempts")
    bits.append(f"{final.get('elapsed_sec', '?')}s")
    return " · ".join(b for b in bits if b)


async def run(question: str, history: list[dict], options: dict, headers: dict,
              show_steps: bool = True) -> dict | None:
    """Stream one question through the pipeline; returns the final answer."""
    final: dict | None = None
    attempt = 0
    async with cl.Step(name="RAG pipeline", type="run") as parent:
        parent.input = question
        try:
            async for event, data in backend.stream_chat(question, history, options, headers):
                if event == "answer":
                    final = data
                    continue
                if event == "error":
                    raise backend.BackendError(data.get("service", "pipeline"),
                                               data.get("detail", "failed"))
                attempt = data.get("attempt", attempt)
                title = _TITLES.get(event, event)
                if attempt:
                    title = f"↻ attempt {attempt + 1} · {title}"
                if show_steps:
                    async with cl.Step(name=title, type="tool", parent_id=parent.id) as step:
                        step.output = _render(event, data)
                if event == "validated":
                    parent.output = f"{title}: {data.get('action')}"
        except backend.BackendError as e:
            parent.is_error = True
            parent.output = str(e)
            await cl.Message(content=f"⚠️ **{e.service}** failed: {e.detail}").send()
            return None
        if final:
            parent.output = _footer(final)

    if final is None:
        await cl.Message(content="⚠️ The pipeline ended without an answer.").send()
        return None

    citations = final.get("citations") or []
    content = final.get("answer_markdown", "").strip()
    footer = sources_footer(citations)
    # Raw HTML is disabled in Chainlit (unsafe_allow_html = false), so the
    # status line is plain Markdown italics.
    content = "\n\n".join(p for p in (content, footer, f"*{_footer(final)}*") if p)
    await cl.Message(content=content, elements=source_elements(citations)).send()
    return final
