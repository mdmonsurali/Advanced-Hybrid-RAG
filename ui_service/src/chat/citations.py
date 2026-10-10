"""Citations -> Chainlit side elements.

Chainlit turns every occurrence of an element's *name* in a message into a
link that opens the element (see prepareContent in the Chainlit frontend).
Naming each source element "[n]" therefore makes the model's own [n]
citation markers clickable without rewriting the answer.
"""

from __future__ import annotations

import re

import chainlit as cl

_ROUTE_LABEL = {"docs": "Document", "gdrive": "Google Drive", "web": "Web",
                "catalog": "Document catalog", "mcp": "MCP tool"}
_SAFE_LINK = re.compile(r"^(https?://|/(?!/))", re.IGNORECASE)


def _link(c: dict) -> str | None:
    link = c.get("link") or c.get("url")
    return link if link and _SAFE_LINK.match(link) else None


def _where(c: dict) -> str:
    parts = [c.get("source") or _ROUTE_LABEL.get(c.get("route"), "Source")]
    if c.get("page"):
        parts.append(f"p. {c['page']}")
    return " · ".join(parts)


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.strip().splitlines())


def source_elements(citations: list[dict]) -> list[cl.Text]:
    elements = []
    for c in citations:
        body = [f"**{_where(c)}**", f"_{_ROUTE_LABEL.get(c.get('route'), c.get('route', ''))}_"]
        if c.get("snippet"):
            body += ["", _quote(c["snippet"])]
        link = _link(c)
        if link:
            body += ["", f"[Open source ↗]({link})"]
        elements.append(cl.Text(name=f"[{c['n']}]", content="\n".join(body), display="side"))
    return elements


def sources_footer(citations: list[dict]) -> str:
    if not citations:
        return ""
    items = [f"[{c['n']}] {_where(c)}" for c in citations]
    return "**Sources:** " + " · ".join(items)
