"""Web search backends for the websearch MCP server.

WEBSEARCH_PROVIDER:
    ddgs    DuckDuckGo & friends via the `ddgs` package (default, no API key)
    tavily  https://tavily.com   (WEBSEARCH_API_KEY)
    brave   Brave Search API     (WEBSEARCH_API_KEY)
"""

from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger("rag_service.websearch")

WEBSEARCH_PROVIDER = os.getenv("WEBSEARCH_PROVIDER", "ddgs").strip().lower()
WEBSEARCH_API_KEY = os.getenv("WEBSEARCH_API_KEY", "")
WEBSEARCH_TIMEOUT = float(os.getenv("WEBSEARCH_TIMEOUT", "15"))
FETCH_USER_AGENT = os.getenv(
    "WEBSEARCH_USER_AGENT", "Mozilla/5.0 (compatible; AdvancedHybridRAG/1.0; +https://github.com/)"
)


def _timelimit(recency_days: int | None) -> str | None:
    if not recency_days:
        return None
    if recency_days <= 1:
        return "d"
    if recency_days <= 7:
        return "w"
    if recency_days <= 31:
        return "m"
    return "y"


def _ddgs(query: str, limit: int, recency_days: int | None) -> list[dict]:
    from ddgs import DDGS

    hits = DDGS().text(query, max_results=limit, timelimit=_timelimit(recency_days))
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": h.get("body", ""),
            "published": h.get("date"),
        }
        for h in hits
    ]


def _tavily(query: str, limit: int, recency_days: int | None) -> list[dict]:
    body = {"query": query, "max_results": limit}
    if recency_days:
        body["days"] = recency_days
    resp = httpx.post(
        "https://api.tavily.com/search",
        json=body,
        timeout=WEBSEARCH_TIMEOUT,
        headers={"Authorization": f"Bearer {WEBSEARCH_API_KEY}"},
    )
    resp.raise_for_status()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "published": r.get("published_date"),
        }
        for r in resp.json().get("results", [])
    ]


def _brave(query: str, limit: int, recency_days: int | None) -> list[dict]:
    params = {"q": query, "count": limit}
    freshness = {"d": "pd", "w": "pw", "m": "pm", "y": "py"}.get(_timelimit(recency_days) or "")
    if freshness:
        params["freshness"] = freshness
    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params=params,
        timeout=WEBSEARCH_TIMEOUT,
        headers={"X-Subscription-Token": WEBSEARCH_API_KEY, "Accept": "application/json"},
    )
    resp.raise_for_status()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
            "published": r.get("page_age"),
        }
        for r in resp.json().get("web", {}).get("results", [])
    ]


_PROVIDERS = {"ddgs": _ddgs, "tavily": _tavily, "brave": _brave}

# Sponsored-result redirectors some engines mix into organic results.
_AD_MARKERS = (
    "bing.com/aclick",
    "duckduckgo.com/y.js",
    "googleadservices.com",
    "doubleclick.net",
    "/aclk?",
)


def _is_ad(url: str) -> bool:
    return any(m in url for m in _AD_MARKERS)


def search(query: str, limit: int = 5, recency_days: int | None = None) -> list[dict]:
    provider = _PROVIDERS.get(WEBSEARCH_PROVIDER)
    if provider is None:
        raise ValueError(f"Unknown WEBSEARCH_PROVIDER: {WEBSEARCH_PROVIDER}")
    if WEBSEARCH_PROVIDER != "ddgs" and not WEBSEARCH_API_KEY:
        raise RuntimeError(f"WEBSEARCH_API_KEY is required for {WEBSEARCH_PROVIDER}")
    return [r for r in provider(query, limit, recency_days) if r["url"] and not _is_ad(r["url"])][
        :limit
    ]


def _assert_public(url: str) -> None:
    """Refuse URLs that resolve to loopback / private / link-local addresses,
    so the tool can't be used to reach services inside the docker network."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    host = urlparse(url).hostname
    if not host:
        raise ValueError("url has no host")
    for info in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"refusing to fetch non-public address {ip}")


def fetch_page(url: str, max_chars: int = 15000) -> dict:
    import trafilatura

    _assert_public(url)
    with httpx.Client(
        timeout=WEBSEARCH_TIMEOUT, follow_redirects=False, headers={"User-Agent": FETCH_USER_AGENT}
    ) as client:
        resp = client.get(url)
        for _ in range(5):  # follow redirects, re-checking each hop
            if not resp.is_redirect or resp.next_request is None:
                break
            url = str(resp.next_request.url)
            _assert_public(url)
            resp = client.get(url)
    resp.raise_for_status()
    text = (
        trafilatura.extract(resp.text, url=url, include_tables=True, include_comments=False) or ""
    )
    return {"url": str(resp.url), "text": text[:max_chars], "truncated": len(text) > max_chars}
