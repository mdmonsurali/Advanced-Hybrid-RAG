"""Web tool safety (SSRF guard, ad filtering) and plain-text chunking."""

from __future__ import annotations

import itertools
import socket

import pytest
from chunking_techniques.text import chunk_text
from mcp_servers import websearch_providers as web


def resolves_to(monkeypatch, ip: str) -> None:
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda host, port: [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]
    )


@pytest.mark.parametrize(
    "ip", ["127.0.0.1", "10.0.0.5", "172.18.0.3", "192.168.1.1", "169.254.169.254", "::1"]
)
def test_fetch_refuses_internal_addresses(monkeypatch, ip):
    resolves_to(monkeypatch, ip)
    with pytest.raises(ValueError, match="non-public"):
        web._assert_public("http://rag_service:8003/health")


def test_fetch_allows_public_addresses(monkeypatch):
    resolves_to(monkeypatch, "93.184.215.14")
    web._assert_public("https://example.com/page")


def test_fetch_needs_a_host():
    with pytest.raises(ValueError, match="no host"):
        web._assert_public("file:///etc/passwd")


def test_search_drops_ads_and_empty_urls(monkeypatch):
    results = [
        {"title": "ad", "url": "https://www.bing.com/aclick?ld=1", "snippet": ""},
        {"title": "empty", "url": "", "snippet": ""},
        {"title": "real", "url": "https://example.com/a", "snippet": ""},
        {"title": "real 2", "url": "https://example.org/b", "snippet": ""},
    ]
    monkeypatch.setattr(web, "WEBSEARCH_PROVIDER", "ddgs")
    monkeypatch.setitem(web._PROVIDERS, "ddgs", lambda q, limit, days: results)
    assert [r["title"] for r in web.search("q", limit=1)] == ["real"]


def test_paid_providers_need_an_api_key(monkeypatch):
    monkeypatch.setattr(web, "WEBSEARCH_PROVIDER", "tavily")
    monkeypatch.setattr(web, "WEBSEARCH_API_KEY", "")
    with pytest.raises(RuntimeError, match="WEBSEARCH_API_KEY"):
        web.search("q")


@pytest.mark.parametrize(
    ("days", "limit"), [(None, None), (1, "d"), (7, "w"), (30, "m"), (90, "y")]
)
def test_recency_maps_to_search_time_limits(days, limit):
    assert web._timelimit(days) == limit


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.\n\nSecond paragraph.") == ["Hello world.\nSecond paragraph."]


def test_long_text_is_split_with_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_text(text, max_chars=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # The start of each chunk repeats the end of the previous one.
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.split()[0] in prev


def test_run_on_text_without_sentence_breaks_is_still_split():
    chunks = chunk_text("x" * 1000, max_chars=300, overlap=0)
    assert [len(c) for c in chunks] == [300, 300, 300, 100]
