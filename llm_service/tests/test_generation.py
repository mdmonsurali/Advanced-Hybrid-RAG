"""STEP 8 compression, STEP 9 answer building and STEP 5 LLM routing."""

from __future__ import annotations

import pytest
from conftest import evidence, plan
from generation import answer, compression, query_analysis
from llm_client.client import LLMOutputError
from schemas import Evidence

pytestmark = pytest.mark.anyio


# ── STEP 8 ───────────────────────────────────────────────────────────────


def ev(id, text, score=0.5, **kw) -> Evidence:
    return Evidence(id=id, route="docs", text=text, score=score, **kw)


def test_weak_chunks_are_dropped_relative_to_the_best():
    kept = compression._drop_weak([ev("a", "x", 0.9), ev("b", "y", 0.5), ev("c", "z", 0.1)])
    assert [e.id for e in kept] == ["a", "b"]  # 0.1 < 0.9 * 0.25


def test_near_duplicates_are_removed():
    text = "refunds are accepted within thirty days of the purchase date"
    kept = compression._dedupe(
        [ev("a", text, 0.9), ev("b", text + ".", 0.8), ev("c", "other", 0.7)]
    )
    assert [e.id for e in kept] == ["a", "c"]


def test_adjacent_chunks_of_one_page_are_merged():
    chunks = [
        ev("a", "Table 1\n| h |\n| row 1 |", 0.4, doc_id="d", page=1, chunk_index=0),
        ev("b", "Table 1\n| h |\n| row 2 |", 0.9, doc_id="d", page=1, chunk_index=1),
        ev("c", "elsewhere", 0.5, doc_id="d", page=2, chunk_index=2),
    ]
    merged, count = compression._merge_adjacent(chunks)
    assert count == 1
    first = next(e for e in merged if e.id == "a")
    assert first.score == 0.9
    assert first.text == "Table 1\n| h |\n| row 1 |\n| row 2 |"  # repeated header dropped


async def test_compress_counts_what_it_removed(llm):
    out = await compression.compress(plan(), evidence(3) + [ev("weak", "noise", 0.01)])
    assert out.dropped == 1
    assert [e.id for e in out.evidence] == ["e0", "e1", "e2"]
    assert out.tokens > 0
    assert llm.calls == []  # everything fit; no extraction needed


async def test_long_chunks_over_budget_are_extracted(llm, monkeypatch):
    monkeypatch.setattr(compression, "LLM_CONTEXT_LEN", 1700)
    monkeypatch.setattr(compression, "LONG_CHUNK_TOKENS", 50)
    long_text = "word " * 2000  # ~2500 tokens with the 4-chars estimate
    llm.queue.append({"relevant_text": "the relevant sentence"})
    out = await compression.compress(plan(), [ev("big", long_text, 0.9)])
    assert out.extracted == 1
    assert out.evidence[0].text == "the relevant sentence"


# ── STEP 9 ───────────────────────────────────────────────────────────────


def test_cited_numbers():
    assert answer.cited_numbers("A [1], B [3][1] and [x].") == [1, 3, 1]


def test_citations_skip_unknown_numbers_and_trim_snippets(monkeypatch):
    monkeypatch.setattr(answer, "SNIPPET_CHARS", 20)
    evs = [ev("a", "one two three four five six seven", doc_id="d", page=4)]
    cites = answer._citations([1, 1, 5], evs)
    assert len(cites) == 1
    assert cites[0].evidence_id == "a"
    assert cites[0].page == 4
    assert cites[0].snippet.endswith(" …")
    assert len(cites[0].snippet) <= 22


async def test_generate_without_evidence_does_not_call_the_llm(llm):
    out = await answer.generate(plan(), [], direct=False)
    assert out.insufficient_evidence is True
    assert llm.calls == []


async def test_generate_builds_citations_from_the_text(llm):
    llm.queue.append({"answer_markdown": "Within 30 days [2].", "confidence": 0.9})
    out = await answer.generate(plan(), evidence(2), direct=False)
    assert [c.n for c in out.citations] == [2]
    assert out.citations[0].evidence_id == "e1"


async def test_generate_reports_schema_failures(llm):
    llm.queue.append(LLMOutputError("bad json twice"))
    out = await answer.generate(plan(), evidence(), direct=False)
    assert out.schema_ok is False


# ── STEP 5 (LLM fallback) ────────────────────────────────────────────────


async def test_route_keeps_only_available_routes(llm):
    llm.queue.append({"routes": ["web", "gdrive", "web"], "reason": "fresh news"})
    out = await query_analysis.route(plan(), ["docs", "web"])
    assert out.routes == ["web"]


async def test_route_falls_back_to_docs(llm):
    llm.queue.append({"routes": ["gdrive"], "reason": "drive"})
    assert (await query_analysis.route(plan(), ["docs", "web"])).routes == ["docs"]
    llm.queue.append({"routes": ["gdrive"], "reason": "drive"})
    assert (await query_analysis.route(plan(), ["web"])).routes == ["web"]
