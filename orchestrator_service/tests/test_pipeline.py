"""The STEP 3-10 pipeline against fake llm/rag services."""

from __future__ import annotations

import clients
import pipeline
import pytest
from conftest import USER

pytestmark = pytest.mark.anyio


async def collect(question="refunds?", history=None, options=None):
    return [e async for e in pipeline.run(question, history or [], USER, options or {})]


def names(events):
    return [name for name, _ in events]


def final(events):
    name, data = events[-1]
    assert name == "answer"
    return data


async def test_happy_path_streams_every_step(services):
    events = await collect()
    assert names(events) == [
        "understanding",
        "transformation",
        "routing",
        "retrieved",
        "reranked",
        "compressed",
        "generating",
        "validated",
        "answer",
    ]
    answer = final(events)
    assert answer["action"] == "accept"
    assert answer["attempts"] == 1
    assert answer["routes"] == ["docs"]
    # Document citations get a link to the source viewer.
    assert answer["citations"][0]["link"] == "/documents/doc-1/source"


async def test_ambiguous_question_asks_back_without_retrieval(services):
    services.plan = {**services.plan, "ambiguous": True, "clarifying_question": "Which product?"}
    events = await collect()
    assert names(events) == ["understanding", "answer"]
    assert final(events)["action"] == "clarify"
    assert final(events)["answer_markdown"] == "Which product?"
    assert services.count("route") == 0


async def test_direct_route_skips_retrieval(services):
    services.decision = {"routes": ["direct"], "available": ["direct"]}
    events = await collect()
    assert "retrieved" not in names(events)
    assert services.calls[-2] == ("generate", (True, "auto"))


async def test_retry_broadens_and_adds_web(services, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_RETRIES", 1)
    services.decision = {"routes": ["catalog"], "available": ["docs", "web", "catalog"]}
    services.verdicts = [
        {"action": "retry", "grounded_score": 0.2},
        {"action": "accept", "grounded_score": 0.9},
    ]
    events = await collect()
    retrieves = [args for name, args in services.calls if name == "retrieve"]
    assert retrieves == [(["catalog"], False), (["catalog", "web"], True)]
    assert final(events)["attempts"] == 2
    assert final(events)["action"] == "accept"


async def test_low_confidence_fallback_keeps_the_best_attempt(services, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_RETRIES", 1)
    services.verdicts = [
        {"action": "retry", "grounded_score": 0.6},
        {"action": "retry", "grounded_score": 0.3},
    ]
    answer = final(await collect())
    assert answer["action"] == "accept_low_confidence"
    assert answer["verdict"]["grounded_score"] == 0.6


async def test_insufficient_evidence_message(services, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_RETRIES", 0)
    services.verdicts = [{"action": "retry", "grounded_score": 0.1}]
    answer = final(await collect())
    assert answer["action"] == "insufficient"
    assert answer["citations"] == []
    assert "docs (ok)" in answer["answer_markdown"]


async def test_service_error_ends_the_stream(services, monkeypatch):
    async def down(*args, **kwargs):
        raise clients.ServiceError("rag_service", "unreachable (ConnectError)")

    monkeypatch.setattr(clients, "retrieve", down)
    events = await collect()
    assert events[-1] == (
        "error",
        {"service": "rag_service", "detail": "unreachable (ConnectError)"},
    )


async def test_external_evidence_replaces_direct_and_is_reported(services):
    services.decision = {"routes": ["direct"], "available": ["direct", "docs"]}
    external = [{"id": "x1", "route": "mcp", "text": "From a tool", "source": "tool"}]
    answer = final(await collect(options={"external_evidence": external}))
    assert answer["routes"] == ["docs", "mcp"]


async def test_history_keeps_only_recent_chat_turns(services, monkeypatch):
    monkeypatch.setattr(pipeline, "HISTORY_TURNS", 2)
    history = [
        {"role": "system", "content": "ignored"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    await collect(history=history)
    _, (_, sent) = services.calls[0]
    assert sent == history[2:]


def test_insufficient_text_lists_what_was_searched():
    text = pipeline._insufficient({"language": "en"}, ["docs", "web"], {"docs": "empty"})
    assert "docs (empty), web (not run)" in text
    assert pipeline._insufficient({"language": "de"}, [], {}).startswith("Not enough evidence")
