"""HTTP layer: identity headers, request validation, SSE and /query."""

from __future__ import annotations

import json

import clients
import main
import pytest
from conftest import HEADERS
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(main.app)


def sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health_reports_degraded_services(client, monkeypatch):
    async def healthy(base):
        return base == clients.RAG_SERVICE_URL

    monkeypatch.setattr(clients, "healthy", healthy)
    assert client.get("/health").json() == {
        "status": "degraded",
        "llm_service": "down",
        "rag_service": "ok",
    }


@pytest.mark.parametrize(
    "headers",
    [{}, {"X-User-Id": "not-a-uuid", "X-User-Role": "user"}, {**HEADERS, "X-User-Role": "root"}],
)
def test_chat_requires_valid_identity_headers(client, headers):
    assert client.post("/chat", json={"question": "hi"}, headers=headers).status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "x" * (main.MAX_QUESTION_CHARS + 1)},
        {"question": "hi", "history": "not a list"},
        {"question": "hi", "options": {"answer_style": "poetic"}},
        {"question": "hi", "options": "not an object"},
    ],
)
def test_bad_requests_are_rejected(client, payload):
    assert client.post("/query", json=payload, headers=HEADERS).status_code == 400


def test_options_are_sanitised():
    opts = main._options(
        {
            "options": {
                "route_hint": ["web", "evil"],
                "external_evidence": [{"id": 1, "text": "t" * 30000}, {"text": "no id"}],
                "session_id": "s" * 500,
            }
        }
    )
    assert opts["route_hint"] == ["web"]
    assert len(opts["external_evidence"]) == 1
    assert len(opts["external_evidence"][0]["text"]) == main.MAX_EXTERNAL_CHARS
    assert opts["external_evidence"][0]["route"] == "mcp"
    assert len(opts["session_id"]) == 200


def test_chat_streams_server_sent_events(client, services):
    resp = client.post("/chat", json={"question": "refunds?"}, headers=HEADERS)
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp.text)
    assert events[0][0] == "understanding"
    assert events[-1][0] == "answer"
    assert events[-1][1]["action"] == "accept"


def test_query_returns_answer_and_trace(client, services):
    body = client.post("/query", json={"question": "refunds?"}, headers=HEADERS).json()
    assert body["answer"]["action"] == "accept"
    assert [step["event"] for step in body["trace"]][:2] == ["understanding", "transformation"]


def test_query_turns_service_errors_into_502(client, services, monkeypatch):
    async def down(*args, **kwargs):
        raise clients.ServiceError("llm_service", "HTTP 500: boom")

    monkeypatch.setattr(clients, "query_analysis", down)
    resp = client.post("/query", json={"question": "refunds?"}, headers=HEADERS)
    assert resp.status_code == 502
    assert resp.json()["detail"] == "llm_service: HTTP 500: boom"


def test_gpu_leases_are_noops_when_swap_is_off(client):
    assert client.post("/gpu/acquire/llm").json() == {"lease_id": "noop", "tenant": "llm"}
    assert client.post("/gpu/renew/noop").status_code == 200
    assert client.post("/gpu/acquire/tpu").status_code == 400
    assert client.get("/gpu/status").json() == {"gpu_swap_enabled": False}
