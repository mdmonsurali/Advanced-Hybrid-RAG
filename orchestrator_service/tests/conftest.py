"""Shared setup for orchestrator_service tests: no Langfuse, no GPU swap, fake services."""

from __future__ import annotations

import os

# Read at import time by tracing.py / gpu_swap.py.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["GPU_SWAP"] = "false"

import uuid

import clients
import pytest
from identity import Identity

USER = Identity(id=uuid.UUID("11111111-1111-1111-1111-111111111111"), role="user")
HEADERS = {"X-User-Id": str(USER.id), "X-User-Role": USER.role}


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeServices:
    """Stand-ins for the llm_service / rag_service calls in clients.py.

    Each attribute is the response the matching call returns; `calls` records
    (name, args) so tests can check what the pipeline sent."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.plan = {
            "standalone_question": "What is the refund policy?",
            "intent": "factual",
            "language": "en",
            "ambiguous": False,
            "clarifying_question": None,
        }
        self.decision = {"routes": ["docs"], "available": ["docs", "web", "catalog", "direct"]}
        self.evidence = [
            {
                "id": "e1",
                "route": "docs",
                "source": "policy.pdf",
                "page": 2,
                "score": 0.91,
                "text": "Refunds within 30 days.",
                "doc_id": "doc-1",
            },
        ]
        self.answer = {
            "citations": [{"n": 1, "route": "docs", "doc_id": "doc-1"}],
            "confidence": 0.8,
        }
        self.verdicts = [{"action": "accept", "grounded_score": 0.9}]

    async def query_analysis(self, question, history):
        self.calls.append(("query_analysis", (question, history)))
        return self.plan

    async def route(self, plan, identity, route_hint, deny_routes):
        self.calls.append(("route", (route_hint, deny_routes)))
        return self.decision

    async def retrieve(self, plan, routes, broaden, identity, external=None, debug=False):
        self.calls.append(("retrieve", (list(routes), broaden)))
        return {
            "evidence": self.evidence,
            "route_status": {r: "ok" for r in routes},
            "stats": {"candidates": len(self.evidence)},
            "candidates": self.evidence,
        }

    async def compress(self, plan, evidence):
        self.calls.append(("compress", (len(evidence),)))
        return {"evidence": evidence, "dropped": 0, "merged": 0, "extracted": 0, "tokens": 42}

    async def generate(self, plan, evidence, direct, answer_style="auto"):
        self.calls.append(("generate", (direct, answer_style)))
        return self.answer

    async def validate(self, plan, evidence, answer, direct):
        verdict = self.verdicts[min(self.count("validate"), len(self.verdicts) - 1)]
        self.calls.append(("validate", (direct,)))
        return {"verdict": verdict, "answer_markdown": "Refunds are accepted for 30 days [1]."}

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)


@pytest.fixture
def services(monkeypatch) -> FakeServices:
    fake = FakeServices()
    for name in ("query_analysis", "route", "retrieve", "compress", "generate", "validate"):
        monkeypatch.setattr(clients, name, getattr(fake, name))
    return fake
