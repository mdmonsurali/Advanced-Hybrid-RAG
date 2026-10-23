"""Shared setup for llm_service tests: local prompts, no Langfuse, no model, no SGLang."""

from __future__ import annotations

import os

# Read at import time by tracing.py and prompts/registry.py.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["PROMPT_SOURCE"] = "local"
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
os.environ.pop("LANGFUSE_SECRET_KEY", None)

import pytest
from llm_client import client
from schemas import Evidence, QueryPlan

# No tokenizer download: count_tokens falls back to its ~4-chars-per-token estimate.
client._tokenizer = False


@pytest.fixture
def anyio_backend():
    return "asyncio"


def plan(**overrides) -> QueryPlan:
    base = {
        "standalone_question": "What is the refund window?",
        "intent": "policy",
        "rewritten": "refund window days",
    }
    return QueryPlan(**{**base, **overrides})


def evidence(n: int = 2, **overrides) -> list[Evidence]:
    return [
        Evidence(
            id=f"e{i}",
            route="docs",
            text=f"Chunk {i}: refunds are accepted within 30 days of purchase.",
            score=1.0 - i * 0.1,
            source="policy.pdf",
            doc_id="doc-1",
            page=i + 1,
            **overrides,
        )
        for i in range(n)
    ]


class FakeLLM:
    """Replaces chat_json in the modules that call it. Each queued value is
    returned (or raised, if it is an exception) by the next call."""

    def __init__(self) -> None:
        self.queue: list = []
        self.calls: list[tuple[list[dict], type]] = []

    async def __call__(self, messages, schema, **kwargs):
        self.calls.append((messages, schema))
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, schema) else schema.model_validate(item)


@pytest.fixture
def llm(monkeypatch) -> FakeLLM:
    from generation import answer, compression, query_analysis, validation

    fake = FakeLLM()
    for module in (answer, compression, query_analysis, validation):
        monkeypatch.setattr(module, "chat_json", fake)
    return fake
