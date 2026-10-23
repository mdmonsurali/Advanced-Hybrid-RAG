"""Shared setup for rag_service tests: no Langfuse, no models, no Qdrant/MinIO."""

from __future__ import annotations

import os

# Read at import time by tracing.py and query_route/router.py.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["WEBSEARCH_ENABLED"] = "true"
os.environ["GDRIVE_ENABLED"] = "true"
os.environ["GDRIVE_ALLOWED_ROLES"] = "master,admin"

import uuid

import pytest
from identity import Identity
from schemas import QueryPlan

USER = Identity(id=uuid.UUID("11111111-1111-1111-1111-111111111111"), role="user")
ADMIN = Identity(id=uuid.UUID("22222222-2222-2222-2222-222222222222"), role="admin")
OTHER_ID = "33333333-3333-3333-3333-333333333333"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def plan(question: str = "What is the refund window?", **overrides) -> QueryPlan:
    return QueryPlan(
        **{"standalone_question": question, "intent": "policy", "rewritten": question, **overrides}
    )
