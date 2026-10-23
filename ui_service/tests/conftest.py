"""Shared setup for ui_service tests: the real app and Chainlit config, fake auth_service."""

from __future__ import annotations

import os
from pathlib import Path

# Read when chainlit / chat_app are imported (main.py mounts the chat app).
os.environ["CHAINLIT_APP_ROOT"] = str(Path(__file__).resolve().parents[1])  # ui_service/.chainlit
os.environ.setdefault("CHAINLIT_AUTH_SECRET", "test-only-secret")
os.environ["CHAT_HISTORY_ENABLED"] = "false"
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

import auth_client
import pytest

USER = auth_client.CurrentUser(id="u-1", username="alice", role="user", must_change_password=False)
ADMIN = auth_client.CurrentUser(id="a-1", username="boss", role="admin", must_change_password=False)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def session(monkeypatch):
    """auth_service stand-in: maps session cookie values to users."""
    sessions: dict[str, auth_client.CurrentUser] = {}

    async def validate(session_id: str):
        return sessions.get(session_id)

    monkeypatch.setattr(auth_client, "validate", validate)
    return sessions
