"""Fixtures for auth_service tests: an isolated JSON store and a fresh app per test."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

# Module-level settings are read at import time, so set them before importing the app.
_TMP = Path(tempfile.mkdtemp(prefix="auth-tests-"))
os.environ["AUTH_DATA_FILE"] = str(_TMP / "auth.json")
os.environ["MASTER_USERNAME"] = "master"
os.environ["MASTER_PASSWORD"] = "master-pass-123"

import db
import main
import pytest
import security
from fastapi.testclient import TestClient
from passlib.context import CryptContext

# bcrypt at 12 rounds takes ~0.3 s per hash; 4 rounds keeps the suite fast.
security._pwd = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=4)

MASTER = ("master", "master-pass-123")


@pytest.fixture(autouse=True)
def fresh_store():
    """Empty in-memory store and no data file before every test."""
    db._users.clear()
    db._sessions.clear()
    db._lock = asyncio.Lock()
    db._loaded = False
    db.DATA_FILE.unlink(missing_ok=True)
    yield
    db.DATA_FILE.unlink(missing_ok=True)


@pytest.fixture
def client(monkeypatch):
    """App with the lifespan run (store loaded, master bootstrapped). The OCR
    document wipe is stubbed and its calls recorded in `client.wiped`."""
    wiped: list[tuple[str, str]] = []

    async def fake_wipe(owner_id, role):
        wiped.append((str(owner_id), role))

    monkeypatch.setattr(main, "_wipe_owner_documents", fake_wipe)
    with TestClient(main.app) as c:
        c.wiped = wiped  # type: ignore[attr-defined]
        yield c


def login(client: TestClient, username: str, password: str) -> dict:
    """Log in and keep the session cookie on the client."""
    resp = client.post("/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()


def create_user(client: TestClient, username: str, role: str = "user") -> dict:
    """Create a user as whoever is logged in on `client`."""
    resp = client.post(
        "/auth/admin/users",
        json={"username": username, "password": f"{username}-pass-123", "role": role},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()
