"""HTTP layer of rag_service (no lifespan: models, Qdrant and MinIO are never loaded)."""

from __future__ import annotations

import main
import pytest
from conftest import USER
from fastapi.testclient import TestClient

HEADERS = {"X-User-Id": str(USER.id), "X-User-Role": USER.role}
PLAN = {"standalone_question": "hi", "intent": "chitchat"}


@pytest.fixture
def client():
    return TestClient(main.app)  # not used as a context manager, so the lifespan never runs


def test_route_requires_identity_headers(client):
    assert client.post("/route", json={"plan": PLAN}).status_code == 401
    bad = {**HEADERS, "X-User-Role": "superuser"}
    assert client.post("/route", json={"plan": PLAN}, headers=bad).status_code == 401


def test_route_endpoint(client):
    body = client.post("/route", json={"plan": PLAN}, headers=HEADERS).json()
    assert body["routes"] == ["direct"]
    assert body["decided_by"] == "rules"


def test_retrieve_rejects_routes_the_user_may_not_use(client):
    resp = client.post("/retrieve", json={"plan": PLAN, "routes": ["gdrive"]}, headers=HEADERS)
    assert resp.status_code == 400
    assert "no permitted routes" in resp.json()["detail"]
