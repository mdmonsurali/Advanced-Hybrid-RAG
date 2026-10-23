"""Shared setup for ocr_service tests: an in-memory document store instead of MinIO."""

from __future__ import annotations

import uuid

import main
import pytest
from fastapi.testclient import TestClient

ALICE = uuid.UUID("11111111-1111-1111-1111-111111111111")
BOB = uuid.UUID("22222222-2222-2222-2222-222222222222")


def headers(user: uuid.UUID, role: str = "user") -> dict[str, str]:
    return {"X-User-Id": str(user), "X-User-Role": role}


class FakeStore:
    """The parts of storage.py the document API uses, backed by a dict."""

    def __init__(self) -> None:
        self.docs: dict[uuid.UUID, dict] = {}

    def add(self, owner: uuid.UUID, access_level: str = "private") -> uuid.UUID:
        doc_id = uuid.uuid4()
        self.docs[doc_id] = {
            "id": doc_id,
            "owner_id": owner,
            "access_level": access_level,
            "status": "ok",
            "original_filename": "report.pdf",
        }
        return doc_id

    async def fetch_document(self, doc_id):
        return self.docs.get(doc_id)

    async def update_document(self, doc_id, **fields):
        self.docs[doc_id].update(fields)

    async def delete_document(self, doc_id, owner_id):
        doc = self.docs.get(doc_id)
        if doc is None or doc["owner_id"] != owner_id:
            return False
        del self.docs[doc_id]
        return True

    async def delete_all_documents_for_owner(self, owner_id):
        doomed = [d for d, doc in self.docs.items() if doc["owner_id"] == owner_id]
        for d in doomed:
            del self.docs[d]
        return doomed

    async def get_artifact_bytes(self, doc_id, kind):
        return "report.md", "text/markdown", b"# Report"

    async def get_image_bytes(self, doc_id, filename):
        return filename, "image/png", b"\x89PNG"


@pytest.fixture
def store(monkeypatch) -> FakeStore:
    fake = FakeStore()
    for name in (
        "fetch_document",
        "update_document",
        "delete_document",
        "delete_all_documents_for_owner",
        "get_artifact_bytes",
        "get_image_bytes",
    ):
        monkeypatch.setattr(main.storage, name, getattr(fake, name))
    return fake


@pytest.fixture
def rag_calls(monkeypatch) -> list[tuple]:
    """Records the fire-and-forget notifications to rag_service."""
    calls: list[tuple] = []
    monkeypatch.setattr(main, "_notify_rag", lambda *args: calls.append(args))
    return calls


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)  # no lifespan: Docling and MinIO are never started
