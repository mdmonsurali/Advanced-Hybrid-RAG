"""rag_service end to end: a document ocr_service would have stored in MinIO is chunked,
embedded (real CPU models), written to Qdrant and found again by hybrid search + rerank,
with the owner / shared access rules applied."""

from __future__ import annotations

import io
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from conftest import MINIO_BUCKET, RAG_URL

LAYOUT = Path(__file__).parent / "fixtures" / "refund_policy.layout.json"
ALICE = str(uuid.uuid4())
BOB = str(uuid.uuid4())
QUESTION = {
    "standalone_question": "How many days do I have to return an item for a refund?",
    "intent": "policy",
    "rewritten": "refund return window days",
}


def headers(user: str) -> dict[str, str]:
    return {"X-User-Id": user, "X-User-Role": "user"}


def put_json(minio, key: str, data: dict) -> None:
    body = json.dumps(data).encode("utf-8")
    minio.put_object(
        MINIO_BUCKET, key, io.BytesIO(body), len(body), content_type="application/json"
    )


def search(user: str) -> list[dict]:
    resp = httpx.post(
        f"{RAG_URL}/retrieve",
        json={"plan": QUESTION, "routes": ["docs"]},
        headers=headers(user),
        timeout=120,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["evidence"]


def finds_policy(evidence: list[dict]) -> bool:
    return any("45 days" in ev["text"] for ev in evidence)


def wait_for_index(doc_id: str, timeout: float = 300) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = httpx.get(f"{RAG_URL}/ingest/{doc_id}", timeout=10)
        if resp.status_code == 200 and resp.json()["status"] in {"ok", "error"}:
            return resp.json()
        time.sleep(2)
    raise AssertionError(f"{doc_id} not indexed after {timeout}s")


@pytest.fixture(scope="module")
def document(minio) -> str:
    """Store a finished OCR result for Alice, as ocr_service would, and index it."""
    doc_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    put_json(minio, f"documents/{doc_id}/layout.json", json.loads(LAYOUT.read_text("utf-8")))
    put_json(
        minio,
        f"documents/{doc_id}/meta.json",
        {
            "id": doc_id,
            "original_name": "refund-policy.pdf",
            "status": "ok",
            "page_count": 1,
            "json_key": f"documents/{doc_id}/layout.json",
            "owner_id": ALICE,
            "access_level": "private",
            "created_at": now,
            "updated_at": now,
        },
    )
    resp = httpx.post(f"{RAG_URL}/ingest/{doc_id}", timeout=30)
    assert resp.status_code == 202, resp.text
    return doc_id


def test_document_is_chunked_embedded_and_stored(document):
    entry = wait_for_index(document)
    assert entry["status"] == "ok", entry
    assert entry["points"] >= 1
    assert entry["owner_id"] == ALICE


def test_owner_finds_the_answer(document):
    wait_for_index(document)
    evidence = search(ALICE)
    assert finds_policy(evidence), evidence
    top = next(ev for ev in evidence if "45 days" in ev["text"])
    assert top["doc_id"] == document
    assert top["owned"] is True


def test_private_document_is_invisible_to_others(document):
    wait_for_index(document)
    assert not finds_policy(search(BOB))


def test_sharing_makes_it_visible_and_deleting_removes_it(document):
    wait_for_index(document)
    resp = httpx.patch(
        f"{RAG_URL}/ingest/{document}/access", json={"access_level": "shared"}, timeout=30
    )
    assert resp.status_code == 200, resp.text
    shared = search(BOB)
    assert finds_policy(shared)
    assert all(not ev["owned"] for ev in shared if ev.get("doc_id") == document)

    assert httpx.delete(f"{RAG_URL}/ingest/{document}", timeout=30).status_code == 204
    assert not finds_policy(search(ALICE))


def test_catalog_route_needs_no_vector_search():
    resp = httpx.post(
        f"{RAG_URL}/route",
        json={"plan": {"standalone_question": "how many documents do I have", "intent": "catalog"}},
        headers=headers(ALICE),
        timeout=30,
    )
    assert resp.json()["routes"] == ["catalog"]
