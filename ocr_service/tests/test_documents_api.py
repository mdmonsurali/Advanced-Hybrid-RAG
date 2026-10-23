"""Document API: who can read, change and delete which documents."""

from __future__ import annotations

import uuid

import pytest
from conftest import ALICE, BOB, headers


def test_identity_headers_are_required(client, store):
    doc = store.add(ALICE)
    assert client.get(f"/documents/{doc}").status_code == 401
    bad = {"X-User-Id": str(ALICE), "X-User-Role": "root"}
    assert client.get(f"/documents/{doc}", headers=bad).status_code == 401


def test_owner_reads_private_document(client, store):
    doc = store.add(ALICE)
    resp = client.get(f"/documents/{doc}", headers=headers(ALICE))
    assert resp.status_code == 200
    assert resp.json()["original_filename"] == "report.pdf"


def test_private_documents_are_hidden_from_others(client, store):
    doc = store.add(ALICE)
    # 404, not 403: Bob must not learn the document exists.
    assert client.get(f"/documents/{doc}", headers=headers(BOB)).status_code == 404
    assert client.get(f"/documents/{doc}/md", headers=headers(BOB)).status_code == 404
    assert client.get(f"/documents/{doc}/images/p1.png", headers=headers(BOB)).status_code == 404


def test_shared_documents_are_readable_by_everyone(client, store):
    doc = store.add(ALICE, access_level="shared")
    assert client.get(f"/documents/{doc}", headers=headers(BOB)).status_code == 200
    md = client.get(f"/documents/{doc}/md", headers=headers(BOB))
    assert md.text == "# Report"
    assert md.headers["content-disposition"] == "inline"


def test_only_the_owner_changes_access(client, store, rag_calls):
    doc = store.add(ALICE)
    url = f"/documents/{doc}/access"
    assert (
        client.patch(url, json={"access_level": "shared"}, headers=headers(BOB)).status_code == 404
    )
    assert (
        client.patch(url, json={"access_level": "public"}, headers=headers(ALICE)).status_code
        == 400
    )

    resp = client.patch(url, json={"access_level": "shared"}, headers=headers(ALICE))
    assert resp.status_code == 200
    assert store.docs[doc]["access_level"] == "shared"
    # rag_service is told so its index uses the new access level.
    assert rag_calls == [("PATCH", f"/ingest/{doc}/access", {"access_level": "shared"})]


def test_only_the_owner_deletes(client, store, rag_calls):
    doc = store.add(ALICE, access_level="shared")
    assert client.delete(f"/documents/{doc}", headers=headers(BOB)).status_code == 404
    assert client.delete(f"/documents/{doc}", headers=headers(ALICE)).status_code == 204
    assert doc not in store.docs
    assert rag_calls == [("DELETE", f"/ingest/{doc}")]


@pytest.mark.parametrize("name", ["..%2Fsecret.png", "a..b.png", "sub%5Cimg.png"])
def test_image_names_cannot_escape_the_document(client, store, name):
    doc = store.add(ALICE)
    resp = client.get(f"/documents/{doc}/images/{name}", headers=headers(ALICE))
    assert resp.status_code in {400, 404}
    assert resp.status_code != 200


def test_bad_ids_and_kinds(client, store):
    doc = store.add(ALICE)
    assert client.get("/documents/not-a-uuid", headers=headers(ALICE)).status_code == 400
    assert client.get(f"/documents/{doc}/exe", headers=headers(ALICE)).status_code == 400
    assert client.get(f"/documents/{uuid.uuid4()}", headers=headers(ALICE)).status_code == 404


def test_account_wipe_is_limited_to_self_or_admins(client, store, rag_calls):
    store.add(ALICE)
    store.add(ALICE)
    store.add(BOB)
    url = f"/internal/owner/{ALICE}/documents"
    assert client.delete(url, headers=headers(BOB)).status_code == 403
    assert client.delete(url, headers=headers(BOB, role="admin")).json() == {"deleted": 2}
    assert [d["owner_id"] for d in store.docs.values()] == [BOB]
    assert len(rag_calls) == 2
