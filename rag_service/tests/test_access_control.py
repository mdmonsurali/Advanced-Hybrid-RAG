"""Who may see which documents: the Qdrant ACL filter and the catalog route."""

from __future__ import annotations

import pytest
import storage
from conftest import OTHER_ID, USER, plan
from hybrid_retrieval import search
from query_route import catalog
from schemas import Filters


def conditions(flt) -> list:
    return list(flt.must or [])


def test_every_search_is_limited_to_own_or_shared_documents():
    for broaden in (False, True):
        must = conditions(search.build_filter(plan(), USER, broaden))
        acl = must[0]
        keys = {(c.key, c.match.value) for c in acl.should}
        assert keys == {("owner_id", str(USER.id)), ("access_level", "shared")}


def test_plan_filters_apply_until_the_search_is_broadened():
    p = plan(filters=Filters(doc_type="Invoice", date_from="2026-01-01", date_to="2026-03-31"))
    strict = conditions(search.build_filter(p, USER, broaden=False))
    assert [c.key for c in strict[1:]] == ["doc_type", "timestamp"]
    assert strict[1].match.value == "invoice"
    assert strict[2].range.gte.year == 2026

    broad = conditions(search.build_filter(p, USER, broaden=True))
    assert len(broad) == 1  # only the ACL survives a broaden retry


def test_sub_questions_are_searched_separately():
    p = plan(rewritten="refund window", sub_questions=["refund for shoes", "refund for books"])
    assert search.search_queries(p) == [
        ("refund for shoes", False),
        ("refund for books", False),
        ("refund window", True),
    ]
    assert search.search_queries(plan(rewritten="")) == [("What is the refund window?", True)]


@pytest.fixture
def registry(monkeypatch):
    entries = {
        "mine": {"status": "ok", "owner_id": str(USER.id), "access_level": "private",
                 "source": "my-invoice.pdf", "doc_type": "invoice", "created_at": "2026-05-02"},
        "shared": {"status": "ok", "owner_id": OTHER_ID, "access_level": "shared",
                   "source": "handbook.pdf", "doc_type": "policy", "created_at": "2026-04-10"},
        "private": {"status": "ok", "owner_id": OTHER_ID, "access_level": "private",
                    "source": "secret.pdf", "doc_type": "invoice", "created_at": "2026-05-03"},
        "failed": {"status": "error", "owner_id": str(USER.id), "access_level": "private",
                   "source": "broken.pdf", "doc_type": "invoice", "created_at": "2026-05-04"},
    }  # fmt: skip
    monkeypatch.setattr(storage, "REGISTRY", entries)
    return entries


def test_catalog_lists_own_and_shared_documents_only(registry):
    [ev] = catalog.catalog_evidence(plan(intent="catalog"), USER)
    assert "2 documents match (1 uploaded by you, 1 shared by others)" in ev.text
    assert "my-invoice.pdf" in ev.text
    assert "handbook.pdf" in ev.text
    assert "secret.pdf" not in ev.text  # someone else's private document
    assert "broken.pdf" not in ev.text  # not indexed


def test_catalog_applies_type_and_date_filters(registry):
    p = plan(intent="catalog", filters=Filters(doc_type="INVOICE", date_from="2026-05-01"))
    [ev] = catalog.catalog_evidence(p, USER)
    assert "1 documents match" in ev.text
    assert "Filters applied: type=INVOICE" in ev.text
