"""STEP 10: PII masking and the accept / retry / blocked decision."""

from __future__ import annotations

import pytest
from conftest import evidence, plan
from generation import validation
from schemas import Evidence, GenerateResponse

pytestmark = pytest.mark.anyio


def answer(text="Refunds are accepted for 30 days [1].", **overrides) -> GenerateResponse:
    base = {
        "answer_markdown": text,
        "citations": [],
        "confidence": 0.8,
        "insufficient_evidence": False,
    }
    return GenerateResponse(**{**base, **overrides})


# ── PII ──────────────────────────────────────────────────────────────────


def test_masks_email_ssn_and_valid_card_numbers():
    text, count = validation.mask_pii(
        "Mail jane@example.com, SSN 123-45-6789, card 4111 1111 1111 1111.", []
    )
    assert "jane@example.com" not in text
    assert "[email redacted]" in text
    assert "[ssn redacted]" in text
    assert "[card redacted]" in text
    assert count == 3


def test_leaves_order_numbers_and_invalid_cards_alone():
    text, count = validation.mask_pii("Order 1234567890123 and ref 4111 1111 1111 1112.", [])
    assert count == 0
    assert "1234567890123" in text


def test_keeps_pii_from_the_users_own_documents():
    own = Evidence(id="e1", route="docs", text="Contact: jane@example.com", owned=True)
    shared = Evidence(id="e2", route="web", text="bob@example.com", owned=False)
    text, count = validation.mask_pii("jane@example.com and bob@example.com", [own, shared])
    assert text == "jane@example.com and [email redacted]"
    assert count == 1


def test_international_phone_numbers_are_masked():
    text, count = validation.mask_pii("Call +49 30 1234 5678 today.", [])
    assert count == 1
    assert "[phone redacted]" in text


# ── decision ─────────────────────────────────────────────────────────────


async def test_schema_failure_asks_for_retry(llm):
    out = await validation.validate(plan(), evidence(), answer(schema_ok=False), direct=False)
    assert out.verdict.action == "retry"
    assert out.verdict.schema_ok is False
    assert llm.calls == []


async def test_direct_answers_are_accepted_without_judge(llm):
    out = await validation.validate(plan(), [], answer("Hello!"), direct=True)
    assert out.verdict.action == "accept"
    assert llm.calls == []


async def test_no_evidence_means_retry(llm):
    out = await validation.validate(plan(), [], answer(), direct=False)
    assert out.verdict.action == "retry"


async def test_grounded_and_cited_answer_is_accepted(llm):
    llm.queue.append({"claims": [{"claim": "30 days", "supported": True}]})
    out = await validation.validate(plan(), evidence(), answer(), direct=False)
    assert out.verdict.action == "accept"
    assert out.verdict.grounded_score == 1.0
    assert out.verdict.citation_ok is True


async def test_unsupported_claims_lower_groundedness(llm):
    llm.queue.append(
        {
            "claims": [
                {"claim": "30 days", "supported": True},
                {"claim": "free shipping", "supported": False},
            ]
        }
    )
    out = await validation.validate(plan(), evidence(), answer(), direct=False)
    assert out.verdict.action == "retry"
    assert out.verdict.grounded_score == 0.5
    assert out.verdict.unsupported_claims == ["free shipping"]


async def test_citation_to_unknown_evidence_means_retry(llm):
    llm.queue.append({"claims": []})
    out = await validation.validate(plan(), evidence(2), answer("See [7]."), direct=False)
    assert out.verdict.citation_ok is False
    assert out.verdict.action == "retry"


async def test_unsafe_answer_is_blocked(llm):
    llm.queue.append({"claims": [], "unsafe": True})
    out = await validation.validate(plan(), evidence(), answer(), direct=False)
    assert out.verdict.action == "blocked"
    assert out.answer_markdown == validation.BLOCKED_MESSAGE


async def test_judge_failure_trusts_valid_citations(llm):
    llm.queue.append(RuntimeError("SGLang down"))
    out = await validation.validate(plan(), evidence(), answer(), direct=False)
    assert out.verdict.action == "accept"
    assert out.verdict.grounded_score == 0.75
