"""STEP 10 — validation.

Checks schema, citation support, groundedness (LLM judge, claim by claim),
safety and PII, then decides what the orchestrator should do next:
accept | retry | clarify | insufficient | blocked.
"""

from __future__ import annotations

import logging
import os
import re

from generation.answer import cited_numbers
from llm_client.client import chat_json
from prompts.judge import judge_messages
from schemas import (
    Evidence,
    GenerateResponse,
    JudgeOutput,
    QueryPlan,
    ValidateResponse,
    Verdict,
)

log = logging.getLogger("llm_service.validation")

MIN_GROUNDED = float(os.getenv("RAG_MIN_GROUNDED", "0.7"))
MASK_PII = os.getenv("RAG_MASK_PII", "true").strip().lower() in {"1", "true", "yes", "on"}

BLOCKED_MESSAGE = "I can't help with that request."

# ── PII ─────────────────────────────────────────────────────────────────

_PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b"),
    "card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # International (+CC ...) or separated national formats only, so plain
    # digit runs like order / invoice numbers are left alone.
    "phone": re.compile(
        r"(?<!\w)(?:\+\d{1,3}[ .-]?(?:\(?\d{1,4}\)?[ .-]?){2,5}\d{2,4}"
        r"|\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4})\b"
    ),
}


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _normalize(value: str) -> str:
    return re.sub(r"[\s()-]", "", value).lower()


def mask_pii(text: str, evidence: list[Evidence]) -> tuple[str, int]:
    """Mask PII in the answer unless the value appears verbatim in evidence
    that belongs to the asking user (their own documents / Drive)."""
    own_text = _normalize(" ".join(e.text for e in evidence if e.owned))
    count = 0

    def _sub(kind: str):
        def repl(m: re.Match) -> str:
            nonlocal count
            value = m.group(0)
            if kind == "card" and not _luhn_ok(value):
                return value
            if kind == "phone" and sum(c.isdigit() for c in value) < 9:
                return value
            if _normalize(value) in own_text:
                return value
            count += 1
            return f"[{kind} redacted]"

        return repl

    for kind, pattern in _PII_PATTERNS.items():
        text = pattern.sub(_sub(kind), text)
    return text, count


# ── Validation ──────────────────────────────────────────────────────────


def _citation_ok(answer: GenerateResponse, evidence: list[Evidence]) -> bool:
    numbers = cited_numbers(answer.answer_markdown)
    if any(n < 1 or n > len(evidence) for n in numbers):
        return False
    if evidence and not answer.insufficient_evidence and not numbers:
        return False
    return True


async def validate(
    plan: QueryPlan, evidence: list[Evidence], answer: GenerateResponse, direct: bool
) -> ValidateResponse:
    if not answer.schema_ok or not answer.answer_markdown.strip():
        return ValidateResponse(
            answer_markdown=answer.answer_markdown,
            verdict=Verdict(
                action="retry",
                reason="answer did not match the output schema",
                grounded_score=0.0,
                citation_ok=False,
                schema_ok=False,
            ),
        )

    text, pii = (
        mask_pii(answer.answer_markdown, evidence) if MASK_PII else (answer.answer_markdown, 0)
    )

    if direct:
        return ValidateResponse(
            answer_markdown=text,
            verdict=Verdict(
                action="accept",
                reason="direct answer (no retrieval)",
                grounded_score=1.0,
                citation_ok=True,
                schema_ok=True,
                pii_masked=pii,
            ),
        )

    if not evidence or answer.insufficient_evidence:
        return ValidateResponse(
            answer_markdown=text,
            verdict=Verdict(
                action="retry",
                reason="no usable evidence for this question",
                grounded_score=0.0,
                citation_ok=True,
                schema_ok=True,
                pii_masked=pii,
            ),
        )

    citation_ok = _citation_ok(answer, evidence)

    try:
        judge = await chat_json(
            judge_messages(plan.standalone_question, answer.answer_markdown, evidence),
            JudgeOutput,
            max_tokens=1024,
        )
    except Exception as e:
        # Without a judge we can only trust citations; be conservative.
        log.warning("[validate] judge failed: %s", e)
        judge = None

    if judge is not None and (judge.unsafe or judge.policy_violation):
        return ValidateResponse(
            answer_markdown=BLOCKED_MESSAGE,
            verdict=Verdict(
                action="blocked",
                reason="answer flagged by the safety check",
                grounded_score=0.0,
                citation_ok=citation_ok,
                schema_ok=True,
                pii_masked=pii,
                unsafe=True,
            ),
        )

    if judge is None:
        grounded = 0.75 if citation_ok else 0.0
        unsupported: list[str] = []
    elif not judge.claims:
        grounded = 1.0
        unsupported = []
    else:
        supported = sum(1 for c in judge.claims if c.supported)
        grounded = supported / len(judge.claims)
        unsupported = [c.claim for c in judge.claims if not c.supported]

    if grounded < MIN_GROUNDED:
        action, reason = "retry", f"groundedness {grounded:.2f} below {MIN_GROUNDED}"
    elif not citation_ok:
        action, reason = "retry", "citations missing or pointing at unknown evidence"
    else:
        action, reason = "accept", "answer is grounded in the cited evidence"

    log.info(
        "[validate] action=%s grounded=%.2f citations_ok=%s pii=%d",
        action,
        grounded,
        citation_ok,
        pii,
    )
    return ValidateResponse(
        answer_markdown=text,
        verdict=Verdict(
            action=action,
            reason=reason,
            grounded_score=round(grounded, 3),
            citation_ok=citation_ok,
            schema_ok=True,
            unsupported_claims=unsupported,
            pii_masked=pii,
        ),
    )
