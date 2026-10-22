"""Prompt for STEP 10: claim-level groundedness and safety judgement."""

from __future__ import annotations

from prompts import registry
from prompts.answer import format_evidence
from schemas import Evidence

JUDGE_SYSTEM = registry.register(
    "rag/judge",
    """\
You are a strict fact-checker for answers produced from evidence.

1. Split the answer into its individual factual claims (ignore greetings, \
formatting and statements that the evidence is insufficient).
2. For each claim decide whether the numbered evidence explicitly supports \
it. Paraphrase is fine; a claim that needs outside knowledge or contradicts \
the evidence is NOT supported. List the evidence numbers that support it.
3. unsafe: true if the answer gives instructions for violence, weapons, \
self-harm, malware or other clearly harmful acts.
4. policy_violation: true if the answer contains hateful, harassing or \
sexual content involving minors.

Return JSON with claims (claim, supported, evidence), unsafe and \
policy_violation.""",
)


def judge_messages(question: str, answer: str, evidence: list[Evidence]) -> list[dict]:
    user = (
        f"Evidence:\n{format_evidence(evidence)}\n\n"
        f"Question: {question}\n\n"
        f"Answer to check:\n{answer}"
    )
    return registry.messages("rag/judge", [{"role": "user", "content": user}])
