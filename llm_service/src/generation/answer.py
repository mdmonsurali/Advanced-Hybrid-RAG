"""STEP 9 — build the prompt and generate the structured, cited answer."""

from __future__ import annotations

import logging
import os
import re

from llm_client.client import LLMOutputError, chat_json
from prompts.answer import answer_messages, direct_messages
from schemas import AnswerOutput, Citation, Evidence, GenerateResponse, QueryPlan

log = logging.getLogger("llm_service.answer")

ANSWER_TOKENS = int(os.getenv("RAG_ANSWER_TOKENS", "1024"))
ANSWER_TEMPERATURE = float(os.getenv("RAG_ANSWER_TEMPERATURE", "0.3"))
SNIPPET_CHARS = int(os.getenv("RAG_CITATION_SNIPPET_CHARS", "600"))

_CITE_RE = re.compile(r"\[(\d{1,3})\]")


def cited_numbers(text: str) -> list[int]:
    return [int(n) for n in _CITE_RE.findall(text)]


def _citations(numbers: list[int], evidence: list[Evidence]) -> list[Citation]:
    out = []
    for n in dict.fromkeys(numbers):
        if 1 <= n <= len(evidence):
            ev = evidence[n - 1]
            snippet = ev.text.strip()
            if len(snippet) > SNIPPET_CHARS:
                snippet = snippet[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …"
            out.append(Citation(n=n, evidence_id=ev.id, route=ev.route, source=ev.source,
                                doc_id=ev.doc_id, page=ev.page, url=ev.url, snippet=snippet))
    return sorted(out, key=lambda c: c.n)


async def generate(plan: QueryPlan, evidence: list[Evidence], direct: bool,
                   style: str = "auto") -> GenerateResponse:
    if direct:
        messages = direct_messages(plan, style)
    elif not evidence:
        # Retrieval found nothing: say so instead of letting the model guess.
        return GenerateResponse(
            answer_markdown="I couldn't find information about this in the available sources.",
            citations=[], confidence=0.0, insufficient_evidence=True,
        )
    else:
        messages = answer_messages(plan, evidence, style)

    try:
        out = await chat_json(messages, AnswerOutput, temperature=ANSWER_TEMPERATURE,
                              max_tokens=ANSWER_TOKENS)
    except LLMOutputError as e:
        log.warning("[answer] structured output failed: %s", e)
        return GenerateResponse(answer_markdown="", citations=[], confidence=0.0,
                                insufficient_evidence=True, schema_ok=False)

    if direct:
        return GenerateResponse(answer_markdown=out.answer_markdown, citations=[],
                                confidence=out.confidence, insufficient_evidence=False)

    numbers = cited_numbers(out.answer_markdown) + list(out.citations)
    return GenerateResponse(
        answer_markdown=out.answer_markdown,
        citations=_citations(numbers, evidence),
        confidence=out.confidence,
        insufficient_evidence=out.insufficient_evidence,
    )
