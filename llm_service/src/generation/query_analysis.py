"""STEP 3 (query understanding), STEP 4 (query transformation), the LLM
fallback for STEP 5 (routing) and the STEP 2 document-metadata call."""

from __future__ import annotations

import logging

from llm_client.client import chat_json
from prompts.metadata import doc_metadata_messages
from prompts.query import route_messages, transform_messages, understanding_messages
from schemas import (
    DocMetadataOutput,
    QueryPlan,
    Route,
    RouteOutput,
    TransformationOutput,
    Turn,
    UnderstandingOutput,
)

log = logging.getLogger("llm_service.query")

# Intents that never hit the vector store, so search rewrites are wasted work.
_NO_TRANSFORM_INTENTS = {"chitchat", "catalog"}


async def analyze(question: str, history: list[Turn]) -> QueryPlan:
    understanding = await chat_json(
        understanding_messages(question, history), UnderstandingOutput, max_tokens=512
    )
    # An ambiguous question with nothing concrete to search for goes straight
    # back to the user; anything else is answerable enough to try.
    if understanding.ambiguous and understanding.entities:
        understanding.ambiguous = False
    if understanding.ambiguous and not understanding.clarifying_question:
        understanding.ambiguous = False

    if understanding.intent in _NO_TRANSFORM_INTENTS or understanding.ambiguous:
        transformation = TransformationOutput(rewritten=understanding.standalone_question)
    else:
        transformation = await chat_json(
            transform_messages(
                understanding.standalone_question, understanding.intent, understanding.entities
            ),
            TransformationOutput,
            temperature=0.2,
            max_tokens=768,
        )
        if understanding.intent != "comparison" and len(transformation.sub_questions) == 1:
            transformation.sub_questions = []

    plan = QueryPlan(**understanding.model_dump(), **transformation.model_dump())
    log.info(
        "[plan] intent=%s subq=%d ambiguous=%s q=%r",
        plan.intent,
        len(plan.sub_questions),
        plan.ambiguous,
        plan.standalone_question[:120],
    )
    return plan


async def route(plan: QueryPlan, available: list[Route]) -> RouteOutput:
    out = await chat_json(route_messages(plan, available), RouteOutput, max_tokens=256)
    routes: list[Route] = [r for r in dict.fromkeys(out.routes) if r in available]
    if not routes:
        routes = ["docs"] if "docs" in available else [available[0]]
    return RouteOutput(routes=routes, reason=out.reason)


async def doc_metadata(name: str, text: str) -> DocMetadataOutput:
    out = await chat_json(doc_metadata_messages(name, text), DocMetadataOutput, max_tokens=384)
    out.doc_type = out.doc_type.strip().lower() or "other"
    return out
