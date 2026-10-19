"""Prompts for STEP 3 (query understanding), STEP 4 (query transformation)
and the LLM fallback of STEP 5 (query routing).

Registered as rag/understanding, rag/transform and rag/route (see
prompts/registry.py)."""

from __future__ import annotations

import json
from datetime import date

from prompts import registry
from schemas import QueryPlan, Route, Turn

UNDERSTANDING_SYSTEM = registry.register("rag/understanding", """\
You analyse questions for a retrieval system over the user's own documents \
(PDFs, scans, Word files), their Google Drive and the web.

Work out what the user is really asking before anything is retrieved:
- standalone_question: rewrite the latest question so it is fully \
understandable without the conversation (resolve "it", "that one", "the \
second table", ...). Keep the user's language.
- intent, exactly one of:
  fact            - a specific value, name, date, definition or number
  comparison      - differences/similarities between two or more things
  troubleshooting - an error, failure or "how do I fix" question
  policy          - rules, obligations, permissions, contract or policy clauses
  summary         - an overview / summary of a document or topic
  catalog         - questions about the document collection itself \
("how many invoices did I upload", "list my PDFs from May")
  chitchat        - greetings, thanks, small talk, questions about you
- entities: key names, codes, products, people, document titles mentioned.
- filters: doc_type (e.g. invoice, contract, report) and ISO date_from / \
date_to only when the user explicitly restricts them. Otherwise null.
- language: ISO 639-1 code of the user's language.
- ambiguous: true only if the question cannot be answered without more \
information from the user (e.g. "what about the other one?" with no context). \
Then give a short clarifying_question in the user's language.

Today is {{today}}.""")

TRANSFORM_SYSTEM = registry.register("rag/transform", """\
You prepare search queries. Never search the raw question directly.
- rewritten: a precise, keyword-rich search query for the question.
- expansions: up to 3 alternative phrasings using synonyms, abbreviations \
spelled out, or domain terms the documents are likely to use.
- sub_questions: only for comparison or multi-part questions, up to 3 \
self-contained questions that each retrieve one part. Otherwise empty.
- hyde_passage: a short hypothetical passage (2-4 sentences) that would \
answer the question, written in the style of the source documents. It is \
used only for semantic search, so plausible wording matters more than \
correct facts.
Write everything in the language of the question.""")

ROUTE_SYSTEM = registry.register("rag/route", """\
You decide where evidence for a question should come from. Available routes:
  docs    - the user's uploaded documents (vector + keyword search)
  catalog - metadata about the uploaded documents (counts, lists, dates, types)
  gdrive  - the user's Google Drive files
  web     - public web search (recent events, public facts not in documents)
  direct  - no retrieval needed (greetings, general knowledge, about the assistant)
Choose one or more routes from the available list only. Prefer docs for \
questions about the user's own material. Add web only when the answer is \
likely public or time-sensitive. Explain the choice in one short sentence.""")


def _history_messages(history: list[Turn], limit: int = 6) -> list[dict]:
    return [{"role": t.role, "content": t.content} for t in history[-limit:]]


def understanding_messages(question: str, history: list[Turn]) -> list[dict]:
    return registry.messages(
        "rag/understanding", [*_history_messages(history), {"role": "user", "content": question}],
        today=date.today().isoformat())


def transform_messages(standalone_question: str, intent: str,
                       entities: list[str]) -> list[dict]:
    user = (f"Question: {standalone_question}\n"
            f"Intent: {intent}\n"
            f"Entities: {', '.join(entities) or '-'}")
    return registry.messages("rag/transform", [{"role": "user", "content": user}])


def route_messages(plan: QueryPlan, available: list[Route]) -> list[dict]:
    user = json.dumps({
        "question": plan.standalone_question,
        "intent": plan.intent,
        "entities": plan.entities,
        "available_routes": available,
    }, ensure_ascii=False)
    return registry.messages("rag/route", [{"role": "user", "content": user}])
