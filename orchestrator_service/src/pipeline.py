"""The query pipeline, STEP 3 -> 10, as an async stream of step events.

    understanding   llm  /tasks/query-analysis (STEP 3)
    transformation       (same call, STEP 4)
    routing         rag  /route               (STEP 5)
    retrieved       rag  /retrieve            (STEP 5 executors + STEP 6)
    reranked                                   (STEP 7)
    compressed      llm  /tasks/compress      (STEP 8)
    generating      llm  /tasks/generate      (STEP 9)
    validated       llm  /tasks/validate      (STEP 10)
    answer          final answer + citations + verdict
    error           a service failed; the stream ends

Options from the UI (all optional):
    route_hint         force routes (/web, /drive, /docs, /catalog commands)
    deny_routes        routes the user switched off in the chat settings
    answer_style       auto | concise | detailed
    external_evidence  results of user-connected MCP tools, reranked with the rest

Validation decides the next move. On `retry` the evidence is fetched again
with broadened retrieval (filters dropped, K doubled, web added when
available) up to RAG_MAX_RETRIES times; if confidence stays low the user
gets a clarifying question or an honest "not enough evidence" answer.
"""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator

import clients
from clients import ServiceError
from identity import Identity

log = logging.getLogger("orchestrator.pipeline")

MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "1"))
RETRY_ADD_WEB = os.getenv("RAG_RETRY_ADD_WEB", "true").strip().lower() in {"1", "true", "yes", "on"}
# When retries run out, an earlier attempt this grounded is still returned
# (flagged as low confidence) instead of "not enough evidence".
FALLBACK_MIN_GROUNDED = float(os.getenv("RAG_FALLBACK_MIN_GROUNDED", "0.5"))
HISTORY_TURNS = int(os.getenv("RAG_HISTORY_TURNS", "6"))

Event = tuple[str, dict]


def _citation_links(citations: list[dict]) -> list[dict]:
    out = []
    for c in citations:
        link = c.get("url")
        if not link and c.get("route") == "docs" and c.get("doc_id"):
            link = f"/documents/{c['doc_id']}/source"
        out.append({**c, "link": link})
    return out


def _insufficient(plan: dict, routes: list[str], route_status: dict) -> str:
    searched = ", ".join(f"{r} ({route_status.get(r, 'not run')})" for r in routes) or "nothing"
    if plan.get("language", "en").startswith("en"):
        return ("I couldn't find enough reliable evidence to answer this.\n\n"
                f"Searched: {searched}. Try rephrasing, naming the document, or uploading "
                "the source that contains the answer.")
    return f"Not enough evidence found. Searched: {searched}."


def _dbg(on: bool, key: str, evs: list[dict] | None) -> dict:
    """Full chunk text for the admin debug trace (empty unless enabled)."""
    if not on or not evs:
        return {}
    return {key: [{"n": i, "id": e.get("id"), "route": e.get("route"), "source": e.get("source"),
                   "page": e.get("page"), "score": round(e.get("score") or 0, 4),
                   "headings": e.get("headings") or [], "text": e.get("text", "")}
                  for i, e in enumerate(evs, start=1)]}


async def run(question: str, history: list[dict], identity: Identity,
              options: dict | None = None) -> AsyncIterator[Event]:
    options = options or {}
    route_hint: list[str] = options.get("route_hint") or []
    deny_routes: list[str] = options.get("deny_routes") or []
    answer_style: str = options.get("answer_style") or "auto"
    external: list[dict] = options.get("external_evidence") or []
    # Chunk-level trace is for admins only (rag_service re-checks the role).
    debug = bool(options.get("debug")) and identity.role in {"admin", "master"}
    t0 = time.perf_counter()
    history = [h for h in history if h.get("role") in {"user", "assistant"}][-HISTORY_TURNS:]
    try:
        # STEP 3 + 4
        plan = await clients.query_analysis(question, history)
        yield "understanding", {k: plan.get(k) for k in (
            "standalone_question", "intent", "entities", "filters", "language", "ambiguous")}
        if plan.get("ambiguous") and plan.get("clarifying_question"):
            yield "answer", {"answer_markdown": plan["clarifying_question"], "citations": [],
                             "confidence": 0.0, "action": "clarify", "routes": [],
                             "attempts": 0, "elapsed_sec": round(time.perf_counter() - t0, 2)}
            return
        yield "transformation", {k: plan.get(k) for k in (
            "rewritten", "expansions", "sub_questions", "hyde_passage")}

        # STEP 5
        decision = await clients.route(plan, identity, route_hint, deny_routes)
        routes: list[str] = decision["routes"]
        if external and "direct" in routes:
            routes = [r for r in routes if r != "direct"] or ["docs"]
        yield "routing", {**decision, "routes": routes,
                          "external_evidence": len(external)}
        direct = routes == ["direct"]

        best: dict | None = None
        route_status: dict = {}
        result: dict | None = None
        for attempt in range(MAX_RETRIES + 1):
            broaden = attempt > 0
            evidence: list[dict] = []
            if not direct:
                if broaden:
                    if "docs" not in routes and "catalog" not in routes:
                        routes = [*routes, "docs"]
                    if RETRY_ADD_WEB and "web" in decision.get("available", []) and "web" not in routes:
                        routes = [*routes, "web"]
                # STEP 5 executors + 6 + 7
                retrieval = await clients.retrieve(plan, routes, broaden, identity, external, debug)
                route_status = retrieval["route_status"]
                yield "retrieved", {"attempt": attempt, "routes": routes, "broaden": broaden,
                                    "route_status": route_status, **retrieval["stats"],
                                    **_dbg(debug, "candidate_chunks", retrieval.get("candidates"))}
                yield "reranked", {"attempt": attempt, "kept": len(retrieval["evidence"]),
                                   "top": [{"source": e["source"], "route": e["route"],
                                            "page": e.get("page"), "score": round(e["score"], 3)}
                                           for e in retrieval["evidence"][:5]],
                                   **_dbg(debug, "chunks", retrieval["evidence"])}
                # STEP 8
                compressed = await clients.compress(plan, retrieval["evidence"])
                evidence = compressed["evidence"]
                yield "compressed", {"attempt": attempt, "evidence": len(evidence),
                                     **{k: compressed[k] for k in ("dropped", "merged", "extracted", "tokens")},
                                     **_dbg(debug, "sent_to_llm", evidence)}

            # STEP 9
            yield "generating", {"attempt": attempt, "evidence": len(evidence), "direct": direct}
            answer = await clients.generate(plan, evidence, direct, answer_style)

            # STEP 10
            validation = await clients.validate(plan, evidence, answer, direct)
            verdict = validation["verdict"]
            yield "validated", {"attempt": attempt, **verdict}

            result = {"answer_markdown": validation["answer_markdown"],
                      "citations": _citation_links(answer.get("citations", [])),
                      "confidence": answer.get("confidence", 0.0),
                      "verdict": verdict, "action": verdict["action"]}
            if verdict["action"] in {"accept", "blocked"}:
                break
            if best is None or verdict["grounded_score"] > best["verdict"]["grounded_score"]:
                best = result
            if direct:
                break

        assert result is not None
        if result["action"] not in {"accept", "blocked"}:
            if best and best["verdict"]["grounded_score"] >= FALLBACK_MIN_GROUNDED:
                result = {**best, "action": "accept_low_confidence"}
            elif plan.get("clarifying_question"):
                result = {**result, "answer_markdown": plan["clarifying_question"],
                          "citations": [], "action": "clarify"}
            else:
                result = {**result, "answer_markdown": _insufficient(plan, routes, route_status),
                          "citations": [], "action": "insufficient"}

        if external and "mcp" not in routes:
            routes = [*routes, "mcp"]
        yield "answer", {**result, "routes": routes, "attempts": attempt + 1,
                         "elapsed_sec": round(time.perf_counter() - t0, 2)}
        log.info("[pipeline] intent=%s routes=%s action=%s attempts=%d %.2fs",
                 plan.get("intent"), routes, result["action"], attempt + 1,
                 time.perf_counter() - t0)
    except ServiceError as e:
        log.warning("[pipeline] %s", e)
        yield "error", {"service": e.service, "detail": e.detail}
