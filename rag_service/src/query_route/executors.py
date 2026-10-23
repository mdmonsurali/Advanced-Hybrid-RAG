"""Run the selected routes concurrently, then rerank the pooled evidence.

docs    -> hybrid_retrieval.search (STEP 6)
gdrive  -> gdrive MCP server
web     -> websearch MCP server
catalog -> index registry (no reranking: it is one computed block)
mcp     -> tool results the UI collected from user-connected MCP servers
direct  -> nothing

All retrieved candidates (docs + Drive + web + MCP tools) go through one reranking pass
(STEP 7). A route that fails is reported as `unavailable`; the others still
answer.
"""

from __future__ import annotations

import asyncio
import logging

import tracing
from chunking_techniques.text import chunk_text
from hybrid_retrieval.rerank import rerank
from hybrid_retrieval.search import search_docs
from identity import Identity
from mcp_servers.client import gdrive_evidence, web_evidence
from schemas import Evidence, QueryPlan, RetrieveResponse, RetrieveStats, RouteStatus

from query_route.catalog import catalog_evidence

log = logging.getLogger("rag_service.executors")


def _external_candidates(external: list[Evidence], question: str) -> list[tuple[Evidence, str]]:
    """Chunk long tool outputs so the reranker sees comparable passages."""
    out = []
    for ev in external:
        pieces = chunk_text(ev.text) or [ev.text]
        for i, piece in enumerate(pieces):
            out.append(
                (
                    ev.model_copy(
                        update={
                            "id": f"{ev.id}:{i}",
                            "route": "mcp",
                            "text": piece,
                            "chunk_index": i,
                        }
                    ),
                    question,
                )
            )
    return out


async def execute(
    plan: QueryPlan,
    routes: list[str],
    identity: Identity,
    broaden: bool = False,
    external: list[Evidence] | None = None,
    debug: bool = False,
) -> RetrieveResponse:
    status: dict[str, RouteStatus] = {}
    per_route: dict[str, int] = {}
    candidates: list[tuple[Evidence, str]] = []
    passthrough: list[Evidence] = []

    async def _docs():
        with tracing.observe(
            "search-docs",
            as_type="retriever",
            input={"queries": len(plan.sub_questions) or 1, "broaden": broaden},
        ) as obs:
            res = await search_docs(plan, identity, broaden)
            obs.update(output=tracing.hits([ev for ev, _ in res]))
            return res

    async def _gdrive():
        with tracing.observe(
            "gdrive", as_type="tool", input={"question": plan.standalone_question}
        ) as obs:
            res = [(ev, plan.standalone_question) for ev in await gdrive_evidence(plan)]
            obs.update(output=tracing.hits([ev for ev, _ in res]))
            return res

    async def _web():
        with tracing.observe(
            "web-search", as_type="tool", input={"question": plan.standalone_question}
        ) as obs:
            res = [(ev, plan.standalone_question) for ev in await web_evidence(plan)]
            obs.update(output=tracing.hits([ev for ev, _ in res]))
            return res

    runners = {"docs": _docs, "gdrive": _gdrive, "web": _web}
    names = [r for r in dict.fromkeys(routes) if r in runners]
    results = await asyncio.gather(*(runners[r]() for r in names), return_exceptions=True)

    for name, res in zip(names, results):
        if isinstance(res, BaseException):
            log.warning("[route:%s] unavailable: %s", name, res)
            status[name] = "unavailable"
            per_route[name] = 0
            continue
        status[name] = "ok" if res else "empty"
        per_route[name] = len(res)
        candidates.extend(res)

    if external:
        ext = _external_candidates(external, plan.standalone_question)
        candidates.extend(ext)
        status["mcp"] = "ok"
        per_route["mcp"] = len(ext)

    if "catalog" in routes:
        cat = catalog_evidence(plan, identity)
        passthrough.extend(cat)
        status["catalog"] = "ok"
        per_route["catalog"] = len(cat)
    if "direct" in routes:
        status["direct"] = "skipped"

    with tracing.observe("rerank", input={"candidates": len(candidates)}) as obs:
        reranked = await rerank(candidates)
        obs.update(output=tracing.hits(reranked))
    evidence = passthrough + reranked
    return RetrieveResponse(
        evidence=evidence,
        route_status=status,
        candidates=(
            sorted((ev for ev, _ in candidates), key=lambda e: e.score, reverse=True)[:150]
            if debug
            else []
        ),
        stats=RetrieveStats(
            candidates=len(candidates), reranked=len(reranked), per_route=per_route
        ),
    )
