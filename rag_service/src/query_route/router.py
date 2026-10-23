"""STEP 5 — query routing: decide where the evidence comes from.

Not every question belongs in the vector database. Cheap, predictable rules
handle the clear cases; the LLM (llm_service /tasks/route) decides the rest.

Routes:
    docs     uploaded documents (hybrid retrieval)
    catalog  metadata about the uploaded documents (counts, lists, types)
    gdrive   Google Drive via the gdrive MCP server (allowed roles only)
    web      public web via the websearch MCP server
    mcp      tool results from MCP servers the user connected in the UI
    direct   no retrieval (chitchat / general questions)

The UI can force routes (`route_hint`, e.g. the /web command) or switch
routes off (`deny_routes`); both are still limited to what the user is
allowed to use, so the UI can never widen Drive access.
"""

from __future__ import annotations

import logging
import os
import re

import httpx
import llm_api
from identity import Identity
from mcp_servers import gdrive_api
from schemas import QueryPlan, Route, RouteResponse

log = logging.getLogger("rag_service.router")

WEBSEARCH_ENABLED = os.getenv("WEBSEARCH_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GDRIVE_ENABLED = os.getenv("GDRIVE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
# The Drive MCP connects one Google account (the operator's), so only these
# app roles may search it.
GDRIVE_ALLOWED_ROLES = {
    r.strip() for r in os.getenv("GDRIVE_ALLOWED_ROLES", "master,admin").split(",") if r.strip()
}

_DRIVE_RE = re.compile(
    r"\b(google\s*drive|g\s*drive|my\s+drive|google\s+(docs?|sheets?|slides?))\b", re.IGNORECASE
)
_WEB_RE = re.compile(
    r"\b(latest|news|today|tonight|yesterday|this\s+(week|month|year)|current(ly)?|"
    r"recent(ly)?|right\s+now|price\s+of|weather|on\s+the\s+(web|internet)|"
    r"search\s+(the\s+)?(web|online|internet)|online)\b",
    re.IGNORECASE,
)


def available_routes(identity: Identity) -> list[Route]:
    routes: list[Route] = ["docs", "catalog", "direct"]
    if GDRIVE_ENABLED and identity.role in GDRIVE_ALLOWED_ROLES and gdrive_api.is_configured():
        routes.append("gdrive")
    if WEBSEARCH_ENABLED:
        routes.append("web")
    return routes


def _rules(plan: QueryPlan, available: list[Route]) -> tuple[list[Route], str] | None:
    text = f"{plan.standalone_question} {plan.rewritten}"
    if plan.intent == "chitchat":
        return ["direct"], "small talk needs no retrieval"
    if plan.intent == "catalog":
        return ["catalog"], "question is about the document collection itself"
    if _DRIVE_RE.search(text):
        if "gdrive" in available:
            return ["gdrive"], "user asked about Google Drive"
        return ["docs"], "Google Drive is not available for this user; searching documents"
    if _WEB_RE.search(text) and "web" in available:
        return ["web", "docs"], "time-sensitive or explicitly web question"
    return None


async def decide(
    plan: QueryPlan,
    identity: Identity,
    route_hint: list[Route] | None = None,
    deny_routes: list[Route] | None = None,
) -> RouteResponse:
    deny = set(deny_routes or [])
    available: list[Route] = [r for r in available_routes(identity) if r not in deny]
    forced: list[Route] = [r for r in dict.fromkeys(route_hint or []) if r in available]
    if forced:
        return RouteResponse(
            routes=forced, reason="route chosen by the user", decided_by="user", available=available
        )
    if route_hint:
        # The requested route isn't permitted (or was switched off).
        return RouteResponse(
            routes=["docs"],
            reason=f"{', '.join(route_hint)} not available; searching documents",
            decided_by="default",
            available=available,
        )
    ruled = _rules(plan, available)
    if ruled:
        routes, reason = ruled
        return RouteResponse(routes=routes, reason=reason, decided_by="rules", available=available)
    try:
        out = await llm_api.route(plan.model_dump(), available)
        routes: list[Route] = [r for r in out.get("routes", []) if r in available] or ["docs"]
        return RouteResponse(
            routes=routes, reason=out.get("reason", ""), decided_by="llm", available=available
        )
    except httpx.HTTPError as e:
        log.warning("[route] llm routing failed, defaulting to docs: %s", e)
        return RouteResponse(
            routes=["docs"],
            reason="default route (router LLM unavailable)",
            decided_by="default",
            available=available,
        )
