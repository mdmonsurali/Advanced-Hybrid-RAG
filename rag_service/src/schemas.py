"""Contracts shared with llm_service and orchestrator_service.

QueryPlan / Evidence mirror llm_service/src/schemas.py field for field so the
orchestrator can pass them between services unchanged.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal["fact", "comparison", "troubleshooting", "policy",
                 "summary", "catalog", "chitchat"]
Route = Literal["docs", "catalog", "gdrive", "web", "mcp", "direct"]
RouteStatus = Literal["ok", "empty", "unavailable", "skipped"]


class Filters(BaseModel):
    doc_type: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class QueryPlan(BaseModel):
    standalone_question: str
    intent: Intent
    entities: list[str] = Field(default_factory=list)
    filters: Filters = Field(default_factory=Filters)
    language: str = "en"
    ambiguous: bool = False
    clarifying_question: Optional[str] = None
    rewritten: str = ""
    expansions: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    hyde_passage: str = ""


class Evidence(BaseModel):
    id: str
    route: Route
    text: str
    score: float = 0.0
    source: str = ""
    doc_id: Optional[str] = None
    page: Optional[int] = None
    url: Optional[str] = None
    date: Optional[str] = None
    headings: list[str] = Field(default_factory=list)
    chunk_index: Optional[int] = None
    owned: bool = False


class RouteRequest(BaseModel):
    plan: QueryPlan
    route_hint: list[Route] = Field(default_factory=list)   # user forced routes (/web, /drive, ...)
    deny_routes: list[Route] = Field(default_factory=list)  # routes the user switched off


class RouteResponse(BaseModel):
    routes: list[Route]
    reason: str
    decided_by: Literal["rules", "llm", "default", "user"]
    available: list[Route]


class RetrieveRequest(BaseModel):
    plan: QueryPlan
    routes: list[Route]
    broaden: bool = False
    # Results of user-connected MCP tools (run by the UI), reranked with the rest.
    external_evidence: list[Evidence] = Field(default_factory=list)
    debug: bool = False   # admins only: also return the pre-rerank candidates


class RetrieveStats(BaseModel):
    candidates: int = 0
    reranked: int = 0
    per_route: dict[str, int] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    evidence: list[Evidence]
    route_status: dict[str, RouteStatus]
    stats: RetrieveStats
    candidates: list[Evidence] = Field(default_factory=list)   # first-stage hits (debug only)


class AccessUpdate(BaseModel):
    access_level: Literal["private", "shared"]
