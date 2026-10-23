"""Request/response contracts for the llm_service task API.

The *Output models double as the JSON schemas handed to SGLang
(`response_format=json_schema`), so they stay flat and simple: xgrammar
compiles them into a grammar and the model can only emit matching JSON.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "fact", "comparison", "troubleshooting", "policy", "summary", "catalog", "chitchat"
]
Route = Literal["docs", "catalog", "gdrive", "web", "mcp", "direct"]
AnswerStyle = Literal["auto", "concise", "detailed"]
Action = Literal["accept", "retry", "clarify", "insufficient", "blocked"]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Filters(BaseModel):
    doc_type: str | None = None
    date_from: str | None = None  # ISO date
    date_to: str | None = None  # ISO date


# ── LLM output schemas ──────────────────────────────────────────────────


class UnderstandingOutput(BaseModel):
    """STEP 3 — what the user is really asking."""

    standalone_question: str
    intent: Intent
    entities: list[str] = Field(default_factory=list, max_length=8)
    filters: Filters = Field(default_factory=Filters)
    language: str = "en"
    ambiguous: bool = False
    clarifying_question: str | None = None


class TransformationOutput(BaseModel):
    """STEP 4 — search-ready rewrites of the question."""

    rewritten: str
    expansions: list[str] = Field(default_factory=list, max_length=3)
    sub_questions: list[str] = Field(default_factory=list, max_length=3)
    hyde_passage: str = ""


class RouteOutput(BaseModel):
    """STEP 5 (LLM fallback) — where the evidence should come from."""

    routes: list[Route] = Field(min_length=1, max_length=4)
    reason: str


class DocMetadataOutput(BaseModel):
    """STEP 2 — document-level metadata attached to every chunk."""

    doc_type: str
    topic: str
    summary: str
    language: str


class AnswerOutput(BaseModel):
    """STEP 9 — the structured answer the model must produce."""

    answer_markdown: str
    citations: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    insufficient_evidence: bool = False


class Claim(BaseModel):
    claim: str
    supported: bool
    evidence: list[int] = Field(default_factory=list)


class JudgeOutput(BaseModel):
    """STEP 10 — claim-level grounding + safety judgement."""

    claims: list[Claim] = Field(default_factory=list, max_length=20)
    unsafe: bool = False
    policy_violation: bool = False


class ExtractOutput(BaseModel):
    """STEP 8 — extractive compression of one long evidence chunk."""

    relevant_text: str


# ── Shared pipeline objects ─────────────────────────────────────────────


class QueryPlan(UnderstandingOutput, TransformationOutput):
    pass


class Evidence(BaseModel):
    id: str
    route: Route
    text: str
    score: float = 0.0
    source: str = ""
    doc_id: str | None = None
    page: int | None = None
    url: str | None = None
    date: str | None = None
    headings: list[str] = Field(default_factory=list)
    chunk_index: int | None = None
    owned: bool = False  # evidence from the asking user's own data


class Citation(BaseModel):
    n: int
    evidence_id: str
    route: Route
    source: str
    doc_id: str | None = None
    page: int | None = None
    url: str | None = None
    snippet: str = ""  # evidence excerpt for the UI's source panel


class Verdict(BaseModel):
    action: Action
    reason: str
    grounded_score: float
    citation_ok: bool
    schema_ok: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    pii_masked: int = 0
    unsafe: bool = False


# ── Task requests / responses ───────────────────────────────────────────


class QueryAnalysisRequest(BaseModel):
    question: str
    history: list[Turn] = Field(default_factory=list)


class RouteRequest(BaseModel):
    plan: QueryPlan
    available_routes: list[Route]


class DocMetadataRequest(BaseModel):
    name: str
    text: str


class DescribeFigureRequest(BaseModel):
    image_b64: str
    mime: Literal["image/jpeg", "image/png"] = "image/jpeg"
    caption: str = ""
    context: str = ""


class DescribeFigureOutput(BaseModel):
    """STEP 1 — what a figure shows, indexed with its caption."""

    description: str


class CompressRequest(BaseModel):
    plan: QueryPlan
    evidence: list[Evidence]


class CompressResponse(BaseModel):
    evidence: list[Evidence]
    dropped: int
    merged: int
    extracted: int
    tokens: int


class GenerateRequest(BaseModel):
    plan: QueryPlan
    evidence: list[Evidence]
    direct: bool = False  # answer without retrieval (chitchat / general)
    answer_style: AnswerStyle = "auto"


class GenerateResponse(BaseModel):
    answer_markdown: str
    citations: list[Citation]
    confidence: float
    insufficient_evidence: bool
    schema_ok: bool = True


class ValidateRequest(BaseModel):
    plan: QueryPlan
    evidence: list[Evidence]
    answer: GenerateResponse
    direct: bool = False


class ValidateResponse(BaseModel):
    answer_markdown: str  # PII-masked
    verdict: Verdict


# ── User MCP tool selection ─────────────────────────────────────────────


class ToolSpec(BaseModel):
    server: str
    name: str
    description: str = ""
    input_schema: dict = Field(default_factory=dict)


class SelectToolsRequest(BaseModel):
    question: str
    tools: list[ToolSpec]


class ToolCall(BaseModel):
    server: str
    name: str
    arguments_json: str = "{}"  # JSON object as a string (keeps the grammar simple)


class SelectToolsOutput(BaseModel):
    """Which user-connected MCP tools to call for a question (may be none)."""

    calls: list[ToolCall] = Field(default_factory=list, max_length=3)
    reason: str = ""
