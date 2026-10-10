"""rag_service — the data side of the Advanced Hybrid RAG pipeline.

STEP 1-2  ingestion/ + chunking_techniques/ + embeddings/ + vectordb/
STEP 5    query_route/ (rules + llm_service fallback, concurrent executors)
STEP 6-7  hybrid_retrieval/ (Qdrant dense + BM25, RRF, cross-encoder rerank)
MCP       mcp_servers/ (FastMCP Google Drive + web search, mounted below)

Exposes:
    GET    /health                     qdrant + minio + models + MCP mounted
    POST   /ingest/{doc_id}            (re)index one document in the background   [ocr_service]
    DELETE /ingest/{doc_id}            drop a document's chunks                     [ocr_service]
    PATCH  /ingest/{doc_id}/access     {"access_level": "private"|"shared"}        [ocr_service]
    GET    /ingest/{doc_id}            index entry: status, chunks, doc_type, topic, summary
    POST   /ingest/sync                backfill missing / stale, drop orphans (?upgrade_heuristic=true)
    POST   /route                      STEP 5    {plan}                      -> {routes, reason}
    POST   /retrieve                   STEP 5-7  {plan, routes, broaden, external_evidence} -> reranked evidence
    /mcp/gdrive/                       FastMCP streamable HTTP (Bearer MCP_TOKEN)
    /mcp/websearch/                    FastMCP streamable HTTP (Bearer MCP_TOKEN)

/ingest/* is called by ocr_service over the docker network; /route and
/retrieve need the X-User-Id / X-User-Role identity headers.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from starlette.responses import JSONResponse

import storage
from embeddings import dense, reranker, sparse
from identity import Identity, require_user
from ingestion import pipeline
from mcp_servers import gdrive_server, websearch_server
from query_route import router
from query_route.executors import execute
from schemas import AccessUpdate, RetrieveRequest, RetrieveResponse, RouteRequest, RouteResponse
from vectordb import qdrant_store

log = logging.getLogger("rag_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# One INFO line per outbound HTTP request (MCP calls, web fetches) drowns the
# pipeline logs.
for _noisy in ("httpx", "httpx2", "primp", "mcp", "fastmcp"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

MCP_TOKEN = os.getenv("MCP_TOKEN", "")
SYNC_ON_STARTUP = os.getenv("RAG_SYNC_ON_STARTUP", "true").strip().lower() in {"1", "true", "yes", "on"}


# ── MCP apps (FastMCP streamable HTTP, mounted under /mcp/*) ─────────────

gdrive_mcp_app = gdrive_server.mcp.http_app(path="/")
websearch_mcp_app = websearch_server.mcp.http_app(path="/")


class _BearerGuard:
    """Reject MCP requests without `Authorization: Bearer $MCP_TOKEN`."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and MCP_TOKEN:
            headers = dict(scope.get("headers") or [])
            supplied = headers.get(b"authorization", b"").decode()
            if not hmac.compare_digest(supplied, f"Bearer {MCP_TOKEN}"):
                await JSONResponse({"detail": "invalid MCP token"}, status_code=401)(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ── Lifespan ─────────────────────────────────────────────────────────────

_ready = {"models": False, "qdrant": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        # FastMCP session managers must run for the mounted apps to work.
        await stack.enter_async_context(gdrive_mcp_app.lifespan(gdrive_mcp_app))
        await stack.enter_async_context(websearch_mcp_app.lifespan(websearch_mcp_app))

        await asyncio.gather(asyncio.to_thread(dense.load), asyncio.to_thread(sparse.load),
                             asyncio.to_thread(reranker.load))
        _ready["models"] = True

        qdrant_store.init_client()
        await qdrant_store.ensure_collection(dense.dim())
        _ready["qdrant"] = True

        storage.init_minio()
        await storage.load_registry()
        pipeline.init()

        if SYNC_ON_STARTUP:
            asyncio.create_task(_startup_sync())
        if not MCP_TOKEN:
            log.warning("MCP_TOKEN is empty: /mcp/* endpoints accept unauthenticated requests")
        log.info("rag_service ready")
        yield


async def _startup_sync() -> None:
    try:
        await pipeline.sync()
    except Exception:
        log.exception("startup sync failed")


app = FastAPI(title="Advanced Hybrid RAG — RAG Service", version="1.0", lifespan=lifespan)
app.mount("/mcp/gdrive", _BearerGuard(gdrive_mcp_app))
app.mount("/mcp/websearch", _BearerGuard(websearch_mcp_app))


class _SilencePathsFilter(logging.Filter):
    """Drop uvicorn access lines for /health so the actual events show."""
    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__()
        self._paths = paths

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f' {p} ' in msg for p in self._paths)


logging.getLogger("uvicorn.access").addFilter(_SilencePathsFilter(("/health",)))


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    if not _ready["models"]:
        raise HTTPException(503, "models not loaded")
    try:
        await qdrant_store.health()
    except Exception as e:
        raise HTTPException(503, f"qdrant not reachable: {e}")
    try:
        storage.health_minio()
    except Exception as e:
        raise HTTPException(503, f"minio not reachable: {e}")
    return {"status": "ok", "qdrant": "ok", "minio": "ok", "models": "ok",
            "indexed_documents": len(storage.REGISTRY)}


@app.post("/ingest/sync")
async def ingest_sync(upgrade_heuristic: bool = False):
    return await pipeline.sync(upgrade_heuristic=upgrade_heuristic)


@app.post("/ingest/{doc_id}", status_code=202)
async def ingest_document(doc_id: str):
    meta = await storage.load_meta(doc_id)
    if meta is None:
        raise HTTPException(404, "Document not found")
    if meta.get("status") != "ok":
        raise HTTPException(409, f"Document is not finished (status={meta.get('status')})")
    queued = pipeline.schedule(doc_id)
    return {"document_id": doc_id, "queued": queued}


@app.delete("/ingest/{doc_id}", status_code=204)
async def remove_document(doc_id: str):
    await pipeline.remove(doc_id)


@app.patch("/ingest/{doc_id}/access")
async def update_access(doc_id: str, body: AccessUpdate):
    await pipeline.set_access(doc_id, body.access_level)
    return {"document_id": doc_id, "access_level": body.access_level}


@app.get("/ingest/{doc_id}")
async def index_entry(doc_id: str):
    entry = storage.REGISTRY.get(doc_id)
    if entry is None:
        raise HTTPException(404, "Document is not indexed")
    count = await qdrant_store.count_document(doc_id)
    return {**entry, "points": count}


@app.post("/route", response_model=RouteResponse)
async def route(body: RouteRequest, identity: Identity = Depends(require_user)):
    return await router.decide(body.plan, identity, body.route_hint, body.deny_routes)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(body: RetrieveRequest, identity: Identity = Depends(require_user)):
    allowed = router.available_routes(identity) + ["mcp"]
    routes = [r for r in body.routes if r in allowed]
    if not routes and not body.external_evidence:
        raise HTTPException(400, f"no permitted routes in {body.routes} (allowed: {allowed})")
    # Tool output comes from arbitrary servers: never treat it as the user's
    # own data, so validation keeps masking any PII it contains.
    external = [ev.model_copy(update={"owned": False}) for ev in body.external_evidence]
    return await execute(body.plan, routes, identity, body.broaden, external)
