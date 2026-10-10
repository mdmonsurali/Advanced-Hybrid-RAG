"""orchestrator_service — drives the Advanced Hybrid RAG query pipeline.

Calls llm_service (STEP 3, 4, 8, 9, 10) and rag_service (STEP 5, 6, 7) in
order, owns the validation retry loop, and streams every step to the UI.

Exposes:
    GET  /health                    llm_service + rag_service reachability
    POST /chat   {question, history, options?}   SSE: one event per pipeline step, then `answer`
    POST /query  {question, history, options?}   JSON: final answer + the step trace

options: {route_hint: [..], deny_routes: [..], answer_style, external_evidence: [Evidence]}
"""

from __future__ import annotations

import json as _json
import logging

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import clients
import pipeline
from identity import Identity, require_user

log = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_QUESTION_CHARS = 4000

app = FastAPI(title="Advanced Hybrid RAG — Orchestrator", version="1.0")


class _SilencePathsFilter(logging.Filter):
    """Drop uvicorn access lines for /health so the actual events show."""
    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__()
        self._paths = paths

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f' {p} ' in msg for p in self._paths)


logging.getLogger("uvicorn.access").addFilter(_SilencePathsFilter(("/health",)))


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {_json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


_ROUTES = {"docs", "catalog", "gdrive", "web", "mcp", "direct"}
MAX_EXTERNAL_EVIDENCE = 30
MAX_EXTERNAL_CHARS = 20000


def _options(payload: dict) -> dict:
    raw = payload.get("options") or {}
    if not isinstance(raw, dict):
        raise HTTPException(400, "options must be an object")
    style = raw.get("answer_style") or "auto"
    if style not in {"auto", "concise", "detailed"}:
        raise HTTPException(400, "answer_style must be auto, concise or detailed")
    external = []
    for ev in (raw.get("external_evidence") or [])[:MAX_EXTERNAL_EVIDENCE]:
        if not isinstance(ev, dict) or not ev.get("id") or not ev.get("text"):
            continue
        external.append({"id": str(ev["id"]), "route": "mcp", "text": str(ev["text"])[:MAX_EXTERNAL_CHARS],
                         "source": str(ev.get("source") or "MCP tool"), "url": ev.get("url"),
                         "owned": False})
    return {
        "route_hint": [r for r in raw.get("route_hint") or [] if r in _ROUTES],
        "deny_routes": [r for r in raw.get("deny_routes") or [] if r in _ROUTES],
        "answer_style": style,
        "external_evidence": external,
    }


def _parse(payload: dict) -> tuple[str, list[dict]]:
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(400, f"question longer than {MAX_QUESTION_CHARS} characters")
    history = payload.get("history") or []
    if not isinstance(history, list):
        raise HTTPException(400, "history must be a list of {role, content}")
    history = [{"role": h.get("role"), "content": str(h.get("content", ""))[:MAX_QUESTION_CHARS]}
               for h in history if isinstance(h, dict)]
    return question, history


@app.get("/health")
async def health():
    llm_ok = await clients.healthy(clients.LLM_SERVICE_URL)
    rag_ok = await clients.healthy(clients.RAG_SERVICE_URL)
    status = {"status": "ok" if llm_ok and rag_ok else "degraded",
              "llm_service": "ok" if llm_ok else "down",
              "rag_service": "ok" if rag_ok else "down"}
    # The orchestrator itself is up either way; the UI shows the degraded parts.
    return status


@app.post("/chat")
async def chat(payload: dict = Body(...), identity: Identity = Depends(require_user)):
    question, history = _parse(payload)
    options = _options(payload)

    async def stream():
        async for event, data in pipeline.run(question, history, identity, options):
            yield _sse(event, data)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


@app.post("/query")
async def query(payload: dict = Body(...), identity: Identity = Depends(require_user)):
    question, history = _parse(payload)
    options = _options(payload)
    trace = []
    final = None
    async for event, data in pipeline.run(question, history, identity, options):
        if event == "answer":
            final = data
        elif event == "error":
            raise HTTPException(502, f"{data['service']}: {data['detail']}")
        else:
            trace.append({"event": event, "data": data})
    return {"answer": final, "trace": trace}
