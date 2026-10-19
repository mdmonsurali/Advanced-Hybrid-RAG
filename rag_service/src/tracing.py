"""Langfuse tracing (langfuse_service).

/route and /retrieve are observations inside the orchestrator's `rag-chat`
trace, joined through two internal headers:

    X-Langfuse-Trace-Id        32 hex chars
    X-Langfuse-Parent-Span-Id  16 hex chars, the caller's observation

Underneath /retrieve each route is its own observation (docs search as a
retriever, Drive / web as tools) followed by the rerank. Calls to
llm_service (routing fallback, document metadata) pass the headers on, so
their LLM generations nest here too. Document ingestion is a trace of its
own (`ingest-document`), owned by the document's owner.

Tracing is off when LANGFUSE_TRACING_ENABLED=false or the API keys are
missing. The SDK then hands out no-op observations and nothing here fails.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

ENABLED = (os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
           and bool(os.getenv("LANGFUSE_PUBLIC_KEY")) and bool(os.getenv("LANGFUSE_SECRET_KEY")))
if not ENABLED:
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

from langfuse import get_client, propagate_attributes  # noqa: E402  (reads the env above)

if not ENABLED:  # off on purpose: skip the SDK's "initialized without public_key" warning
    logging.getLogger("langfuse").setLevel(logging.ERROR)

langfuse = get_client()

TRACE_HEADER = "X-Langfuse-Trace-Id"
PARENT_HEADER = "X-Langfuse-Parent-Span-Id"
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")

# Span input/output is for reading in the UI, not a copy of every chunk.
MAX_TEXT = 300
MAX_ITEMS = 40


def compact(value: Any) -> Any:
    """JSON-friendly copy with long strings and lists cut down."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + f"… (+{len(value) - MAX_TEXT} chars)"
    if isinstance(value, Mapping):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [compact(v) for v in value[:MAX_ITEMS]]
        if len(value) > MAX_ITEMS:
            items.append(f"… (+{len(value) - MAX_ITEMS} more)")
        return items
    return value


def hits(evidence: list) -> list[dict]:
    """What a retrieval step returned, one line per candidate."""
    return [{"id": e.id, "route": e.route, "source": e.source, "page": e.page,
             "score": round(e.score or 0.0, 4)} for e in evidence]


def incoming_context(headers: Mapping[str, str]) -> dict | None:
    trace_id = (headers.get(TRACE_HEADER) or "").strip().lower()
    if not ENABLED or not _HEX32.match(trace_id):
        return None
    ctx = {"trace_id": trace_id}
    parent = (headers.get(PARENT_HEADER) or "").strip().lower()
    if _HEX16.match(parent):
        ctx["parent_span_id"] = parent
    return ctx


def outgoing_headers() -> dict[str, str]:
    """Headers that attach the callee's spans to the current observation."""
    if not ENABLED:
        return {}
    trace_id = langfuse.get_current_trace_id()
    span_id = langfuse.get_current_observation_id()
    return {TRACE_HEADER: trace_id, PARENT_HEADER: span_id} if trace_id and span_id else {}


def new_trace() -> dict:
    """Context for a fresh trace. Needed because FastAPI wraps every request
    in an OpenTelemetry span of its own that Langfuse doesn't export; a root
    observation nested under it would point at a parent that never arrives."""
    return {"trace_id": langfuse.create_trace_id()}


@contextmanager
def observe(name: str, *, as_type: str = "span", input: Any = None, metadata: Any = None,
            headers: Mapping[str, str] | None = None) -> Iterator[Any]:
    """An observation under the current one. With `headers` (the incoming
    request headers) it joins the caller's trace instead, or starts a new one
    when the caller sent none. Errors mark it ERROR."""
    with langfuse.start_as_current_observation(
        name=name, as_type=as_type, input=compact(input), metadata=metadata,
        trace_context=(incoming_context(headers) or new_trace()) if headers is not None else None,
    ) as obs:
        try:
            yield obs
        except Exception as e:
            obs.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:500])
            raise


@contextmanager
def ingest_trace(doc_id: str, source: str, owner_id: str) -> Iterator[Any]:
    """Root of a document-ingestion trace (chunk → metadata → embed → upsert)."""
    with observe("ingest-document", as_type="chain", headers={},
                 input={"doc_id": doc_id, "source": source}) as obs, \
            propagate_attributes(trace_name="ingest-document", user_id=owner_id or None,
                                 tags=["ingestion"], metadata={"doc_id": doc_id}):
        yield obs


def output(obs: Any, value: Any) -> Any:
    """Record `value` as the observation's output and return it."""
    obs.update(output=compact(value))
    return value


def shutdown() -> None:
    if ENABLED:
        langfuse.shutdown()
