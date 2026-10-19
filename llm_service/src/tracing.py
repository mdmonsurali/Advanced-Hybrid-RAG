"""Langfuse tracing (langfuse_service).

Every /tasks/* request is one observation (query-analysis, generate,
validate, ...). It joins the caller's trace through two internal headers:

    X-Langfuse-Trace-Id        32 hex chars
    X-Langfuse-Parent-Span-Id  16 hex chars, the caller's observation

so the orchestrator's `rag-chat` trace shows each step with its LLM calls
underneath. A request without them starts a new trace. The LLM calls
themselves are recorded as generations by the Langfuse OpenAI wrapper in
llm_client/client.py (prompt, output, token usage, latency, prompt version).

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

from langfuse import get_client  # noqa: E402  (reads the env above)

if not ENABLED:  # off on purpose: skip the SDK's "initialized without public_key" warning
    logging.getLogger("langfuse").setLevel(logging.ERROR)

langfuse = get_client()

TRACE_HEADER = "X-Langfuse-Trace-Id"
PARENT_HEADER = "X-Langfuse-Parent-Span-Id"
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")

# Span input/output is for reading in the UI; the full prompts and evidence
# are on the generations.
MAX_TEXT = 500
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


def incoming_context(headers: Mapping[str, str]) -> dict | None:
    trace_id = (headers.get(TRACE_HEADER) or "").strip().lower()
    if not ENABLED or not _HEX32.match(trace_id):
        return None
    ctx = {"trace_id": trace_id}
    parent = (headers.get(PARENT_HEADER) or "").strip().lower()
    if _HEX16.match(parent):
        ctx["parent_span_id"] = parent
    return ctx


def new_trace() -> dict:
    """Context for a fresh trace. Needed because FastAPI wraps every request
    in an OpenTelemetry span of its own that Langfuse doesn't export; a root
    observation nested under it would point at a parent that never arrives."""
    return {"trace_id": langfuse.create_trace_id()}


@contextmanager
def task(name: str, headers: Mapping[str, str], *, as_type: str = "chain",
         input: Any = None) -> Iterator[Any]:
    """One observation for a task request, inside the caller's trace (or a new
    trace when the caller sent none)."""
    with langfuse.start_as_current_observation(
        name=name, as_type=as_type, input=compact(input),
        trace_context=incoming_context(headers) or new_trace(),
    ) as obs:
        try:
            yield obs
        except Exception as e:
            obs.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:500])
            raise


def output(obs: Any, value: Any) -> Any:
    """Record `value` as the observation's output and return it."""
    obs.update(output=compact(value))
    return value


def shutdown() -> None:
    if ENABLED:
        langfuse.shutdown()
