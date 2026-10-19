"""Langfuse tracing (langfuse_service): one trace per chat question.

The orchestrator opens the root observation `rag-chat`. rag_service and
llm_service add their steps (route, retrieve, compress, generate, validate)
under it from two internal headers sent on every call to them:

    X-Langfuse-Trace-Id        32 hex chars
    X-Langfuse-Parent-Span-Id  16 hex chars, the observation that made the call

so one question shows up in Langfuse as one tree across the three services.
After step 10 the verdict is attached to the trace as scores.

Tracing is off when LANGFUSE_TRACING_ENABLED=false or the API keys are
missing. The SDK then hands out no-op observations and nothing here fails.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

log = logging.getLogger("orchestrator.tracing")

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

# Span input/output is for reading in the UI; full prompts and chunk texts
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


def current_trace_id() -> str | None:
    return langfuse.get_current_trace_id() if ENABLED else None


@contextmanager
def trace_question(question: str, user_id: str, role: str, session_id: str | None,
                   options: dict) -> Iterator[Any]:
    """Root observation of one chat question, with user and session set on
    every observation below it."""
    with langfuse.start_as_current_observation(
        name="rag-chat", as_type="chain", trace_context=new_trace(),
        input={"question": question,
               "route_hint": options.get("route_hint") or [],
               "deny_routes": options.get("deny_routes") or [],
               "answer_style": options.get("answer_style"),
               "external_evidence": len(options.get("external_evidence") or [])},
    ) as root, propagate_attributes(
        trace_name="rag-chat", user_id=user_id, session_id=session_id,
        tags=["chat", f"role:{role}"],
    ):
        try:
            yield root
        except Exception as e:
            root.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:500])
            raise


def score_answer(trace_id: str | None, answer: dict) -> None:
    """Validation outcome (STEP 10) as scores on the trace."""
    if not ENABLED or not trace_id:
        return
    verdict = answer.get("verdict") or {}
    scores: list[tuple[str, Any, str]] = [
        ("outcome", answer.get("action") or "unknown", "CATEGORICAL"),
        ("attempts", float(answer.get("attempts") or 0), "NUMERIC"),
    ]
    if verdict:
        scores.append(("groundedness", float(verdict.get("grounded_score") or 0.0), "NUMERIC"))
        scores.append(("citations_valid", 1.0 if verdict.get("citation_ok") else 0.0, "BOOLEAN"))
    if answer.get("action") not in {"clarify", "insufficient"}:
        scores.append(("answer_confidence", float(answer.get("confidence") or 0.0), "NUMERIC"))
    for name, value, data_type in scores:
        try:
            langfuse.create_score(trace_id=trace_id, name=name, value=value, data_type=data_type,
                                  comment=verdict.get("reason") if name == "outcome" else None)
        except Exception as e:  # scores are best-effort
            log.warning("[tracing] score %s not sent: %s", name, e)


def shutdown() -> None:
    if ENABLED:
        langfuse.shutdown()
