"""👍/👎 feedback → `user_feedback` score on the answer's Langfuse trace.

Every answer message carries the trace id the orchestrator returned
(metadata.langfuse_trace_id). The score reuses the Chainlit feedback id, so
changing a vote updates the same score and removing it deletes the score.

Calls Langfuse's public REST API directly (no SDK in this service). Scores
are best-effort: a Langfuse outage never breaks the chat.
"""

from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger("ui_service.feedback_scores")

BASE_URL = os.getenv("LANGFUSE_BASE_URL", "http://langfuse_service:3000").rstrip("/")
PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
# Same environment as the traces, so environment filters keep the scores.
ENVIRONMENT = os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "default")
ENABLED = (os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
           and bool(PUBLIC_KEY) and bool(SECRET_KEY))
TIMEOUT = httpx.Timeout(5.0)

SCORE_NAME = "user_feedback"


async def send(score_id: str, trace_id: str | None, value: int, comment: str | None) -> None:
    if not ENABLED or not trace_id:
        return
    body = {"id": score_id, "traceId": trace_id, "name": SCORE_NAME, "value": 1 if value else 0,
            "dataType": "BOOLEAN", "comment": comment or None, "environment": ENVIRONMENT}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, auth=(PUBLIC_KEY, SECRET_KEY)) as client:
            resp = await client.post(f"{BASE_URL}/api/public/scores", json=body)
        if resp.status_code >= 400:
            log.warning("[feedback] score not saved: HTTP %s %s", resp.status_code, resp.text[:200])
    except httpx.HTTPError as e:
        log.warning("[feedback] Langfuse unreachable: %s", e.__class__.__name__)


async def delete(score_id: str) -> None:
    if not ENABLED:
        return
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, auth=(PUBLIC_KEY, SECRET_KEY)) as client:
            resp = await client.delete(f"{BASE_URL}/api/public/scores/{score_id}")
        if resp.status_code >= 400 and resp.status_code != 404:
            log.warning("[feedback] score not deleted: HTTP %s", resp.status_code)
    except httpx.HTTPError as e:
        log.warning("[feedback] Langfuse unreachable: %s", e.__class__.__name__)
