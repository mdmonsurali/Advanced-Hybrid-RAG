"""httpx client for the llm_service task endpoints rag_service needs
(STEP 1 figure descriptions, STEP 2 document metadata, STEP 5 routing
fallback)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import httpx
import tracing

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8005")
LLM_API_TIMEOUT = float(os.getenv("LLM_API_TIMEOUT", "120"))


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=LLM_API_TIMEOUT) as client:
        resp = await client.post(
            f"{LLM_SERVICE_URL}{path}", json=payload, headers=tracing.outgoing_headers()
        )
    resp.raise_for_status()
    return resp.json()


async def doc_metadata(name: str, text: str) -> dict:
    return await _post("/tasks/doc-metadata", {"name": name, "text": text})


async def describe_figure(image_b64: str, mime: str, caption: str, context: str) -> dict:
    return await _post(
        "/tasks/describe-figure",
        {"image_b64": image_b64, "mime": mime, "caption": caption, "context": context},
    )


async def route(plan: dict, available: Sequence[str]) -> dict:
    return await _post("/tasks/route", {"plan": plan, "available_routes": available})


async def healthy() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{LLM_SERVICE_URL}/health")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False
