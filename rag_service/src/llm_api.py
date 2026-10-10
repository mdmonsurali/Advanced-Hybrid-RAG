"""httpx client for the llm_service task endpoints rag_service needs
(STEP 2 document metadata, STEP 5 routing fallback)."""

from __future__ import annotations

import os

import httpx

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8005")
LLM_API_TIMEOUT = float(os.getenv("LLM_API_TIMEOUT", "120"))


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=LLM_API_TIMEOUT) as client:
        resp = await client.post(f"{LLM_SERVICE_URL}{path}", json=payload)
    resp.raise_for_status()
    return resp.json()


async def doc_metadata(name: str, text: str) -> dict:
    return await _post("/tasks/doc-metadata", {"name": name, "text": text})


async def route(plan: dict, available: list[str]) -> dict:
    return await _post("/tasks/route", {"plan": plan, "available_routes": available})


async def healthy() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{LLM_SERVICE_URL}/health")
        return resp.status_code == 200
    except httpx.HTTPError:
        return False
