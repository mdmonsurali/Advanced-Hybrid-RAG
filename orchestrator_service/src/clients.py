"""httpx clients for rag_service and llm_service."""

from __future__ import annotations

import os

import httpx

import tracing
from identity import Identity, identity_headers

RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag_service:8003")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm_service:8005")
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "120"))
LLM_TIMEOUT = float(os.getenv("LLM_API_TIMEOUT", "300"))


class ServiceError(RuntimeError):
    def __init__(self, service: str, detail: str):
        super().__init__(f"{service}: {detail}")
        self.service = service
        self.detail = detail


async def _post(
    service: str, url: str, payload: dict, timeout: float, headers: dict | None = None
) -> dict:
    # The callee's spans join this question's Langfuse trace.
    headers = {**(headers or {}), **tracing.outgoing_headers()}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as e:
        raise ServiceError(service, f"unreachable ({e.__class__.__name__})") from e
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ServiceError(service, f"HTTP {resp.status_code}: {str(detail)[:300]}")
    return resp.json()


# ── llm_service ─────────────────────────────────────────────────────────


async def query_analysis(question: str, history: list[dict]) -> dict:
    return await _post(
        "llm_service",
        f"{LLM_SERVICE_URL}/tasks/query-analysis",
        {"question": question, "history": history},
        LLM_TIMEOUT,
    )


async def compress(plan: dict, evidence: list[dict]) -> dict:
    return await _post(
        "llm_service",
        f"{LLM_SERVICE_URL}/tasks/compress",
        {"plan": plan, "evidence": evidence},
        LLM_TIMEOUT,
    )


async def generate(
    plan: dict, evidence: list[dict], direct: bool, answer_style: str = "auto"
) -> dict:
    return await _post(
        "llm_service",
        f"{LLM_SERVICE_URL}/tasks/generate",
        {"plan": plan, "evidence": evidence, "direct": direct, "answer_style": answer_style},
        LLM_TIMEOUT,
    )


async def validate(plan: dict, evidence: list[dict], answer: dict, direct: bool) -> dict:
    return await _post(
        "llm_service",
        f"{LLM_SERVICE_URL}/tasks/validate",
        {"plan": plan, "evidence": evidence, "answer": answer, "direct": direct},
        LLM_TIMEOUT,
    )


# ── rag_service ─────────────────────────────────────────────────────────


async def route(
    plan: dict, identity: Identity, route_hint: list[str], deny_routes: list[str]
) -> dict:
    return await _post(
        "rag_service",
        f"{RAG_SERVICE_URL}/route",
        {"plan": plan, "route_hint": route_hint, "deny_routes": deny_routes},
        RAG_TIMEOUT,
        identity_headers(identity),
    )


async def retrieve(
    plan: dict,
    routes: list[str],
    broaden: bool,
    identity: Identity,
    external_evidence: list[dict] | None = None,
    debug: bool = False,
) -> dict:
    return await _post(
        "rag_service",
        f"{RAG_SERVICE_URL}/retrieve",
        {
            "plan": plan,
            "routes": routes,
            "broaden": broaden,
            "external_evidence": external_evidence or [],
            "debug": debug,
        },
        RAG_TIMEOUT,
        identity_headers(identity),
    )


async def healthy(base: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            return (await client.get(f"{base}/health")).status_code == 200
    except httpx.HTTPError:
        return False
