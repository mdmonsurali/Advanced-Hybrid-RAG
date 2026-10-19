"""httpx clients the chat uses. Every user-scoped call carries the
X-User-Id / X-User-Role identity headers, exactly like the page proxies in
main.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncIterator

import httpx

OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr_service:8001")
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://rag_service:8003")
LLM_SERVICE_URL = os.environ.get("LLM_SERVICE_URL", "http://llm_service:8005")
ORCHESTRATOR_SERVICE_URL = os.environ.get("ORCHESTRATOR_SERVICE_URL",
                                          "http://orchestrator_service:8004")

_UPLOAD_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
}
SUPPORTED_UPLOADS = set(_UPLOAD_TYPES)


class BackendError(RuntimeError):
    def __init__(self, service: str, detail: str):
        super().__init__(f"{service}: {detail}")
        self.service = service
        self.detail = detail


def _detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))[:300]
    except ValueError:
        return resp.text[:300]


# ── orchestrator ─────────────────────────────────────────────────────────

async def stream_chat(question: str, history: list[dict], options: dict,
                      headers: dict) -> AsyncIterator[tuple[str, dict]]:
    """Yield (event, data) pairs from orchestrator /chat (SSE)."""
    payload = {"question": question, "history": history, "options": options}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)) as client:
            async with client.stream("POST", f"{ORCHESTRATOR_SERVICE_URL}/chat",
                                     json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise BackendError("orchestrator", f"HTTP {resp.status_code}: {_detail(resp)}")
                event, data = "message", ""
                async for line in resp.aiter_lines():
                    if line.startswith("event: "):
                        event = line[7:].strip()
                    elif line.startswith("data: "):
                        data += line[6:]
                    elif line == "" and data:
                        yield event, json.loads(data)
                        event, data = "message", ""
    except httpx.HTTPError as e:
        raise BackendError("orchestrator", f"unreachable ({e.__class__.__name__})") from e


# ── ocr_service ──────────────────────────────────────────────────────────

async def ocr_upload(path: str, name: str, headers: dict) -> dict:
    ext = Path(name).suffix.lower()
    if ext not in _UPLOAD_TYPES:
        raise BackendError("ocr_service", f"unsupported file type {ext or '(none)'}")
    content = await _read(path)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=10)) as client:
            resp = await client.post(f"{OCR_SERVICE_URL}/ocr", headers=headers,
                                     files={"file": (name, content, _UPLOAD_TYPES[ext])})
    except httpx.HTTPError as e:
        raise BackendError("ocr_service", f"unreachable ({e.__class__.__name__})") from e
    if resp.status_code != 200:
        raise BackendError("ocr_service", f"HTTP {resp.status_code}: {_detail(resp)}")
    return resp.json()


async def _read(path: str) -> bytes:
    import asyncio
    return await asyncio.to_thread(Path(path).read_bytes)


async def get_document(doc_id: str, headers: dict) -> dict | None:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{OCR_SERVICE_URL}/documents/{doc_id}", headers=headers)
    return resp.json() if resp.status_code == 200 else None


async def set_access(doc_id: str, level: str, headers: dict) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.patch(f"{OCR_SERVICE_URL}/documents/{doc_id}/access",
                                  json={"access_level": level}, headers=headers)
    if resp.status_code != 200:
        raise BackendError("ocr_service", f"HTTP {resp.status_code}: {_detail(resp)}")
    return resp.json()


# ── rag_service ──────────────────────────────────────────────────────────

async def index_status(doc_id: str) -> dict | None:
    """rag_service index entry; callers must check ownership via ocr first."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{RAG_SERVICE_URL}/ingest/{doc_id}")
    return resp.json() if resp.status_code == 200 else None


# ── llm_service ──────────────────────────────────────────────────────────

# Calls straight to llm_service first take an `llm` GPU lease from the
# orchestrator, which waits out a GPU swap to OCR (no-op when GPU_SWAP=false).
GPU_WAIT_TIMEOUT = float(os.environ.get("GPU_WAIT_TIMEOUT", "1800"))


async def _gpu_acquire(client: httpx.AsyncClient) -> str | None:
    try:
        resp = await client.post(f"{ORCHESTRATOR_SERVICE_URL}/gpu/acquire/llm",
                                 timeout=httpx.Timeout(GPU_WAIT_TIMEOUT, connect=10))
        return resp.json().get("lease_id") if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError):
        return None     # orchestrator down: call the LLM anyway


async def _gpu_release(client: httpx.AsyncClient, lease_id: str | None) -> None:
    if lease_id and lease_id != "noop":
        try:
            await client.post(f"{ORCHESTRATOR_SERVICE_URL}/gpu/release/{lease_id}", timeout=10)
        except httpx.HTTPError:
            pass        # the lease expires on its own


async def select_tools(question: str, tools: list[dict]) -> dict:
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            lease_id = await _gpu_acquire(client)
            try:
                resp = await client.post(f"{LLM_SERVICE_URL}/tasks/select-tools",
                                         json={"question": question, "tools": tools})
            finally:
                await _gpu_release(client, lease_id)
    except httpx.HTTPError as e:
        raise BackendError("llm_service", f"unreachable ({e.__class__.__name__})") from e
    if resp.status_code != 200:
        raise BackendError("llm_service", f"HTTP {resp.status_code}: {_detail(resp)}")
    return resp.json()
