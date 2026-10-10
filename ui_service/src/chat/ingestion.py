"""Ingest files dropped into the chat.

For every attached file:
    1. ocr_service /ocr        Docling conversion (CPU) -> document id
    2. rag_service /ingest/id  polled until indexing finishes (chunks, type, topic)
    3. actions                 Share with everyone · Summarize

Progress is shown live in one Chainlit step per file. Questions sent with
the files are answered after ingestion, so they can use the new documents.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import chainlit as cl

from chat import backend

INGEST_TIMEOUT_SEC = int(os.getenv("CHAT_INGEST_TIMEOUT_SEC", "900"))
INGEST_CONCURRENCY = int(os.getenv("CHAT_INGEST_CONCURRENCY", "2"))
POLL_SEC = 2.0


def uploaded_files(message: cl.Message) -> list:
    return [el for el in (message.elements or []) if getattr(el, "path", None)]


def doc_actions(doc_id: str, name: str, shared: bool) -> list[cl.Action]:
    return [
        cl.Action(name="toggle_share",
                  payload={"doc_id": doc_id, "name": name, "shared": shared},
                  label="Make private" if shared else "Share with everyone",
                  icon="lock" if shared else "users",
                  tooltip="Shared documents are searchable by every user"),
        cl.Action(name="summarize_document", payload={"doc_id": doc_id, "name": name},
                  label="Summarize", icon="file-text"),
    ]


async def _wait_for_index(doc_id: str, step: cl.Step, head: str) -> dict:
    deadline = time.monotonic() + INGEST_TIMEOUT_SEC
    while time.monotonic() < deadline:
        entry = await backend.index_status(doc_id)
        status = (entry or {}).get("status")
        if status in {"ok", "error"}:
            return entry
        step.output = f"{head}\n- ⏳ Indexing (chunking, metadata, embeddings)…"
        await step.update()
        await asyncio.sleep(POLL_SEC)
    return {"status": "timeout"}


async def _ingest_one(el, headers: dict, sem: asyncio.Semaphore) -> dict:
    name = el.name or Path(el.path).name
    async with sem, cl.Step(name=f"Ingest {name}", type="tool") as step:
        step.input = name
        if Path(name).suffix.lower() not in backend.SUPPORTED_UPLOADS:
            step.is_error = True
            step.output = "Unsupported file type (PDF, DOCX, PNG, JPG)."
            return {"name": name, "status": "unsupported"}

        step.output = "- ⏳ Docling OCR…"
        await step.update()
        try:
            doc = await backend.ocr_upload(el.path, name, headers)
        except backend.BackendError as e:
            step.is_error = True
            step.output = f"- ❌ OCR failed: {e.detail}"
            return {"name": name, "status": "error", "error": e.detail}

        doc_id = doc["document_id"]
        head = (f"- ✅ OCR: {doc.get('page_count') or '?'} page(s), "
                f"{doc.get('scan_type', '')}, {doc.get('elapsed_sec', '?')}s")
        entry = await _wait_for_index(doc_id, step, head)

        if entry.get("status") == "ok":
            step.output = (f"{head}\n- ✅ Indexed **{entry.get('chunk_count', 0)}** chunks "
                           f"({entry.get('table_chunks', 0)} tables) · "
                           f"type **{entry.get('doc_type', '?')}** · topic _{entry.get('topic', '')}_")
            return {"name": name, "doc_id": doc_id, "status": "ok", "entry": entry}
        step.is_error = True
        reason = entry.get("error") or ("timed out" if entry.get("status") == "timeout" else "failed")
        step.output = f"{head}\n- ❌ Indexing {reason}"
        return {"name": name, "doc_id": doc_id, "status": "error", "error": reason}


async def ingest(files: list, headers: dict) -> list[dict]:
    sem = asyncio.Semaphore(max(1, INGEST_CONCURRENCY))
    results = await asyncio.gather(*(_ingest_one(el, headers, sem) for el in files))

    for res in results:
        if res["status"] != "ok":
            continue
        entry = res["entry"]
        summary = entry.get("summary") or ""
        await cl.Message(
            content=(f"📄 **{res['name']}** is ready to ask about "
                     f"({entry.get('doc_type', 'document')}, {entry.get('chunk_count', 0)} chunks)."
                     + (f"\n\n_{summary}_" if summary else "")
                     + f"\n\n[Open document ↗](/documents/{res['doc_id']}/source)"),
            actions=doc_actions(res["doc_id"], res["name"], shared=False),
        ).send()
    return results
