"""STEP 1–2 driver: load → merge cross-page tables → describe figures →
chunk → metadata → embed (dense + BM25) → upsert.

Triggered by ocr_service after a document finishes (POST /ingest/{id}) and
by /ingest/sync (startup backfill + cleanup). Jobs run in the background,
one document at a time, because chunking and embedding are CPU-bound.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone

from docling_core.types.doc import DoclingDocument

import llm_api
import storage
import tracing
from chunking_techniques.hybrid import build_chunker, chunk_document
from chunking_techniques.table_merge import merge_cross_page_tables
from embeddings import dense, sparse
from ingestion import figures, metadata
from vectordb import qdrant_store

log = logging.getLogger("rag_service.ingestion")

INGEST_CONCURRENCY = int(os.getenv("RAG_INGEST_CONCURRENCY", "1"))
# Bump when chunking changes so /ingest/sync re-indexes existing documents.
CHUNKER_VERSION = 2

_sem: asyncio.Semaphore | None = None
_chunker = None
_tasks: set[asyncio.Task] = set()
_running: set[str] = set()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init() -> None:
    """Call after the dense model is loaded (the chunker uses its tokenizer)."""
    global _sem, _chunker
    _sem = asyncio.Semaphore(max(1, INGEST_CONCURRENCY))
    _chunker = build_chunker(dense.tokenizer())


async def ingest(doc_id: str) -> dict:
    """Index one document end-to-end. Returns its registry entry."""
    assert _sem is not None and _chunker is not None, "pipeline.init() not called"
    async with _sem:
        meta = await storage.load_meta(doc_id)
        if meta is None:
            raise LookupError(f"document {doc_id} not found")
        if meta.get("status") != "ok" or not meta.get("json_key"):
            raise ValueError(f"document {doc_id} is not finished (status={meta.get('status')})")

        entry = {
            "doc_id": doc_id,
            "source": meta.get("original_name") or doc_id,
            "owner_id": str(meta.get("owner_id") or ""),
            "access_level": meta.get("access_level") or "private",
            "created_at": meta.get("created_at"),
            "page_count": meta.get("page_count"),
            "status": "indexing",
            "updated_at": _now(),
        }
        await storage.put_index({**storage.REGISTRY.get(doc_id, {}), **entry})

        with tracing.ingest_trace(doc_id, entry["source"], entry["owner_id"]) as obs:
            return tracing.output(obs, await _index(doc_id, meta, entry))


async def _index(doc_id: str, meta: dict, entry: dict) -> dict:
    """STEP 1-2 for one document: chunk → metadata → embed → upsert."""
    try:
        layout = await storage.load_layout(doc_id)
        if layout is None:
            raise LookupError(f"layout.json missing for {doc_id}")
        doc = DoclingDocument.model_validate(layout)

        row_pages = await asyncio.to_thread(merge_cross_page_tables, doc)
        descriptions, figure_status = await figures.describe(doc_id, doc)
        chunks = await asyncio.to_thread(chunk_document, doc, _chunker, descriptions, row_pages)
        doc_meta = await metadata.document_metadata(meta, chunks)
        payloads = metadata.chunk_payloads(meta, doc_meta, chunks)

        embed_texts = [c.embed_text for c in chunks]
        dense_vecs = await asyncio.to_thread(dense.embed, embed_texts)
        sparse_vecs = await asyncio.to_thread(sparse.embed_documents, embed_texts)

        await qdrant_store.delete_document(doc_id)
        if chunks:
            await qdrant_store.upsert_chunks(doc_id, dense_vecs, sparse_vecs, payloads)

        entry.update(doc_meta)
        entry.update({
            "status": "ok",
            "chunk_count": len(chunks),
            "table_chunks": sum(1 for c in chunks if c.chunk_type == "table"),
            "figure_chunks": sum(1 for c in chunks if c.chunk_type == "figure"),
            "merged_tables": sum(len(p) - 1 for p in row_pages.values()),
            "figure_descriptions": figure_status,
            "chunker_version": CHUNKER_VERSION,
            "embed_model": dense.EMBED_MODEL,
            "indexed_at": _now(),
            "updated_at": _now(),
            "error": None,
        })
        await storage.put_index(entry)
        log.info("[ingest] %s (%s): ingested %d chunks, doc_type=%s",
                 entry["source"], doc_id, len(chunks), entry.get("doc_type"))
        return entry
    except Exception as exc:
        log.exception("[ingest] %s failed", doc_id)
        entry.update({"status": "error", "error": str(exc), "updated_at": _now()})
        await storage.put_index(entry)
        raise


def _degraded(entry: dict) -> bool:
    """Indexed without the LLM (heuristic metadata or undescribed figures)."""
    return entry.get("metadata_source") == "heuristic"         or entry.get("figure_descriptions") == "unavailable"


def schedule(doc_id: str) -> bool:
    """Queue a background ingest; False if one is already running for it.

    A document indexed while the LLM was off (GPU swapped to OCR) is redone
    once if the LLM is back by the time it finishes; the orchestrator's
    post-swap /ingest/sync covers the ones that finish earlier."""
    if doc_id in _running:
        return False
    _running.add(doc_id)

    async def _run():
        try:
            entry = await ingest(doc_id)
            if _degraded(entry) and await llm_api.healthy():
                log.info("[ingest] %s indexed without the LLM, which is back: re-indexing", doc_id)
                await ingest(doc_id)
        except Exception:
            pass                                  # already logged + recorded
        finally:
            _running.discard(doc_id)

    task = asyncio.create_task(_run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True


async def remove(doc_id: str) -> None:
    await qdrant_store.delete_document(doc_id)
    await storage.delete_index(doc_id)
    log.info("[ingest] removed %s", doc_id)


async def set_access(doc_id: str, access_level: str) -> None:
    await qdrant_store.set_access(doc_id, access_level)
    await storage.update_index(doc_id, access_level=access_level, updated_at=_now())


async def sync(upgrade_heuristic: bool = False) -> dict:
    """Bring the index in line with ocr_service's documents:
    ingest finished documents that are missing / failed / stale (other
    embedding model or chunker version), fix access levels, and drop index
    entries whose document is gone. `upgrade_heuristic` also redoes documents
    indexed while the LLM was down (heuristic metadata, undescribed figures)."""
    metas = {str(m["id"]): m for m in await storage.list_ocr_documents()}
    queued, removed, access_fixed = [], [], []

    for doc_id, meta in metas.items():
        if meta.get("status") != "ok":
            continue
        entry = storage.REGISTRY.get(doc_id)
        stale = (entry is None or entry.get("status") == "error"
                 or entry.get("embed_model") != dense.EMBED_MODEL
                 or entry.get("chunker_version") != CHUNKER_VERSION
                 or (upgrade_heuristic and (entry.get("metadata_source") == "heuristic"
                                            or entry.get("figure_descriptions") == "unavailable")))
        if stale:
            if schedule(doc_id):
                queued.append(doc_id)
            continue
        level = meta.get("access_level") or "private"
        if entry.get("access_level") != level:
            await set_access(doc_id, level)
            access_fixed.append(doc_id)

    for doc_id in list(storage.REGISTRY):
        if doc_id not in metas:
            await remove(doc_id)
            removed.append(doc_id)

    log.info("[sync] queued=%d removed=%d access_fixed=%d",
             len(queued), len(removed), len(access_fixed))
    return {"queued": queued, "removed": removed, "access_fixed": access_fixed}
