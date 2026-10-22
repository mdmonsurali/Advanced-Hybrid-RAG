"""STEP 2 — metadata on every chunk.

Without metadata, filtering, freshness, permissions and citations break
down, so every chunk carries: source, page range, timestamp, topic,
document type, access level, owner, language, headings and chunk type.

Document-level fields (doc_type / topic / summary / language) come from one
llm_service call per document. If the LLM is unreachable (e.g. still loading)
a heuristic fallback is used and the entry is marked so /ingest/sync can
upgrade it later.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

import llm_api
from chunking_techniques import RawChunk

log = logging.getLogger("rag_service.metadata")

# How much of the document the metadata call sees.
META_SAMPLE_CHARS = int(os.getenv("RAG_META_SAMPLE_CHARS", "4000"))

_EXT_TYPES = {
    ".pdf": "pdf",
    ".docx": "document",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
}


def _sample(chunks: list[RawChunk]) -> str:
    out, total = [], 0
    for c in chunks:
        if total >= META_SAMPLE_CHARS:
            break
        piece = c.text[: META_SAMPLE_CHARS - total]
        out.append(piece)
        total += len(piece)
    return "\n\n".join(out)


def _heuristic(meta: dict, chunks: list[RawChunk]) -> dict:
    name = meta.get("original_name") or ""
    topic = next((c.headings[-1] for c in chunks if c.headings), Path(name).stem)
    summary = (chunks[0].text[:300] + "…") if chunks else ""
    return {
        "doc_type": _EXT_TYPES.get(Path(name).suffix.lower(), "other"),
        "topic": topic[:120],
        "summary": summary,
        "language": "und",
        "metadata_source": "heuristic",
    }


async def document_metadata(meta: dict, chunks: list[RawChunk]) -> dict:
    name = meta.get("original_name") or meta.get("id", "")
    if not chunks:
        return _heuristic(meta, chunks)
    try:
        out = await llm_api.doc_metadata(name, _sample(chunks))
        out["metadata_source"] = "llm"
        return out
    except httpx.HTTPError as e:
        log.warning("[meta] llm_service unavailable for %s, using heuristic: %s", name, e)
        return _heuristic(meta, chunks)


def chunk_payloads(meta: dict, doc_meta: dict, chunks: list[RawChunk]) -> list[dict]:
    doc_id = str(meta["id"])
    base = {
        "doc_id": doc_id,
        "source": meta.get("original_name") or doc_id,
        "owner_id": str(meta.get("owner_id") or ""),
        "access_level": meta.get("access_level") or "private",
        "timestamp": meta.get("created_at"),
        "doc_type": doc_meta.get("doc_type", "other"),
        "topic": doc_meta.get("topic", ""),
        "language": doc_meta.get("language", "und"),
    }
    return [
        {
            **base,
            "chunk_index": c.index,
            "chunk_type": c.chunk_type,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "headings": c.headings,
            "text": c.text,
        }
        for c in chunks
    ]
