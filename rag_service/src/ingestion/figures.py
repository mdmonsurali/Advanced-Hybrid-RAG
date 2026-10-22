"""STEP 1 — describe figures so their content can be retrieved.

For every picture in the document large enough to carry information, the
crop ocr_service stored in MinIO is sent to llm_service (Qwen3.5 is a
vision-language model) with its caption and surrounding paragraph, and the
returned description goes into the figure chunk (chunking_techniques/figures.py).

Descriptions are cached in rag/{doc_id}/figures.json, keyed by the crop's
content hash, so re-indexing a document doesn't describe its figures again.
If llm_service can't describe images (still loading, or a text-only model)
the remaining figures are indexed with their caption only and the entry is
marked so `/ingest/sync?upgrade_heuristic=true` can redo it later.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
from pathlib import PurePath

import httpx
import llm_api
import storage
from chunking_techniques.figures import picture_description
from chunking_techniques.tables import preceding_paragraph
from docling_core.types.doc import DoclingDocument, PictureItem
from PIL import Image  # installed with sentence-transformers

log = logging.getLogger("rag_service.figures")


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


FIGURE_DESCRIPTIONS = _env_bool("RAG_FIGURE_DESCRIPTIONS", "true")
# Crops smaller than this on either side (pixels, at DOCLING_IMAGES_SCALE) are
# logos / icons and are skipped.
FIGURE_MIN_PX = int(os.getenv("RAG_FIGURE_MIN_PX", "150"))
# Longest side sent to the model; bigger crops are downscaled (fewer image tokens).
FIGURE_MAX_PX = int(os.getenv("RAG_FIGURE_MAX_PX", "1024"))
FIGURE_MAX_PER_DOC = int(os.getenv("RAG_FIGURE_MAX_PER_DOC", "40"))


def _candidates(doc: DoclingDocument) -> list[PictureItem]:
    out = []
    for item, _level in doc.iterate_items(with_groups=False):
        if not isinstance(item, PictureItem) or item.image is None or picture_description(item):
            continue
        size = item.image.size
        if size.width < FIGURE_MIN_PX or size.height < FIGURE_MIN_PX:
            continue
        out.append(item)
    return out[:FIGURE_MAX_PER_DOC]


def _crop_path(pic: PictureItem) -> str:
    uri = pic.image.uri
    return uri.as_posix() if isinstance(uri, PurePath) else str(uri)


def _encode(png: bytes) -> tuple[str, str]:
    img = Image.open(io.BytesIO(png))
    if max(img.size) > FIGURE_MAX_PX:
        img.thumbnail((FIGURE_MAX_PX, FIGURE_MAX_PX))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


async def describe(doc_id: str, doc: DoclingDocument) -> tuple[dict[str, str], str]:
    """({picture self_ref: description}, status) where status is
    llm | cached | none (nothing to describe) | off | unavailable."""
    if not FIGURE_DESCRIPTIONS:
        return {}, "off"
    pictures = _candidates(doc)
    if not pictures:
        return {}, "none"

    cache = await storage.load_figure_cache(doc_id)
    out: dict[str, str] = {}
    status, called = "cached", 0
    for pic in pictures:
        png = await storage.load_picture(doc_id, _crop_path(pic))
        if png is None:
            continue
        key = hashlib.sha256(png).hexdigest()[:24]
        if key in cache:
            out[pic.self_ref] = cache[key]
            continue
        if status == "unavailable":
            continue
        try:
            b64, mime = _encode(png)
        except OSError as e:
            log.warning("[figures] %s: unreadable crop %s: %s", doc_id, pic.image.uri, e)
            continue
        try:
            resp = await llm_api.describe_figure(
                b64, mime, pic.caption_text(doc).strip(), preceding_paragraph(doc, pic) or ""
            )
        except httpx.HTTPError as e:
            log.warning(
                "[figures] %s: describing figures unavailable, indexing captions only: %s",
                doc_id,
                e,
            )
            status = "unavailable"
            continue
        description = (resp.get("description") or "").strip()
        cache[key] = description
        out[pic.self_ref] = description
        called += 1
        if status == "cached":
            status = "llm"

    if called:
        await storage.put_figure_cache(doc_id, cache)
    log.info(
        "[figures] %s: %d/%d figures described (%d new, status=%s)",
        doc_id,
        sum(1 for d in out.values() if d),
        len(pictures),
        called,
        status,
    )
    return out, status
