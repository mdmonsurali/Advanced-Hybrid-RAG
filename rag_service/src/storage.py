"""MinIO access for rag_service.

Reads what ocr_service writes (documents/{uuid}/meta.json + layout.json) and
owns the per-document index registry under rag/{uuid}/index.json. The
registry is loaded into memory at startup and serves the catalog route, so
rag_service (single replica) stays in sync by updating it on every write.

Public surface:
    init_minio() / load_registry()
    health_minio()
    load_meta(doc_id) / load_layout(doc_id) / list_ocr_documents()
    load_picture(doc_id, rel_path)
    REGISTRY (doc_id -> entry), put_index(entry), delete_index(doc_id)
    load_figure_cache(doc_id) / put_figure_cache(doc_id, cache)  rag/{uuid}/figures.json
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os

from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error

log = logging.getLogger("rag_service.storage")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "changeme")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ocr")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

minio: Minio | None = None
# doc_id (str) -> registry entry (doc metadata + index status)
REGISTRY: dict[str, dict] = {}
_registry_lock = asyncio.Lock()


def init_minio() -> None:
    global minio
    minio = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
        region=MINIO_REGION,
    )
    if not minio.bucket_exists(MINIO_BUCKET):
        log.warning("minio bucket %s not found, creating", MINIO_BUCKET)
        minio.make_bucket(MINIO_BUCKET)
    log.info("minio ready: bucket=%s endpoint=%s", MINIO_BUCKET, MINIO_ENDPOINT)


def health_minio() -> None:
    if minio is None:
        raise RuntimeError("minio client not initialised")
    minio.bucket_exists(MINIO_BUCKET)


# ── Low-level helpers ─────────────────────────────────────────────────────


def _get_object_bytes(name: str) -> bytes:
    assert minio is not None
    resp = minio.get_object(MINIO_BUCKET, name)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


class _BytesReader:
    """Minimal file-like wrapper for bytes that the minio SDK accepts."""

    __slots__ = ("_b", "_pos")

    def __init__(self, b: bytes):
        self._b = b
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self._pos + n > len(self._b):
            chunk = self._b[self._pos :]
            self._pos = len(self._b)
            return chunk
        chunk = self._b[self._pos : self._pos + n]
        self._pos += n
        return chunk


def _put_json(name: str, obj: dict) -> None:
    assert minio is not None
    body = _json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8")
    minio.put_object(
        MINIO_BUCKET,
        name,
        data=_BytesReader(body),
        length=len(body),
        content_type="application/json",
    )


async def _read_json(name: str) -> dict | None:
    try:
        return _json.loads(await asyncio.to_thread(_get_object_bytes, name))
    except S3Error as e:
        if e.code in {"NoSuchKey", "NoSuchObject"}:
            return None
        raise


# ── ocr_service documents (read-only) ─────────────────────────────────────


async def load_meta(doc_id: str) -> dict | None:
    return await _read_json(f"documents/{doc_id}/meta.json")


async def load_layout(doc_id: str) -> dict | None:
    return await _read_json(f"documents/{doc_id}/layout.json")


async def load_picture(doc_id: str, rel_path: str) -> bytes | None:
    """A picture crop ocr_service stored (rel_path: images/picture-N.png)."""
    if not rel_path.startswith("images/") or ".." in rel_path:
        return None
    try:
        return await asyncio.to_thread(_get_object_bytes, f"documents/{doc_id}/{rel_path}")
    except S3Error as e:
        if e.code in {"NoSuchKey", "NoSuchObject"}:
            return None
        raise


async def load_figure_cache(doc_id: str) -> dict:
    return await _read_json(f"rag/{doc_id}/figures.json") or {}


async def put_figure_cache(doc_id: str, cache: dict) -> None:
    await asyncio.to_thread(_put_json, f"rag/{doc_id}/figures.json", cache)


async def list_ocr_documents() -> list[dict]:
    """Every documents/*/meta.json (used by /ingest/sync)."""

    def _list() -> list[dict]:
        assert minio is not None
        metas = []
        for obj in minio.list_objects(MINIO_BUCKET, prefix="documents/", recursive=True):
            if obj.object_name.endswith("/meta.json"):
                try:
                    metas.append(_json.loads(_get_object_bytes(obj.object_name)))
                except (S3Error, ValueError) as e:
                    log.warning("skipping unreadable %s: %s", obj.object_name, e)
        return metas

    return await asyncio.to_thread(_list)


# ── Index registry (rag/{doc_id}/index.json) ──────────────────────────────


async def load_registry() -> None:
    def _load() -> list[dict]:
        assert minio is not None
        entries = []
        for obj in minio.list_objects(MINIO_BUCKET, prefix="rag/", recursive=True):
            if obj.object_name.endswith("/index.json"):
                try:
                    entries.append(_json.loads(_get_object_bytes(obj.object_name)))
                except (S3Error, ValueError) as e:
                    log.warning("skipping unreadable %s: %s", obj.object_name, e)
        return entries

    REGISTRY.clear()
    for entry in await asyncio.to_thread(_load):
        REGISTRY[entry["doc_id"]] = entry
    log.info("index registry loaded: %d documents", len(REGISTRY))


async def put_index(entry: dict) -> None:
    async with _registry_lock:
        REGISTRY[entry["doc_id"]] = entry
        await asyncio.to_thread(_put_json, f"rag/{entry['doc_id']}/index.json", entry)


async def update_index(doc_id: str, **fields) -> dict | None:
    async with _registry_lock:
        entry = REGISTRY.get(doc_id)
        if entry is None:
            return None
        entry.update(fields)
        await asyncio.to_thread(_put_json, f"rag/{doc_id}/index.json", entry)
        return entry


async def delete_index(doc_id: str) -> None:
    def _delete() -> None:
        assert minio is not None
        keys = [
            DeleteObject(o.object_name)
            for o in minio.list_objects(MINIO_BUCKET, prefix=f"rag/{doc_id}/", recursive=True)
        ]
        for err in minio.remove_objects(MINIO_BUCKET, keys):
            log.warning("minio delete error: %s", err)

    async with _registry_lock:
        REGISTRY.pop(doc_id, None)
        await asyncio.to_thread(_delete)
