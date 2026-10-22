"""Storage layer for OCR — everything lives in MinIO.

Owns one MinIO client (MINIO_*). Each document's metadata is a JSON object
next to its bytes; at startup every meta.json is loaded into an in-memory
index, which serves the list / count / lookup calls. ocr_service is the only
writer, so the index stays in sync by updating it on every write.

Public surface used by main.py:
    init_minio() / load_index()
    health_minio()
    insert_document(original_name, ext, source_bytes, owner_id) -> uuid
    update_document(doc_id, **fields) -> None
    put_artifact(doc_id, kind, body) -> str          # kind in {md, json, docx}
    fetch_document(doc_id) -> dict | None
    list_documents(limit, offset, status=None, owner_id=None) -> list[dict]
    count_documents(owner_id=None, status=None) -> int
    get_artifact_bytes(doc_id, kind) -> (filename, content_type, bytes)

Paths inside the MinIO bucket:
    documents/{uuid}/meta.json
    documents/{uuid}/source.{ext}
    documents/{uuid}/output.md
    documents/{uuid}/layout.json
    documents/{uuid}/output.docx
    documents/{uuid}/images/picture-N.png
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Optional

from minio import Minio
from minio.deleteobjects import DeleteObject
from minio.error import S3Error

log = logging.getLogger("ocr_service.storage")

# ── Config from env ───────────────────────────────────────────────────────

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "changeme")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ocr")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

UPLOAD_CONCURRENCY = int(os.getenv("MINIO_UPLOAD_CONCURRENCY", "8"))

# ── Module state ────────

minio: Optional[Minio] = None
_upload_sem: Optional[asyncio.Semaphore] = None
# doc_id -> metadata dict (owner_id as UUID, timestamps as ISO strings).
_index: dict[_uuid.UUID, dict] = {}
_meta_lock = asyncio.Lock()


# ── Lifecycle ────


def init_minio() -> None:
    global minio, _upload_sem
    minio = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
        region=MINIO_REGION,
    )
    _upload_sem = asyncio.Semaphore(UPLOAD_CONCURRENCY)
    # Bucket is created by minio-init compose service; verify it exists.
    if not minio.bucket_exists(MINIO_BUCKET):
        log.warning("minio bucket %s not found, creating", MINIO_BUCKET)
        minio.make_bucket(MINIO_BUCKET)
    log.info("minio ready: bucket=%s endpoint=%s", MINIO_BUCKET, MINIO_ENDPOINT)


def _decode_meta(raw: bytes) -> dict:
    meta = _json.loads(raw)
    meta["id"] = _uuid.UUID(meta["id"])
    if meta.get("owner_id"):
        meta["owner_id"] = _uuid.UUID(meta["owner_id"])
    return meta


async def load_index() -> None:
    """Read every documents/*/meta.json into the in-memory index."""

    def _load() -> list[dict]:
        assert minio is not None
        metas = []
        for obj in minio.list_objects(MINIO_BUCKET, prefix="documents/", recursive=True):
            if obj.object_name.endswith("/meta.json"):
                try:
                    metas.append(_decode_meta(_get_object_bytes(obj.object_name)))
                except (S3Error, ValueError, KeyError) as e:
                    log.warning("skipping unreadable %s: %s", obj.object_name, e)
        return metas

    _index.clear()
    for meta in await asyncio.to_thread(_load):
        _index[meta["id"]] = meta
    log.info("document index loaded: %d documents", len(_index))


# ── Health probes ────


def health_minio() -> None:
    if minio is None:
        raise RuntimeError("minio client not initialised")
    # bucket_exists is cheap (HEAD request) and confirms creds + reachability.
    minio.bucket_exists(MINIO_BUCKET)


# ── Helpers ───

ARTIFACT_KEY = {
    "md": "output.md",
    "json": "layout.json",
    "docx": "output.docx",
}
ARTIFACT_CONTENT_TYPE = {
    "md": "text/markdown",
    "json": "application/json",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
SOURCE_CONTENT_TYPE = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _source_object_name(doc_id: _uuid.UUID, ext: str) -> str:
    return f"documents/{doc_id}/source{ext}"


def _artifact_object_name(doc_id: _uuid.UUID, kind: str) -> str:
    return f"documents/{doc_id}/{ARTIFACT_KEY[kind]}"


async def _put_object(name: str, body: bytes, content_type: str) -> None:
    """MinIO PUT in a thread, throttled by the upload semaphore."""
    assert _upload_sem is not None and minio is not None
    async with _upload_sem:
        await asyncio.to_thread(
            minio.put_object,
            MINIO_BUCKET,
            name,
            data=_BytesReader(body),
            length=len(body),
            content_type=content_type,
        )


def _get_object_bytes(name: str) -> bytes:
    """Sync MinIO GET — used inside asyncio.to_thread."""
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


# ── CRUD ─────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _meta_object_name(doc_id: _uuid.UUID) -> str:
    return f"documents/{doc_id}/meta.json"


async def _write_meta(meta: dict) -> None:
    body = _json.dumps(meta, default=str, ensure_ascii=False).encode("utf-8")
    await _put_object(_meta_object_name(meta["id"]), body, "application/json")


async def insert_document(
    original_name: str, ext: str, source_bytes: bytes, owner_id: _uuid.UUID
) -> _uuid.UUID:
    """Create a 'queued' document and upload its source bytes to MinIO.

    Returns the new document UUID. Caller drives state transitions via
    update_document() and put_artifact().
    """
    doc_id = _uuid.uuid4()
    source_key = _source_object_name(doc_id, ext)
    await _put_object(
        source_key,
        source_bytes,
        SOURCE_CONTENT_TYPE.get(ext.lower(), "application/octet-stream"),
    )
    now = _now()
    meta = {
        "id": doc_id,
        "original_name": original_name,
        "status": "queued",
        "scan_type": None,
        "page_count": None,
        "source_key": source_key,
        "markdown_key": None,
        "json_key": None,
        "docx_key": None,
        "elapsed_sec": None,
        "error": None,
        "owner_id": owner_id,
        "access_level": "private",
        "created_at": now,
        "updated_at": now,
    }
    async with _meta_lock:
        _index[doc_id] = meta
        await _write_meta(meta)
    return doc_id


_ALLOWED_UPDATE_COLS = {
    "status",
    "scan_type",
    "page_count",
    "markdown_key",
    "json_key",
    "docx_key",
    "elapsed_sec",
    "error",
    "access_level",
}


async def update_document(doc_id: _uuid.UUID, **fields: Any) -> None:
    """Update an existing document's metadata. Caller specifies fields by name."""
    cols = {k: v for k, v in fields.items() if k in _ALLOWED_UPDATE_COLS}
    if not cols:
        return
    async with _meta_lock:
        meta = _index.get(doc_id)
        if meta is None:
            return
        meta.update(cols)
        meta["updated_at"] = _now()
        await _write_meta(meta)


async def put_artifact(doc_id: _uuid.UUID, kind: str, body: bytes) -> str:
    """Upload a generated artifact (md / json / docx) to MinIO. Returns the key."""
    if kind not in ARTIFACT_KEY:
        raise ValueError(f"unknown artifact kind: {kind}")
    key = _artifact_object_name(doc_id, kind)
    await _put_object(key, body, ARTIFACT_CONTENT_TYPE[kind])
    return key


IMAGE_CONTENT_TYPE = "image/png"


def _image_object_name(doc_id: _uuid.UUID, filename: str) -> str:
    safe = os.path.basename(filename)
    return f"documents/{doc_id}/images/{safe}"


async def put_image(doc_id: _uuid.UUID, filename: str, body: bytes) -> str:
    """Upload a cropped Picture PNG to MinIO. Returns the object key."""
    key = _image_object_name(doc_id, filename)
    await _put_object(key, body, IMAGE_CONTENT_TYPE)
    return key


async def get_image_bytes(doc_id: _uuid.UUID, filename: str) -> tuple[str, str, bytes]:
    """Fetch a stored Picture PNG. Returns (filename, content_type, body)."""
    key = _image_object_name(doc_id, filename)
    body = await asyncio.to_thread(_get_object_bytes, key)
    return os.path.basename(filename), IMAGE_CONTENT_TYPE, body


async def fetch_document(doc_id: _uuid.UUID) -> Optional[dict]:
    meta = _index.get(doc_id)
    return dict(meta) if meta else None


def _filtered(owner_id: Optional[_uuid.UUID], status: Optional[str]) -> list[dict]:
    return [
        m
        for m in _index.values()
        if (owner_id is None or m.get("owner_id") == owner_id)
        and (not status or m.get("status") == status)
    ]


async def count_documents(
    owner_id: Optional[_uuid.UUID] = None, status: Optional[str] = None
) -> int:
    """Total documents matching the same filters list_documents uses."""
    return len(_filtered(owner_id, status))


def _delete_minio_prefix(prefix: str) -> int:
    """Synchronous MinIO recursive prefix delete. Returns object count removed.
    Called inside asyncio.to_thread by the async wrappers below."""
    assert minio is not None
    keys = [
        DeleteObject(obj.object_name)
        for obj in minio.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)
    ]
    if not keys:
        return 0
    errors = list(minio.remove_objects(MINIO_BUCKET, keys))
    for err in errors:
        log.warning("minio delete error: %s", err)
    return len(keys) - len(errors)


async def _drop_document(doc_id: _uuid.UUID) -> None:
    _index.pop(doc_id, None)
    try:
        removed = await asyncio.to_thread(_delete_minio_prefix, f"documents/{doc_id}/")
        log.info("deleted document %s (%d MinIO objects)", doc_id, removed)
    except Exception as e:
        # Gone from the index; orphan MinIO bytes are recoverable, don't fail the call.
        log.warning("MinIO cleanup failed for %s: %s", doc_id, e)


async def delete_document(doc_id: _uuid.UUID, owner_id: _uuid.UUID) -> bool:
    """Delete one document (all MinIO objects under documents/{id}/).
    Owner-scoped: returns False if it doesn't exist or belongs to someone
    else, True if it was deleted."""
    async with _meta_lock:
        meta = _index.get(doc_id)
        if meta is None or meta.get("owner_id") != owner_id:
            return False
        await _drop_document(doc_id)
    return True


async def delete_all_documents_for_owner(owner_id: _uuid.UUID) -> list[_uuid.UUID]:
    """Delete every document owned by `owner_id`. Returns the removed ids.
    Used when a user account is deleted."""
    async with _meta_lock:
        ids = [m["id"] for m in _filtered(owner_id, None)]
        for doc_id in ids:
            await _drop_document(doc_id)
    log.info("deleted %d documents for owner %s", len(ids), owner_id)
    return ids


_LIST_FIELDS = (
    "id",
    "original_name",
    "status",
    "scan_type",
    "page_count",
    "elapsed_sec",
    "error",
    "created_at",
    "updated_at",
    "owner_id",
    "access_level",
)


async def list_documents(
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    owner_id: Optional[_uuid.UUID] = None,
) -> list[dict]:
    rows = sorted(_filtered(owner_id, status), key=lambda m: m["created_at"], reverse=True)
    out = [{k: m.get(k) for k in _LIST_FIELDS} for m in rows[offset : offset + limit]]
    for row in out:
        row["access_level"] = row["access_level"] or "private"
    return out


async def get_artifact_bytes(doc_id: _uuid.UUID, kind: str) -> tuple[str, str, bytes]:
    """Resolve a document's artifact and return (suggested_filename, content_type, bytes).

    `kind` is one of {source, md, json, docx}. Raises FileNotFoundError if the
    artifact hasn't been written yet, or KeyError if the document is unknown.
    """
    doc = await fetch_document(doc_id)
    if doc is None:
        raise KeyError(f"document not found: {doc_id}")

    if kind == "source":
        key = doc["source_key"]
        # filename derived from source_key extension
        ext = os.path.splitext(key)[1] or ".bin"
        content_type = SOURCE_CONTENT_TYPE.get(ext.lower(), "application/octet-stream")
        # User-facing name: original + ext
        filename = doc["original_name"]
    else:
        col = {"md": "markdown_key", "json": "json_key", "docx": "docx_key"}[kind]
        key = doc[col]
        if not key:
            raise FileNotFoundError(f"document {doc_id} has no {kind} artifact yet")
        content_type = ARTIFACT_CONTENT_TYPE[kind]
        stem = os.path.splitext(doc["original_name"])[0] or str(doc_id)
        suffix = {"md": ".md", "json": ".json", "docx": ".docx"}[kind]
        filename = f"{stem}{suffix}"

    body = await asyncio.to_thread(_get_object_bytes, key)
    return filename, content_type, body
