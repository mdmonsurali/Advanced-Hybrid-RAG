"""Docling ocr_service — UUID-based persistence in MinIO.

Exposes:
    GET    /health                              Docling + MinIO probe
    POST   /ocr           (file)                single-file pipeline, returns {document_id}
    POST   /ocr/batch     (files[])             SSE batch pipeline, one event per file
    POST   /ocr/batch-zip ({"ids":[uuid,...]})  ZIP bundle of N documents
    GET    /documents     ?limit&offset&status  list documents (no bytes)
    GET    /documents/{id}                      JSON metadata (no bytes)
    PATCH  /documents/{id}/access               {"access_level": "private"|"shared"} (owner only)
    GET    /documents/{id}/{kind}               stream source|md|json|docx from MinIO
    GET    /preview/{id}                        DOCX -> PDF via LibreOffice
    POST   /preview-input (file)                DOCX upload -> PDF preview
    POST   /gpu/enable | /gpu/disable           Docling GPU worker on/off (orchestrator GPU swap)

Bytes live in MinIO at documents/{uuid}/{source.ext,output.md,layout.json,output.docx}
plus picture crops under documents/{uuid}/images/. layout.json is a DoclingDocument.
Metadata lives next to the bytes in documents/{uuid}/meta.json.

Documents are private to their owner unless marked shared; shared documents
are readable (not writable) by every logged-in user. Every finished, deleted
or re-shared document is reported to rag_service so its index stays in sync.
"""

from __future__ import annotations

import asyncio
import io
import json as _json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid as _uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import docling_engine
import gpu_worker
import httpx
import storage
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from PIL import Image

log = logging.getLogger("ocr_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

PREVIEW_TIMEOUT_SEC = int(os.environ.get("PREVIEW_TIMEOUT_SEC", "360"))
SUPPORTED_EXTS = {".pdf", ".docx", ".jpg", ".jpeg", ".png"}
PREVIEW_INPUT_EXTS = {".docx", ".doc", ".odt", ".rtf"}

ARTIFACT_KINDS = {"source", "md", "json", "docx"}
ACCESS_LEVELS = {"private", "shared"}

RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://rag_service:8003")


# Identity — trusted from ui_service over the docker network. Drop the
# host port mapping for 8001 in production so this header can't be forged.


class Identity:
    __slots__ = ("id", "role")

    def __init__(self, id: _uuid.UUID, role: str):
        self.id = id
        self.role = role


def require_user(
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_user_role: str | None = Header(default=None, alias="X-User-Role"),
) -> Identity:
    if not x_user_id or not x_user_role:
        raise HTTPException(401, "Missing identity headers")
    try:
        uid = _uuid.UUID(x_user_id)
    except (ValueError, TypeError):
        raise HTTPException(401, "Invalid X-User-Id")
    if x_user_role not in {"user", "admin", "master"}:
        raise HTTPException(401, "Invalid X-User-Role")
    return Identity(id=uid, role=x_user_role)


def _assert_owner_or_404(doc: dict, user_id: _uuid.UUID) -> None:
    """Use 404 (not 403) so we don't leak existence of other users' docs."""
    if doc.get("owner_id") != user_id:
        raise HTTPException(404, "Document not found")


def _assert_readable_or_404(doc: dict, user_id: _uuid.UUID) -> None:
    """Owners can read their documents; everyone can read shared ones."""
    if doc.get("owner_id") != user_id and doc.get("access_level") != "shared":
        raise HTTPException(404, "Document not found")


# rag_service sync hooks — fire-and-forget so OCR never waits on indexing.

_rag_tasks: set[asyncio.Task] = set()


async def _call_rag(method: str, path: str, payload: dict | None = None) -> None:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method, f"{RAG_SERVICE_URL}{path}", json=payload)
        if resp.status_code >= 400:
            log.warning("[rag] %s %s -> %d: %s", method, path, resp.status_code, resp.text[:200])
    except httpx.HTTPError as e:
        log.warning("[rag] %s %s failed: %s", method, path, e)


def _notify_rag(method: str, path: str, payload: dict | None = None) -> None:
    task = asyncio.create_task(_call_rag(method, path, payload))
    _rag_tasks.add(task)
    task.add_done_callback(_rag_tasks.discard)


# Lifespan: Docling models + MinIO client + document index


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(docling_engine.init_converter)
    storage.init_minio()
    await storage.load_index()
    log.info("storage layer ready (minio)")
    yield
    await gpu_worker.disable()


app = FastAPI(title="Docling OCR Service", version="3.0", lifespan=lifespan)


class _SilencePathsFilter(logging.Filter):
    """Drop uvicorn access lines for /health so the actual events show."""

    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__()
        self._paths = paths

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f" {p} " in msg for p in self._paths)


logging.getLogger("uvicorn.access").addFilter(_SilencePathsFilter(("/health",)))


# GPU worker — called only by orchestrator_service's gpu_swap over the docker
# network (no identity: drop the host port mapping for 8001 in production).


@app.post("/gpu/enable")
async def gpu_enable():
    if not docling_engine.GPU_SWAP:
        raise HTTPException(409, "GPU_SWAP is off for ocr_service")
    try:
        return await gpu_worker.enable()
    except Exception as e:
        log.exception("[gpu] enable failed")
        raise HTTPException(500, f"GPU worker failed to start: {e}")


@app.post("/gpu/disable")
async def gpu_disable():
    return await gpu_worker.disable()


@app.get("/gpu/status")
async def gpu_status():
    return {"gpu_swap": docling_engine.GPU_SWAP, "enabled": gpu_worker.enabled()}


# Pipeline helpers


def _detect_scan_type(in_path: Path) -> str:
    """Return 'native' if file has extractable text, else 'scanned'.
    Metadata only: Docling decides per region where OCR is needed."""
    import pypdfium2 as pdfium  # local import keeps cold start tight

    ext = in_path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg"}:
        return "scanned"
    if ext == ".docx":
        return "native"
    if ext == ".pdf":
        try:
            pdf = pdfium.PdfDocument(str(in_path))
            try:
                for i in range(len(pdf)):
                    page = pdf[i]
                    text = page.get_textpage().get_text_range()
                    if len(text.strip()) >= 32:
                        return "native"
            finally:
                pdf.close()
            return "scanned"
        except Exception as e:
            log.warning("scan-type probe failed for %s: %s", in_path.name, e)
            return "scanned"
    return "scanned"


def _docx_page_count(in_path: Path) -> int:
    """Docling doesn't paginate DOCX; read the page count Word saved in
    docProps/app.xml, falling back to 1."""
    try:
        with zipfile.ZipFile(in_path) as zf:
            app_xml = zf.read("docProps/app.xml").decode("utf-8", errors="replace")
        m = re.search(r"<(?:\w+:)?Pages>(\d+)</", app_xml)
        if m and int(m.group(1)) > 0:
            return int(m.group(1))
    except (KeyError, zipfile.BadZipFile, OSError):
        pass
    return 1


async def _upload_picture_assets(
    doc_id: _uuid.UUID, pictures: list[tuple[str, Image.Image]]
) -> None:
    """PNG-encode each picture and upload it to MinIO under images/{fname}."""

    async def _go(fname: str, img: Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        await storage.put_image(doc_id, fname, buf.getvalue())

    if pictures:
        await asyncio.gather(*(_go(f, img) for f, img in pictures))


def _html_to_docx(html: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "output.html"
        src.write_text(html, encoding="utf-8")
        out = _libreoffice_convert(
            src,
            Path(tmp),
            "docx:MS Word 2007 XML",
            PREVIEW_TIMEOUT_SEC,
            infilter="HTML (StarWriter)",
        )
        return out.read_bytes()


# Core pipeline — runs against an on-disk temp file, persists to MinIO + DB


async def _run_pipeline(
    in_path: Path, source_bytes: bytes, original_name: str, owner_id: _uuid.UUID
) -> dict:
    """Process one document end-to-end:
       1. Create metadata with status=queued, PUT source to MinIO.
       2. Docling convert → markdown / DoclingDocument JSON / DOCX bytes.
       3. PUT pictures + three artifacts to MinIO; mark metadata ok.

    On failure the document is left with status=error and the original exception
    is re-raised so the caller can surface it (single mode) or convert it to
    an SSE result event (batch mode).
    """
    ext = in_path.suffix.lower()
    doc_id = await storage.insert_document(original_name, ext, source_bytes, owner_id)

    try:
        await storage.update_document(doc_id, status="processing")
        t0 = time.perf_counter()

        scan_type = _detect_scan_type(in_path)
        log.info("[ocr] %s id=%s: type=%s", original_name, doc_id, scan_type)

        doc = await docling_engine.convert(in_path)
        page_count = doc.num_pages() or (_docx_page_count(in_path) if ext == ".docx" else 1)

        # HTML first: it embeds the pictures, and reference_pictures() then
        # repoints them at images/picture-N.png for the markdown + JSON.
        html = docling_engine.export_html(doc)
        pictures = docling_engine.reference_pictures(doc)
        md_bytes = docling_engine.export_markdown(doc).encode("utf-8")
        layout_json = docling_engine.export_json(doc)
        json_bytes = _json.dumps(layout_json, indent=2, ensure_ascii=False).encode("utf-8")

        docx_bytes, _ = await asyncio.gather(
            asyncio.to_thread(_html_to_docx, html),
            _upload_picture_assets(doc_id, pictures),
        )

        md_key, json_key, docx_key = await asyncio.gather(
            storage.put_artifact(doc_id, "md", md_bytes),
            storage.put_artifact(doc_id, "json", json_bytes),
            storage.put_artifact(doc_id, "docx", docx_bytes),
        )

        elapsed = round(time.perf_counter() - t0, 2)
        await storage.update_document(
            doc_id,
            status="ok",
            page_count=page_count,
            scan_type=scan_type,
            markdown_key=md_key,
            json_key=json_key,
            docx_key=docx_key,
            elapsed_sec=elapsed,
            error=None,
        )
        _notify_rag("POST", f"/ingest/{doc_id}")

        return {
            "document_id": str(doc_id),
            "original_name": original_name,
            "status": "ok",
            "scan_type": scan_type,
            "page_count": page_count,
            "elapsed_sec": elapsed,
        }
    except Exception as exc:
        await storage.update_document(doc_id, status="error", error=str(exc))
        raise


# LibreOffice conversion helper (DOCX export + previews)

# soffice can't run two conversions on one profile, and a hung instance is
# killed before each run, so conversions are serialized.
_soffice_lock = threading.Lock()


def _libreoffice_convert(
    src: Path, outdir: Path, target: str, timeout: int, infilter: str | None = None
) -> Path:
    """Convert `src` with headless LibreOffice to `target` (e.g. "pdf" or
    "docx:MS Word 2007 XML") and return the output path."""
    cmd = ["libreoffice", "--headless"]
    if infilter:
        cmd.append(f"--infilter={infilter}")
    cmd += ["--convert-to", target, "--outdir", str(outdir), str(src)]
    with _soffice_lock:
        subprocess.run(["pkill", "-9", "-f", "soffice"], check=False, capture_output=True)
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except subprocess.CalledProcessError as exc:
            log.error("LibreOffice failed: %s", exc.stderr.decode(errors="replace"))
            raise HTTPException(500, "LibreOffice conversion failed")
        except subprocess.TimeoutExpired:
            subprocess.run(["pkill", "-9", "-f", "soffice"], check=False, capture_output=True)
            raise HTTPException(504, "LibreOffice conversion timed out")

    out_path = outdir / (src.stem + "." + target.split(":", 1)[0])
    if not out_path.exists():
        raise HTTPException(500, "Converted file not found")
    return out_path


def _convert_to_pdf(src: Path, outdir: Path, timeout: int) -> Path:
    return _libreoffice_convert(src, outdir, "pdf", timeout)


# Routes


@app.get("/health")
async def health():
    if not docling_engine.is_ready():
        raise HTTPException(503, "docling converter not initialized")
    # MinIO
    try:
        storage.health_minio()
    except Exception as e:
        raise HTTPException(503, f"minio not reachable: {e}")
    return {"status": "ok", "docling": "ok", "minio": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...), identity: Identity = Depends(require_user)):
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(415, f"Unsupported file type: {ext}")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty upload")

    work = Path(tempfile.mkdtemp(prefix="ocr_in_"))
    in_path = work / name
    in_path.write_bytes(content)

    try:
        return await _run_pipeline(
            in_path, source_bytes=content, original_name=name, owner_id=identity.id
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("pipeline failed")
        raise HTTPException(500, f"Pipeline failed: {exc}")
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)


# ── SSE batch ──────


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {_json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@app.post("/ocr/batch")
async def ocr_batch(
    files: list[UploadFile] = File(...), identity: Identity = Depends(require_user)
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    # Read every upload body up-front (the request body is gone once the
    # StreamingResponse starts).
    prepared: list[dict] = []
    for upload in files:
        original = Path(upload.filename or "upload").name
        ext = Path(original).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            prepared.append(
                {
                    "original_name": original,
                    "ext": ext,
                    "content": None,
                    "early_error": f"Unsupported file type: {ext}",
                }
            )
            continue
        content = await upload.read()
        if not content:
            prepared.append(
                {
                    "original_name": original,
                    "ext": ext,
                    "content": None,
                    "early_error": "Empty upload",
                }
            )
            continue
        prepared.append(
            {"original_name": original, "ext": ext, "content": content, "early_error": None}
        )

    async def event_stream():
        import shutil

        work_root = Path(tempfile.mkdtemp(prefix="ocr_batch_"))
        t0 = time.perf_counter()
        try:
            items = [
                {"index": i, "original_name": p["original_name"], "status": "queued"}
                for i, p in enumerate(prepared)
            ]
            yield _sse("start", {"total": len(prepared), "items": items})

            for i, p in enumerate(prepared):
                if p["early_error"]:
                    yield _sse(
                        "result",
                        {
                            "index": i,
                            "original_name": p["original_name"],
                            "status": "error",
                            "error": p["early_error"],
                            "elapsed_sec": 0,
                        },
                    )
                    continue

                sub = work_root / f"item_{i}"
                sub.mkdir(parents=True, exist_ok=True)
                in_path = sub / (p["original_name"] or f"item_{i}{p['ext']}")
                in_path.write_bytes(p["content"])

                try:
                    row = await _run_pipeline(
                        in_path,
                        source_bytes=p["content"],
                        original_name=p["original_name"],
                        owner_id=identity.id,
                    )
                    row["index"] = i
                    yield _sse("result", row)
                except HTTPException as exc:
                    yield _sse(
                        "result",
                        {
                            "index": i,
                            "original_name": p["original_name"],
                            "status": "error",
                            "error": exc.detail,
                            "elapsed_sec": 0,
                        },
                    )
                except Exception as exc:
                    log.exception("[batch] pipeline failed for %s", p["original_name"])
                    yield _sse(
                        "result",
                        {
                            "index": i,
                            "original_name": p["original_name"],
                            "status": "error",
                            "error": f"Pipeline failed: {exc}",
                            "elapsed_sec": 0,
                        },
                    )

            yield _sse(
                "end",
                {
                    "total": len(prepared),
                    "total_elapsed_sec": round(time.perf_counter() - t0, 2),
                },
            )
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Bulk download ──────


@app.post("/ocr/batch-zip")
async def ocr_batch_zip(payload: dict = Body(...), identity: Identity = Depends(require_user)):
    ids = payload.get("ids") or []
    if not isinstance(ids, list) or not ids:
        raise HTTPException(400, 'Body must be {"ids": [...]}')

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for raw in ids:
            try:
                doc_id = _uuid.UUID(str(raw))
            except (ValueError, TypeError):
                continue
            doc = await storage.fetch_document(doc_id)
            if not doc or doc.get("status") != "ok":
                continue
            if doc.get("owner_id") != identity.id:
                continue
            stem = Path(doc["original_name"]).stem or str(doc_id)
            for kind in ("md", "json", "docx"):
                try:
                    filename, _ctype, body = await storage.get_artifact_bytes(doc_id, kind)
                except (KeyError, FileNotFoundError):
                    continue
                zf.writestr(f"{stem}/ocr_{filename}", body)

    data = buf.getvalue()
    if len(data) < 30:
        raise HTTPException(404, "No output files found for the given ids")

    return StreamingResponse(
        iter([data]),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="OCR_results.zip"'},
    )


# ── Document API ──────


@app.get("/documents")
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    identity: Identity = Depends(require_user),
):
    rows = await storage.list_documents(
        limit=limit, offset=offset, status=status, owner_id=identity.id
    )
    total = await storage.count_documents(owner_id=identity.id, status=status)
    return JSONResponse(
        content=_json.loads(_json.dumps(rows, default=str)),
        headers={
            "X-Total-Count": str(total),
            "Access-Control-Expose-Headers": "X-Total-Count",
        },
    )


def _parse_uuid(raw: str) -> _uuid.UUID:
    try:
        return _uuid.UUID(raw)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid document id (must be UUID)")


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str, identity: Identity = Depends(require_user)):
    uid = _parse_uuid(doc_id)
    doc = await storage.fetch_document(uid)
    if doc is None:
        raise HTTPException(404, "Document not found")
    _assert_readable_or_404(doc, identity.id)
    return JSONResponse(content=_json.loads(_json.dumps(doc, default=str)))


@app.patch("/documents/{doc_id}/access")
async def set_document_access(
    doc_id: str, payload: dict = Body(...), identity: Identity = Depends(require_user)
):
    level = payload.get("access_level")
    if level not in ACCESS_LEVELS:
        raise HTTPException(400, f"access_level must be one of {sorted(ACCESS_LEVELS)}")
    uid = _parse_uuid(doc_id)
    doc = await storage.fetch_document(uid)
    if doc is None:
        raise HTTPException(404, "Document not found")
    _assert_owner_or_404(doc, identity.id)
    await storage.update_document(uid, access_level=level)
    _notify_rag("PATCH", f"/ingest/{uid}/access", {"access_level": level})
    return {"document_id": str(uid), "access_level": level}


@app.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, identity: Identity = Depends(require_user)):
    uid = _parse_uuid(doc_id)
    ok = await storage.delete_document(uid, identity.id)
    if not ok:
        # Either missing or not the caller's. Use 404 to avoid leaking existence.
        raise HTTPException(404, "Document not found")
    _notify_rag("DELETE", f"/ingest/{uid}")
    return Response(status_code=204)


@app.delete("/internal/owner/{owner_id}/documents")
async def delete_owner_documents(owner_id: str, identity: Identity = Depends(require_user)):
    """Internal: called by auth_service when a user account is deleted, so the
    user's OCR history is wiped from DB + MinIO. Caller must be that same user
    or an admin (the caller already proved identity via X-User-Id headers
    forwarded by auth_service)."""
    target = _parse_uuid(owner_id)
    if identity.role not in {"admin", "master"} and identity.id != target:
        raise HTTPException(403, "Forbidden")
    removed = await storage.delete_all_documents_for_owner(target)
    for doc_id in removed:
        _notify_rag("DELETE", f"/ingest/{doc_id}")
    return {"deleted": len(removed)}


@app.get("/documents/{doc_id}/images/{filename}")
async def get_document_image(
    doc_id: str, filename: str, identity: Identity = Depends(require_user)
):
    uid = _parse_uuid(doc_id)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    doc = await storage.fetch_document(uid)
    if doc is None:
        raise HTTPException(404, "Image not found")
    _assert_readable_or_404(doc, identity.id)
    try:
        fname, content_type, body = await storage.get_image_bytes(uid, filename)
    except Exception:
        raise HTTPException(404, "Image not found")
    return Response(
        content=body,
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="{fname}"'},
    )


@app.get("/documents/{doc_id}/{kind}")
async def get_document_artifact(doc_id: str, kind: str, identity: Identity = Depends(require_user)):
    if kind not in ARTIFACT_KINDS:
        raise HTTPException(400, f"Unknown kind: {kind}")
    uid = _parse_uuid(doc_id)
    doc = await storage.fetch_document(uid)
    if doc is None:
        raise HTTPException(404, "Document not found")
    _assert_readable_or_404(doc, identity.id)
    try:
        filename, content_type, body = await storage.get_artifact_bytes(uid, kind)
    except KeyError:
        raise HTTPException(404, "Document not found")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    # JSON/MD viewed inline so the UI can pretty-print without download dance.
    disposition = "inline" if kind in {"md", "json"} else f'attachment; filename="ocr_{filename}"'
    return Response(
        content=body,
        media_type=content_type,
        headers={"Content-Disposition": disposition},
    )


# ── DOCX → PDF preview ──────


@app.get("/preview/{doc_id}")
async def preview_docx(doc_id: str, identity: Identity = Depends(require_user)):
    uid = _parse_uuid(doc_id)
    doc = await storage.fetch_document(uid)
    if doc is None:
        raise HTTPException(404, "Document not found")
    _assert_readable_or_404(doc, identity.id)
    try:
        _filename, _ctype, docx_bytes = await storage.get_artifact_bytes(uid, "docx")
    except KeyError:
        raise HTTPException(404, "Document not found")
    except FileNotFoundError:
        raise HTTPException(404, "DOCX not available for this document")

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"{uid}.docx"
        src.write_bytes(docx_bytes)
        pdf_path = _convert_to_pdf(src, Path(tmp), PREVIEW_TIMEOUT_SEC)
        pdf_bytes = pdf_path.read_bytes()

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{uid}.pdf"'},
    )


@app.post("/preview-input")
async def preview_input(file: UploadFile = File(...), identity: Identity = Depends(require_user)):
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in PREVIEW_INPUT_EXTS:
        raise HTTPException(415, f"Cannot preview {ext}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / name
        src.write_bytes(data)
        pdf_path = _convert_to_pdf(src, Path(tmp), timeout=120)
        pdf_bytes = pdf_path.read_bytes()
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{Path(name).stem}.pdf"'},
    )
