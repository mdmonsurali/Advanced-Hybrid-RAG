"""Google Drive API v3 wrapper (read-only) for the gdrive MCP server.

GDRIVE_AUTH:
    oauth            $GDRIVE_SECRETS_DIR/token.json, created once on the host
                     with `python gdrive_auth.py` (your own Drive)
    service_account  $GDRIVE_SECRETS_DIR/service_account.json (only files
                     shared with the service account are visible)
"""

from __future__ import annotations

import io
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("rag_service.gdrive")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
GDRIVE_AUTH = os.getenv("GDRIVE_AUTH", "oauth").strip().lower()
GDRIVE_SECRETS_DIR = Path(os.getenv("GDRIVE_SECRETS_DIR", "/secrets/gdrive"))

_EXPORTS = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}
_TEXT_MIMES = ("text/", "application/json", "application/xml")
_FIELDS = "id, name, mimeType, modifiedTime, webViewLink, size, owners(displayName)"

_service = None
_lock = threading.Lock()


def is_configured() -> bool:
    name = "token.json" if GDRIVE_AUTH == "oauth" else "service_account.json"
    return (GDRIVE_SECRETS_DIR / name).exists()


def _credentials():
    if GDRIVE_AUTH == "service_account":
        from google.oauth2 import service_account

        return service_account.Credentials.from_service_account_file(
            str(GDRIVE_SECRETS_DIR / "service_account.json"), scopes=SCOPES
        )

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_path = GDRIVE_SECRETS_DIR / "token.json"
    if not token_path.exists():
        raise RuntimeError(f"{token_path} not found — run gdrive_auth.py on the host first")
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())  # refreshed in memory; secrets mount is read-only
        else:
            raise RuntimeError("Google Drive token is invalid — re-run gdrive_auth.py")
    return creds


def _drive():
    global _service
    with _lock:
        if _service is None:
            from googleapiclient.discovery import build

            _service = build("drive", "v3", credentials=_credentials(), cache_discovery=False)
        return _service


def _escape(term: str) -> str:
    return term.replace("\\", "\\\\").replace("'", "\\'")


def search_files(query: str, mime_types: list[str] | None = None, limit: int = 10) -> list[dict]:
    clauses = [f"fullText contains '{_escape(query)}'", "trashed = false"]
    if mime_types:
        clauses.append("(" + " or ".join(f"mimeType = '{_escape(m)}'" for m in mime_types) + ")")
    res = (
        _drive()
        .files()
        .list(
            q=" and ".join(clauses),
            pageSize=limit,
            fields=f"files({_FIELDS})",
            orderBy="modifiedTime desc",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    return res.get("files", [])


def get_file_metadata(file_id: str) -> dict:
    return _drive().files().get(fileId=file_id, fields=_FIELDS, supportsAllDrives=True).execute()


def _download(request) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload

    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _status, done = downloader.next_chunk()
    return buf.getvalue()


def read_file(file_id: str, max_chars: int = 20000) -> dict:
    meta = get_file_metadata(file_id)
    mime = meta.get("mimeType", "")
    files = _drive().files()
    text = ""
    if mime in _EXPORTS:
        text = _download(files.export_media(fileId=file_id, mimeType=_EXPORTS[mime])).decode(
            "utf-8", errors="replace"
        )
    elif mime == "application/pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(_download(files.get_media(fileId=file_id))))
        text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
    elif mime.startswith(_TEXT_MIMES):
        text = _download(files.get_media(fileId=file_id)).decode("utf-8", errors="replace")
    return {
        **meta,
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
        "readable": bool(text),
    }
