#!/usr/bin/env python3
"""
main.py — FastAPI UI service entry point.

Serves the HTML frontend, gates every route behind a session cookie, and
proxies authenticated requests to ocr_service (documents) and
orchestrator_service (RAG chat) with X-User-Id / X-User-Role headers so the
backends can scope every document and answer to its user.

    GET  /login                      login page
    POST /login                      establish session
    POST /logout                     drop session
    GET  /change-password            forced when must_change_password
    POST /change-password            update password
    GET  /me                         current user (for the UI chip)
    GET  /admin                      admin console (admin only)
    GET  /admin/users                list users (admin only)
    POST /admin/users                create user (admin only)
    PATCH  /admin/users/{id}         rename / reset password / role / disable
    DELETE /admin/users/{id}         delete

    POST /ocr                        single-file upload
    POST /ocr/batch                  SSE multi-file upload
    POST /ocr/batch-zip              ZIP bundle download
    GET  /documents                  list rows (owner-scoped)
    GET  /documents/{id}             one row metadata
    PATCH /documents/{id}/access     {"access_level": "private"|"shared"}
    GET  /documents/{id}/{kind}      stream source|md|json|docx
    GET  /preview/{id}               DOCX -> PDF preview
    POST /preview-input              ad-hoc DOCX upload -> PDF preview

    GET  /                           -> /chat
    GET  /ocr                        single-file OCR page
    /chat/*                          Chainlit RAG chat (chat_app.py): pipeline steps,
                                     citations, uploads, MCP, history
    GET  /chat-files/{thread}/{id}   stored chat element content (owner only)
    POST /rag/chat                   SSE proxy -> orchestrator /chat (pipeline steps + answer)
    POST /rag/query                  JSON proxy -> orchestrator /query
"""

import json
import logging
import os
from pathlib import Path
from urllib.parse import quote

import auth_client
import httpx
import uvicorn
from chainlit.utils import mount_chainlit
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from middleware import AuthMiddleware

_HERE = Path(__file__).parent.resolve()
TEMPLATES_DIR = _HERE / "templates"

OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr_service:8001")
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://rag_service:8003")
# /ocr waits for a GPU swap (LLM stop + Docling load) before converting.
OCR_PROXY_TIMEOUT = float(os.environ.get("OCR_PROXY_TIMEOUT", "1800"))
ORCHESTRATOR_SERVICE_URL = os.environ.get(
    "ORCHESTRATOR_SERVICE_URL",
    "http://orchestrator_service:8004",
)

app = FastAPI(title="Advanced Hybrid RAG UI", docs_url=None, redoc_url=None)
app.add_middleware(AuthMiddleware)


class _SilencePathsFilter(logging.Filter):
    """Drop uvicorn access lines for /health probes so real events show."""

    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__()
        self._paths = paths

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(f" {p} " in msg for p in self._paths)


logging.getLogger("uvicorn.access").addFilter(_SilencePathsFilter(("/health",)))
app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR)), name="static")


# ── Template rendering ──────────────────────────────────────────────────


def _render(name: str, *, user: auth_client.CurrentUser | None = None) -> str:
    """Load a static HTML file and substitute the @@PROFILE_MENU@@ sentinel
    so every page gets the same Profile dropdown without a templating engine."""
    text = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
    if user is None:
        return text
    # Reusable inline SVGs (stroke=currentColor so CSS controls the colour).
    robot_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        '<rect x="4" y="8" width="16" height="11" rx="2.5"/>'
        '<path d="M12 5v3"/><circle cx="12" cy="4" r="1.2"/>'
        '<circle cx="9" cy="13" r="1.2"/><circle cx="15" cy="13" r="1.2"/>'
        '<path d="M9 17h6"/>'
        '<path d="M4 12H2.5"/><path d="M21.5 12H20"/>'
        "</svg>"
    )
    history_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        '<path d="M3 12a9 9 0 1 0 3-6.7"/><polyline points="3 4 3 9 8 9"/>'
        '<path d="M12 7v5l3 2"/>'
        "</svg>"
    )
    settings_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9c.3.4.8.7 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>'
        "</svg>"
    )
    admin_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        '<path d="M12 3l8 3v6c0 5-3.5 8.5-8 9-4.5-.5-8-4-8-9V6l8-3z"/>'
        '<path d="M9 12l2 2 4-4"/>'
        "</svg>"
    )
    logout_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">'
        '<path d="M15 17l5-5-5-5"/><path d="M20 12H9"/>'
        '<path d="M12 21H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h6"/>'
        "</svg>"
    )

    admin_item = (
        f'<a class="profile-item" href="/admin">'
        f'<span class="profile-ico">{admin_svg}</span>'
        f'<span class="profile-item-label">Administration</span></a>'
        if user.role in ("admin", "master")
        else ""
    )
    menu = f"""
<div class="profile-wrap" id="profile-wrap">
  <button type="button" class="profile-btn" id="profile-btn"
          aria-haspopup="true" aria-expanded="false" aria-label="Profile menu">
    <span class="profile-avatar">{robot_svg}</span>
  </button>
  <div class="profile-menu" id="profile-menu" role="menu" hidden>
    <a class="profile-head" href="/settings">
      <span class="profile-head-avatar">{robot_svg}</span>
      <span class="profile-head-text">
        <span class="profile-head-name">{_e(user.username)}</span>
        <span class="profile-head-role">{_e(user.role)} account</span>
      </span>
    </a>
    <div class="profile-divider"></div>
    {admin_item}
    <a class="profile-item" href="/history">
      <span class="profile-ico">{history_svg}</span>
      <span class="profile-item-label">Task History</span>
    </a>
    <a class="profile-item" href="/settings">
      <span class="profile-ico">{settings_svg}</span>
      <span class="profile-item-label">Settings</span>
    </a>
    <div class="profile-divider"></div>
    <button type="button" class="profile-item profile-logout" id="profile-logout">
      <span class="profile-ico">{logout_svg}</span>
      <span class="profile-item-label">Log out</span>
    </button>
  </div>
</div>
<script>
(function() {{
  const wrap = document.getElementById('profile-wrap');
  const btn  = document.getElementById('profile-btn');
  const menu = document.getElementById('profile-menu');
  const logoutBtn = document.getElementById('profile-logout');
  if (!wrap || !btn || !menu) return;
  function close() {{ menu.hidden = true; btn.setAttribute('aria-expanded','false'); }}
  function toggle() {{
    const open = menu.hidden;
    menu.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
  }}
  btn.addEventListener('click', (e) => {{ e.stopPropagation(); toggle(); }});
  document.addEventListener('click', (e) => {{
    if (!wrap.contains(e.target)) close();
  }});
  document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') close(); }});
  if (logoutBtn) {{
    logoutBtn.addEventListener('click', async (e) => {{
      e.preventDefault();
      try {{ await fetch('/logout', {{ method: 'POST', credentials: 'same-origin' }}); }} catch (_) {{}}
      // Prevent back-button from re-showing the previous authed page
      window.location.replace('/login');
    }});
  }}
}})();

/* ── Centered modal helpers (window.uiConfirm / uiPrompt / uiAlert) ── */
(function() {{
  function escapeHtml(s) {{
    return String(s ?? "").replace(/[&<>"']/g, c => ({{
      "&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"
    }}[c]));
  }}
  function buildModal({{ title, body, html, input, inputType, inputPlaceholder,
                        okLabel, cancelLabel, danger, hideCancel }}) {{
    return new Promise((resolve) => {{
      const backdrop = document.createElement('div');
      backdrop.className = 'ui-modal-backdrop';
      backdrop.setAttribute('role', 'dialog');
      backdrop.setAttribute('aria-modal', 'true');
      const inputHtml = input
        ? `<input class="ui-modal-input" id="ui-modal-input" type="${{escapeHtml(inputType || 'text')}}" placeholder="${{escapeHtml(inputPlaceholder || '')}}" />`
        : '';
      const cancelHtml = hideCancel
        ? ''
        : `<button type="button" class="ui-modal-btn ui-modal-btn-cancel" id="ui-modal-cancel">${{escapeHtml(cancelLabel || 'Cancel')}}</button>`;
      const okCls = danger ? 'ui-modal-btn-danger' : 'ui-modal-btn-ok';
      const bodyHtml = html ? body : escapeHtml(body || '');
      backdrop.innerHTML = `
        <div class="ui-modal-card" role="document">
          ${{title ? `<h2 class="ui-modal-title">${{escapeHtml(title)}}</h2>` : ''}}
          ${{bodyHtml ? `<div class="ui-modal-body">${{bodyHtml}}</div>` : ''}}
          ${{inputHtml}}
          <div class="ui-modal-actions">
            ${{cancelHtml}}
            <button type="button" class="ui-modal-btn ${{okCls}}" id="ui-modal-ok">${{escapeHtml(okLabel || 'OK')}}</button>
          </div>
        </div>`;
      document.body.appendChild(backdrop);
      const inputEl  = backdrop.querySelector('#ui-modal-input');
      const okBtn    = backdrop.querySelector('#ui-modal-ok');
      const cancelBtn = backdrop.querySelector('#ui-modal-cancel');
      function close(result) {{
        document.removeEventListener('keydown', onKey, true);
        backdrop.remove();
        resolve(result);
      }}
      function onKey(e) {{
        if (e.key === 'Escape') {{ e.preventDefault(); close(hideCancel ? true : null); }}
        else if (e.key === 'Enter' && (!inputEl || document.activeElement === inputEl || document.activeElement === okBtn)) {{
          e.preventDefault();
          close(input ? (inputEl.value) : true);
        }}
      }}
      document.addEventListener('keydown', onKey, true);
      backdrop.addEventListener('click', (e) => {{
        if (e.target === backdrop) close(hideCancel ? true : null);
      }});
      okBtn.addEventListener('click', () => close(input ? inputEl.value : true));
      if (cancelBtn) cancelBtn.addEventListener('click', () => close(null));
      setTimeout(() => {{ (inputEl || okBtn).focus(); }}, 30);
    }});
  }}
  window.uiConfirm = (opts) => buildModal({{
    title: opts.title || 'Are you sure?',
    body:  opts.body  || '',
    html:  !!opts.html,
    okLabel: opts.okLabel || 'OK',
    cancelLabel: opts.cancelLabel || 'Cancel',
    danger: !!opts.danger,
  }}).then(v => v === true);
  window.uiPrompt = (opts) => buildModal({{
    title: opts.title || '',
    body:  opts.body  || '',
    html:  !!opts.html,
    input: true,
    inputType: opts.inputType || 'text',
    inputPlaceholder: opts.placeholder || '',
    okLabel: opts.okLabel || 'OK',
    cancelLabel: opts.cancelLabel || 'Cancel',
    danger: !!opts.danger,
  }}).then(v => (v === null ? null : String(v)));
  window.uiAlert = (opts) => buildModal({{
    title: opts.title || '',
    body:  opts.body  || '',
    html:  !!opts.html,
    okLabel: opts.okLabel || 'OK',
    hideCancel: true,
    danger: !!opts.danger,
  }}).then(() => undefined);
}})();
</script>
"""
    return text.replace("<!-- @@PROFILE_MENU@@ -->", menu)


def _e(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _require_user(request: Request) -> auth_client.CurrentUser:
    user = getattr(request.state, "user", None)
    if user is None:
        # Should be unreachable because middleware would have redirected,
        # but keep the guard for routes that get called programmatically.
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _require_admin(request: Request) -> auth_client.CurrentUser:
    user = _require_user(request)
    if user.role not in ("admin", "master"):
        raise HTTPException(status_code=403, detail="Admin only")
    return user


def _identity_headers(user: auth_client.CurrentUser) -> dict:
    return {"X-User-Id": user.id, "X-User-Role": user.role}


def _safe_next(raw: str | None) -> str:
    if not raw:
        return "/"
    if not raw.startswith("/") or raw.startswith("//"):
        return "/"
    return raw


# ── Auth pages ──────────────────────────────────────────────────────────


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return _render("login.html")


@app.post("/login")
async def login_submit(
    username: str = Form(...),
    password: str = Form(...),
    next: str | None = Form(None),
):
    cookie, body = await auth_client.login(username, password)
    if cookie is None:
        detail = body.get("detail") if isinstance(body, dict) else "Login failed"
        return RedirectResponse(
            url=f"/login?err={quote(str(detail))}",
            status_code=303,
        )
    redirect_to = _safe_next(next)
    if body.get("must_change_password"):
        redirect_to = "/change-password"
    resp = RedirectResponse(url=redirect_to, status_code=303)
    resp.set_cookie(
        key=auth_client.SESSION_COOKIE_NAME,
        value=cookie,
        httponly=True,
        secure=False,  # mirror auth_service COOKIE_SECURE default; nginx terminates TLS in prod
        samesite="lax",
        path="/",
    )
    return resp


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/logout")
async def logout(request: Request):
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME)
    if sid:
        await auth_client.logout(sid)
    resp = JSONResponse({"status": "ok"})
    resp.delete_cookie(auth_client.SESSION_COOKIE_NAME, path="/")
    return resp


@app.post("/account/delete")
async def delete_account(request: Request):
    user = _require_user(request)
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    code, body = await auth_client.delete_me(sid)
    if code != 204:
        return JSONResponse(content=body, status_code=code)
    await _delete_chat_history(user.id)
    resp = JSONResponse({"status": "ok", "username": user.username})
    resp.delete_cookie(auth_client.SESSION_COOKIE_NAME, path="/")
    return resp


@app.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request):
    return _render("change-password.html")


@app.post("/change-password")
async def change_password_submit(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
):
    if new_password != confirm_password:
        return RedirectResponse(
            "/change-password?err=" + quote("Passwords do not match"),
            status_code=303,
        )
    if len(new_password) < 8:
        return RedirectResponse(
            "/change-password?err=" + quote("Password must be at least 8 characters"),
            status_code=303,
        )
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    status_code, body = await auth_client.change_password(sid, current_password, new_password)
    if status_code != 200:
        detail = body.get("detail") if isinstance(body, dict) else "Update failed"
        return RedirectResponse(
            "/change-password?err=" + quote(str(detail)),
            status_code=303,
        )
    return RedirectResponse("/", status_code=303)


@app.get("/me")
async def me(request: Request):
    user = _require_user(request)
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "must_change_password": user.must_change_password,
    }


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    user = _require_user(request)
    return _render("history.html", user=user)


@app.get("/ingest", response_class=HTMLResponse)
async def ingest_page(request: Request):
    user = _require_user(request)
    return _render("ingest.html", user=user)


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    user = _require_user(request)
    return _render("settings.html", user=user)


# ── Admin console ───────────────────────────────────────────────────────


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    user = _require_admin(request)
    return _render("admin.html", user=user)


@app.get("/admin/users")
async def admin_users_list(request: Request):
    _require_admin(request)
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    code, body = await auth_client.admin_list_users(sid)
    return JSONResponse(content=body, status_code=code)


@app.post("/admin/users")
async def admin_users_create(request: Request, payload: dict = Body(...)):
    _require_admin(request)
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    code, body = await auth_client.admin_create_user(
        sid,
        username=payload.get("username", ""),
        password=payload.get("password", ""),
        role=payload.get("role", "user"),
    )
    return JSONResponse(content=body, status_code=code)


@app.patch("/admin/users/{user_id}")
async def admin_users_update(request: Request, user_id: str, payload: dict = Body(...)):
    _require_admin(request)
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    code, body = await auth_client.admin_update_user(sid, user_id, payload)
    return JSONResponse(content=body, status_code=code)


@app.delete("/admin/users/{user_id}")
async def admin_users_delete(request: Request, user_id: str):
    _require_admin(request)
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME) or ""
    code, body = await auth_client.admin_delete_user(sid, user_id)
    if code == 204:
        await _delete_chat_history(user_id)
        return Response(status_code=204)
    return JSONResponse(content=body, status_code=code)


# ── Main UI pages ───────────────────────────────────────────────────────


@app.get("/")
async def index():
    return RedirectResponse(url="/chat/", status_code=307)


@app.get("/ocr", response_class=HTMLResponse)
async def ocr_page(request: Request):
    user = _require_user(request)
    return _render("index.html", user=user)


@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    user = _require_user(request)
    return _render("batch.html", user=user)


# ── OCR proxies — every call carries identity headers ───────────────────


@app.post("/ocr")
async def ocr(request: Request, file: UploadFile = File(...)):
    user = _require_user(request)
    content = await file.read()
    async with httpx.AsyncClient(timeout=OCR_PROXY_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{OCR_SERVICE_URL}/ocr",
                files={"file": (file.filename, content, file.content_type)},
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


_ALLOWED_KINDS = {"source", "md", "json", "docx"}


@app.get("/documents/{doc_id}/index-status")
async def document_index_status(request: Request, doc_id: str):
    """RAG indexing state for one document. Access is checked through
    ocr_service first because rag_service /ingest/* has no identity check."""
    user = _require_user(request)
    did = quote(doc_id, safe="")
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            doc = await client.get(
                f"{OCR_SERVICE_URL}/documents/{did}", headers=_identity_headers(user)
            )
            if doc.status_code != 200:
                raise HTTPException(status_code=doc.status_code, detail="Document not found")
            resp = await client.get(f"{RAG_SERVICE_URL}/ingest/{did}")
        except httpx.HTTPError:
            raise HTTPException(status_code=503, detail="Service is unavailable")
    if resp.status_code == 404:
        return {"status": "pending"}
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    d = resp.json()
    return {
        "status": d.get("status", "pending"),
        "chunk_count": d.get("chunk_count"),
        "points": d.get("points"),
        "error": d.get("error"),
    }


@app.get("/documents/{doc_id}/images/{filename}")
async def document_image(request: Request, doc_id: str, filename: str):
    user = _require_user(request)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.get(
                f"{OCR_SERVICE_URL}/documents/{doc_id}/images/{filename}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return StreamingResponse(
        iter([resp.content]),
        media_type=resp.headers.get("content-type", "image/png"),
    )


@app.get("/documents/{doc_id}/{kind}")
async def document_artifact(request: Request, doc_id: str, kind: str):
    user = _require_user(request)
    if kind not in _ALLOWED_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown kind: {kind}")
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.get(
                f"{OCR_SERVICE_URL}/documents/{doc_id}/{kind}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    content_type = resp.headers.get("content-type", "application/octet-stream")
    disposition = resp.headers.get("content-disposition")
    headers = {}
    if disposition:
        headers["Content-Disposition"] = disposition
    return StreamingResponse(
        iter([resp.content]),
        media_type=content_type,
        headers=headers,
    )


@app.get("/preview/{doc_id}")
async def preview(request: Request, doc_id: str):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=360) as client:
        try:
            resp = await client.get(
                f"{OCR_SERVICE_URL}/preview/{doc_id}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return StreamingResponse(
        iter([resp.content]),
        media_type="application/pdf",
    )


@app.get("/documents")
async def list_documents(
    request: Request, limit: int = 50, offset: int = 0, status: str | None = None
):
    user = _require_user(request)
    params = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{OCR_SERVICE_URL}/documents",
                params=params,
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    headers = {}
    total = resp.headers.get("x-total-count")
    if total is not None:
        headers["X-Total-Count"] = total
    return JSONResponse(content=resp.json(), headers=headers)


@app.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.delete(
                f"{OCR_SERVICE_URL}/documents/{doc_id}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code == 204:
        return Response(status_code=204)
    detail = resp.text
    try:
        detail = resp.json().get("detail", detail)
    except (ValueError, AttributeError):  # body is not JSON, or not a JSON object
        pass
    raise HTTPException(status_code=resp.status_code, detail=detail)


@app.patch("/documents/{doc_id}/access")
async def set_document_access(request: Request, doc_id: str, payload: dict = Body(...)):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.patch(
                f"{OCR_SERVICE_URL}/documents/{doc_id}/access",
                json=payload,
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    body = (
        resp.json()
        if resp.headers.get("content-type", "").startswith("application/json")
        else {"detail": resp.text}
    )
    return JSONResponse(content=body, status_code=resp.status_code)


@app.get("/documents/{doc_id}")
async def get_document(request: Request, doc_id: str):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                f"{OCR_SERVICE_URL}/documents/{doc_id}",
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


@app.post("/ocr/batch")
async def ocr_batch(request: Request, files: list[UploadFile] = File(...)):
    user = _require_user(request)
    file_tuples = []
    for f in files:
        content = await f.read()
        file_tuples.append(("files", (f.filename, content, f.content_type)))

    async def proxy():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{OCR_SERVICE_URL}/ocr/batch",
                    files=file_tuples,
                    headers=_identity_headers(user),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        yield (
                            "event: error\n"
                            f'data: {{"status": {resp.status_code}, '
                            f'"detail": {body.decode("utf-8", "replace")!r}}}\n\n'
                        ).encode()
                        return
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
            except httpx.ConnectError:
                yield (b'event: error\ndata: {"detail": "OCR service is unavailable"}\n\n')

    return StreamingResponse(
        proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ocr/batch-zip")
async def ocr_batch_zip(request: Request, payload: dict = Body(...)):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resp = await client.post(
                f"{OCR_SERVICE_URL}/ocr/batch-zip",
                json=payload,
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return StreamingResponse(
        iter([resp.content]),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="OCR_results.zip"',
        },
    )


@app.post("/preview-input")
async def preview_input(request: Request, file: UploadFile = File(...)):
    user = _require_user(request)
    content = await file.read()
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{OCR_SERVICE_URL}/preview-input",
                files={"file": (file.filename, content, file.content_type)},
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="OCR service is unavailable")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return StreamingResponse(
        iter([resp.content]),
        media_type="application/pdf",
    )


# ── RAG chat proxies — every call carries identity headers ─────────────


@app.post("/rag/chat")
async def rag_chat(request: Request, payload: dict = Body(...)):
    user = _require_user(request)

    async def proxy():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{ORCHESTRATOR_SERVICE_URL}/chat",
                    json=payload,
                    headers=_identity_headers(user),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        detail = body.decode("utf-8", "replace")
                        yield (
                            "event: error\n"
                            f'data: {{"service": "orchestrator", "detail": '
                            f"{json.dumps(detail[:300])}}}\n\n"
                        ).encode()
                        return
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
            except httpx.ConnectError:
                yield (
                    b"event: error\n"
                    b'data: {"service": "orchestrator", '
                    b'"detail": "RAG service is unavailable"}\n\n'
                )

    return StreamingResponse(
        proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/rag/query")
async def rag_query(request: Request, payload: dict = Body(...)):
    user = _require_user(request)
    async with httpx.AsyncClient(timeout=600) as client:
        try:
            resp = await client.post(
                f"{ORCHESTRATOR_SERVICE_URL}/query",
                json=payload,
                headers=_identity_headers(user),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="RAG service is unavailable")
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ── Chainlit chat (mounted last so the routes above take precedence) ───


def _chat_data_layer():
    import chat_app

    return chat_app._data_layer


async def _delete_chat_history(user_id: str) -> None:
    layer = _chat_data_layer()
    if layer is None:
        return
    try:
        await layer.delete_user_data(user_id)
    except Exception as e:  # history cleanup must never block account deletion
        logging.getLogger("ui_service").warning(
            "chat history cleanup for %s failed: %s", user_id, e
        )


@app.get("/chat-files/{thread_id}/{element_id}")
async def chat_file(request: Request, thread_id: str, element_id: str):
    """Element content of a persisted chat thread (citation text, attachments)."""
    user = _require_user(request)
    layer = _chat_data_layer()
    if layer is None or await layer.get_thread_author(thread_id) != user.id:
        raise HTTPException(status_code=404, detail="Not found")
    found = await layer.element_content(thread_id, element_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Not found")
    body, mime = found
    return Response(
        content=body,
        media_type=mime,
        headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"},
    )


mount_chainlit(app=app, target=str(Path(__file__).parent / "chat_app.py"), path="/chat")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
