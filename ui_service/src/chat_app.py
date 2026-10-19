"""Chainlit app — the Advanced Hybrid RAG chat (mounted at /chat by main.py).

    auth      header_auth_callback: the existing montashi_session cookie
              (single sign-on with /login; no second password prompt)
    chat      orchestrator /chat SSE rendered as the 10-step pipeline
    upload    files dropped in the chat -> Docling OCR -> indexing
    settings  web / Drive switches, answer style, /docs /web /drive /catalog
    MCP       user-connected MCP servers feed tool results into answers
    history   threads persisted in MinIO (chat/data_layer.py)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import chainlit as cl
from chainlit.config import config as cl_config

import auth_client
from chat import ingestion, mcp_tools, pipeline_stream, settings
from chat.data_layer import MinioDataLayer
from chat.identity import SessionExpired, current_user, headers_for, session_cookie_from, to_chainlit_user

log = logging.getLogger("ui_service.chat")

HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "6"))
CHAT_HISTORY_ENABLED = os.getenv("CHAT_HISTORY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
MCP_USER_ALLOWED_URLS = [u.strip() for u in os.getenv("MCP_USER_ALLOWED_URLS", "").split(",") if u.strip()]

# MCP: user-provided servers are only possible for allowlisted URL prefixes.
cl_config.features.mcp.enabled = True
cl_config.features.mcp.user_servers.enabled = bool(MCP_USER_ALLOWED_URLS)
cl_config.features.mcp.user_servers.allowed_urls = MCP_USER_ALLOWED_URLS
# Chainlit's own session should not outlive the auth_service session.
cl_config.project.user_session_timeout = int(os.getenv("SESSION_TTL_HOURS", "12")) * 3600

_SESSION_EXPIRED = ("🔒 Your session has expired. [Sign in again](/login?next=/chat) "
                    "to continue.")

if CHAT_HISTORY_ENABLED:
    _data_layer = MinioDataLayer()

    @cl.data_layer
    def get_data_layer():
        return _data_layer
else:
    _data_layer = None


# ── Auth ─────────────────────────────────────────────────────────────────

@cl.header_auth_callback
async def header_auth(headers) -> Optional[cl.User]:
    sid = session_cookie_from(headers.get("cookie", ""))
    user = await auth_client.validate(sid or "")
    if user is None or user.must_change_password:
        return None
    return to_chainlit_user(user)


@cl.on_logout
async def on_logout(request, response):
    sid = request.cookies.get(auth_client.SESSION_COOKIE_NAME)
    if sid:
        await auth_client.logout(sid)
    response.delete_cookie(auth_client.SESSION_COOKIE_NAME, path="/")
    return {"success": True}


# ── Session setup ────────────────────────────────────────────────────────

def _role() -> str:
    user = cl.user_session.get("user")
    return (getattr(user, "metadata", None) or {}).get("role", "user")


async def _init_session(history: list[dict] | None = None) -> None:
    role = _role()
    values = await cl.ChatSettings(settings.widgets(role)).send()
    cl.user_session.set("settings", {**settings.DEFAULTS, **(values or {})})
    cl.user_session.set("history", history or [])
    await cl.context.emitter.set_commands(settings.commands(role))


@cl.set_starters
async def starters(user: Optional[cl.User] = None, language: Optional[str] = None):
    items = [
        cl.Starter(label="Ask a fact", message="What does the agreement say about late payment interest?"),
        cl.Starter(label="Compare", message="Compare the Basic and Pro plans."),
        cl.Starter(label="My documents", message="How many documents have I uploaded, and of which types?",
                   command="catalog"),
        cl.Starter(label="Search the web", message="What is the latest SGLang release?", command="web"),
    ]
    role = (getattr(user, "metadata", None) or {}).get("role", "user")
    if settings.can_use_drive(role):
        items.append(cl.Starter(label="Google Drive", message="Find my budget spreadsheet.",
                                command="drive"))
    return items


@cl.on_chat_start
async def on_chat_start():
    await _init_session()


@cl.on_settings_update
async def on_settings_update(values: dict):
    cl.user_session.set("settings", {**settings.DEFAULTS, **(values or {})})


_FOOTER_RE = re.compile(r"\n\n(\*\*Sources:\*\*|\*(✅|⚠️|❓|🔍|⛔)).*", re.DOTALL)


@cl.on_chat_resume
async def on_chat_resume(thread):
    history = []
    for step in thread.get("steps", []):
        if step.get("type") == "user_message" and step.get("output"):
            history.append({"role": "user", "content": step["output"]})
        elif step.get("type") == "assistant_message" and step.get("output") and not step.get("parentId"):
            history.append({"role": "assistant", "content": _FOOTER_RE.sub("", step["output"])})
    await _init_session(history[-HISTORY_TURNS * 2:])


# ── Messages ─────────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    try:
        user = await current_user()
    except SessionExpired:
        await cl.Message(content=_SESSION_EXPIRED).send()
        return
    headers = headers_for(user)

    files = ingestion.uploaded_files(message)
    if files:
        await ingestion.ingest(files, headers)

    question = (message.content or "").strip()
    if not question:
        return
    await answer(question, headers, command=message.command)


async def answer(question: str, headers: dict, command: Optional[str] = None,
                 route_hint: Optional[list[str]] = None) -> None:
    prefs = cl.user_session.get("settings") or settings.DEFAULTS
    options = settings.pipeline_options(prefs, command)
    if route_hint:
        options["route_hint"] = route_hint
    if options["route_hint"] != ["catalog"]:
        options["external_evidence"] = await mcp_tools.collect_evidence(question)
    # Groups this thread's questions into one Langfuse session.
    options["session_id"] = cl.context.session.thread_id

    history = cl.user_session.get("history") or []
    final = await pipeline_stream.run(question, history, options, headers,
                                      show_steps=bool(prefs.get("show_steps", True)))
    if final:
        history = [*history, {"role": "user", "content": question},
                   {"role": "assistant", "content": final.get("answer_markdown", "")}]
        cl.user_session.set("history", history[-HISTORY_TURNS * 2:])


# ── Actions (from ingestion messages) ────────────────────────────────────

@cl.action_callback("toggle_share")
async def on_toggle_share(action: cl.Action):
    try:
        user = await current_user()
    except SessionExpired:
        await cl.Message(content=_SESSION_EXPIRED).send()
        return
    from chat import backend
    p = action.payload
    level = "private" if p.get("shared") else "shared"
    try:
        await backend.set_access(p["doc_id"], level, headers_for(user))
    except backend.BackendError as e:
        await cl.Message(content=f"⚠️ Could not change sharing: {e.detail}").send()
        return
    await action.remove()
    text = (f"🔓 **{p['name']}** is now shared: every user can search it." if level == "shared"
            else f"🔒 **{p['name']}** is private again.")
    await cl.Message(content=text,
                     actions=ingestion.doc_actions(p["doc_id"], p["name"], level == "shared")).send()


@cl.action_callback("summarize_document")
async def on_summarize(action: cl.Action):
    try:
        user = await current_user()
    except SessionExpired:
        await cl.Message(content=_SESSION_EXPIRED).send()
        return
    name = action.payload.get("name", "the document")
    question = f"Summarize the document “{name}”."
    await cl.Message(content=question, author=user.username, type="user_message").send()
    await answer(question, headers_for(user), route_hint=["docs"])


# ── MCP ──────────────────────────────────────────────────────────────────

@cl.on_mcp_connect
async def on_mcp_connect(connection, session):
    await mcp_tools.on_connect(connection, session)


@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str, session):
    await mcp_tools.on_disconnect(name, session)
