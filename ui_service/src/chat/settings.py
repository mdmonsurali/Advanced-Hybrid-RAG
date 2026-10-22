"""Chat settings and slash commands -> orchestrator pipeline options.

Settings (per chat):  Search the web · Search Google Drive (allowed roles) ·
                      Show pipeline steps · Answer style
Commands:             /docs  /web  /drive  /catalog  force a route

The backend re-checks every route against the user's permissions, so these
controls can only narrow what is searched, never widen it.
"""

from __future__ import annotations

import os

from chainlit.input_widget import Select, Switch

GDRIVE_ALLOWED_ROLES = {
    r.strip() for r in os.getenv("GDRIVE_ALLOWED_ROLES", "master,admin").split(",") if r.strip()
}

DEFAULTS = {
    "web": True,
    "gdrive": True,
    "show_steps": True,
    "answer_style": "auto",
    "debug_chunks": False,
}

_COMMAND_ROUTES = {"docs": ["docs"], "web": ["web"], "drive": ["gdrive"], "catalog": ["catalog"]}


def can_use_drive(role: str) -> bool:
    return role in GDRIVE_ALLOWED_ROLES


def widgets(role: str) -> list:
    items = [
        Switch(
            id="web",
            label="Search the web",
            initial=DEFAULTS["web"],
            description="Allow the public web as a source (web-search MCP)",
        )
    ]
    if can_use_drive(role):
        items.append(
            Switch(
                id="gdrive",
                label="Search Google Drive",
                initial=DEFAULTS["gdrive"],
                description="Allow the connected Google Drive as a source",
            )
        )
    if role in {"admin", "master"}:
        items.append(
            Switch(
                id="debug_chunks",
                label="Debug: show retrieved chunks",
                initial=DEFAULTS["debug_chunks"],
                description="Admin only: show every retrieved chunk, the reranked ones and the exact evidence sent to the LLM (needs pipeline steps on)",
            )
        )
    items += [
        Switch(
            id="show_steps",
            label="Show pipeline steps",
            initial=DEFAULTS["show_steps"],
            description="Display the 10-step RAG pipeline while answering",
        ),
        Select(
            id="answer_style",
            label="Answer style",
            values=["auto", "concise", "detailed"],
            initial_value=DEFAULTS["answer_style"],
        ),
    ]
    return items


def commands(role: str) -> list[dict]:
    cmds = [
        {
            "id": "docs",
            "icon": "file-search",
            "description": "Search only documents (yours + shared)",
        },
        {"id": "web", "icon": "globe", "description": "Search the web"},
        {"id": "catalog", "icon": "library", "description": "Ask about your document collection"},
    ]
    if can_use_drive(role):
        cmds.insert(2, {"id": "drive", "icon": "hard-drive", "description": "Search Google Drive"})
    return cmds


def pipeline_options(settings: dict | None, command: str | None) -> dict:
    s = {**DEFAULTS, **(settings or {})}
    deny = []
    if not s.get("web"):
        deny.append("web")
    if not s.get("gdrive"):
        deny.append("gdrive")
    return {
        "route_hint": _COMMAND_ROUTES.get(command or "", []),
        "deny_routes": deny,
        "answer_style": s.get("answer_style") or "auto",
        "debug": bool(s.get("debug_chunks")),
    }
