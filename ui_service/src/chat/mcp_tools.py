"""User-connected MCP servers.

Users connect their own MCP servers from Chainlit's MCP panel (streamable
HTTP / SSE, limited to MCP_USER_ALLOWED_URLS). For each question:

    1. llm_service /tasks/select-tools picks up to 3 tool calls (or none)
    2. each call runs on the user's MCP session, shown as a tool step
    3. text results become `external_evidence` for the orchestrator, where
       they are reranked with docs / Drive / web and cited as server/tool

Tool output is untrusted: it is only ever evidence, never instructions, and
the validation step checks every claim built on it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import chainlit as cl

from chat import backend

log = logging.getLogger("ui_service.mcp")

MCP_TOOL_TIMEOUT_SEC = float(os.getenv("MCP_TOOL_TIMEOUT_SEC", "20"))
MCP_RESULT_MAX_CHARS = int(os.getenv("MCP_RESULT_MAX_CHARS", "12000"))
SESSION_KEY = "mcp_tools"


def _tool_dict(server: str, tool) -> dict:
    return {"server": server, "name": tool.name, "description": tool.description or "",
            "input_schema": getattr(tool, "inputSchema", None) or {}}


async def on_connect(connection, session) -> None:
    result = await session.list_tools()
    tools = [_tool_dict(connection.name, t) for t in result.tools]
    registry = cl.user_session.get(SESSION_KEY) or {}
    registry[connection.name] = tools
    cl.user_session.set(SESSION_KEY, registry)
    names = ", ".join(f"`{t['name']}`" for t in tools) or "no tools"
    await cl.Message(content=f"🔌 Connected MCP server **{connection.name}**: {names}.\n\n"
                             "Its tools are used automatically when a question needs them.").send()


async def on_disconnect(name: str, session) -> None:
    registry = cl.user_session.get(SESSION_KEY) or {}
    registry.pop(name, None)
    cl.user_session.set(SESSION_KEY, registry)


def connected_tools() -> list[dict]:
    registry = cl.user_session.get(SESSION_KEY) or {}
    return [t for tools in registry.values() for t in tools]


def _result_text(result) -> str:
    parts = []
    for block in getattr(result, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        elif getattr(block, "type", None) == "resource":
            res = getattr(block, "resource", None)
            if getattr(res, "text", None):
                parts.append(res.text)
    if not parts and getattr(result, "structuredContent", None):
        parts.append(json.dumps(result.structuredContent, ensure_ascii=False, indent=2))
    return "\n\n".join(parts).strip()


async def _call(call: dict, idx: int) -> dict | None:
    server, name = call["server"], call["name"]
    mcp_session = cl.context.session.mcp_sessions.get(server)
    if mcp_session is None:
        return None
    args = json.loads(call.get("arguments_json") or "{}")
    async with cl.Step(name=f"{server} · {name}", type="tool") as step:
        step.input = json.dumps(args, ensure_ascii=False, indent=2)
        step.language = "json"
        try:
            result = await asyncio.wait_for(mcp_session.client.call_tool(name, args),
                                            timeout=MCP_TOOL_TIMEOUT_SEC)
        except Exception as e:
            step.is_error = True
            step.output = f"Tool call failed: {e}"
            return None
        text = _result_text(result)
        if getattr(result, "isError", False) or not text:
            step.is_error = bool(getattr(result, "isError", False))
            step.output = text or "(no text output)"
            return None
        step.output = text[:2000] + ("…" if len(text) > 2000 else "")
        return {"id": f"mcp:{server}:{name}:{idx}", "text": text[:MCP_RESULT_MAX_CHARS],
                "source": f"{server}/{name}"}


async def collect_evidence(question: str) -> list[dict]:
    tools = connected_tools()
    if not tools:
        return []
    try:
        selection = await backend.select_tools(question, tools)
    except backend.BackendError as e:
        log.warning("[mcp] tool selection failed: %s", e)
        return []
    calls = selection.get("calls") or []
    if not calls:
        return []
    results = await asyncio.gather(*(_call(c, i) for i, c in enumerate(calls)))
    return [r for r in results if r]
