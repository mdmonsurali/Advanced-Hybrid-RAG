"""Prompt for choosing which user-connected MCP tools to call for a question."""

from __future__ import annotations

import json

from prompts import registry
from schemas import ToolSpec

SELECT_TOOLS_SYSTEM = registry.register("rag/select-tools", """\
You decide whether external tools are needed to answer a question. The \
tools come from MCP servers the user connected. Their results will be used \
as evidence next to the user's documents and the web.

Rules:
- Call a tool only if its description clearly matches what the question needs.
- At most 3 calls. No calls is a valid and common answer.
- arguments_json must be a JSON object that matches the tool's input schema.
- Use the exact server and tool names given.
Explain the choice in one short sentence (reason).""")

_MAX_SCHEMA_CHARS = 1200


def _tool_block(t: ToolSpec) -> dict:
    schema = json.dumps(t.input_schema, ensure_ascii=False)
    if len(schema) > _MAX_SCHEMA_CHARS:
        schema = schema[:_MAX_SCHEMA_CHARS] + "…"
    return {"server": t.server, "name": t.name, "description": t.description[:500],
            "input_schema": schema}


def select_tools_messages(question: str, tools: list[ToolSpec]) -> list[dict]:
    payload = json.dumps({"question": question, "tools": [_tool_block(t) for t in tools]},
                         ensure_ascii=False)
    return registry.messages("rag/select-tools", [{"role": "user", "content": payload}])
