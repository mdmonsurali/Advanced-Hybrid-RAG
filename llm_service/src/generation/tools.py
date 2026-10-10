"""Tool selection for user-connected MCP servers.

The UI holds the MCP sessions (the user connected them in the browser); this
task only decides which tools to call and with which arguments. Calls that
name unknown tools or carry invalid JSON arguments are dropped.
"""

from __future__ import annotations

import json
import logging

from llm_client.client import chat_json
from prompts.tools import select_tools_messages
from schemas import SelectToolsOutput, ToolSpec

log = logging.getLogger("llm_service.tools")


async def select(question: str, tools: list[ToolSpec]) -> SelectToolsOutput:
    out = await chat_json(select_tools_messages(question, tools), SelectToolsOutput,
                          max_tokens=512)
    known = {(t.server, t.name) for t in tools}
    calls = []
    for call in out.calls:
        if (call.server, call.name) not in known:
            log.info("[tools] dropping unknown tool %s/%s", call.server, call.name)
            continue
        try:
            args = json.loads(call.arguments_json or "{}")
        except json.JSONDecodeError:
            log.info("[tools] dropping %s/%s: invalid arguments", call.server, call.name)
            continue
        if not isinstance(args, dict):
            continue
        call.arguments_json = json.dumps(args, ensure_ascii=False)
        calls.append(call)
    return SelectToolsOutput(calls=calls, reason=out.reason)
