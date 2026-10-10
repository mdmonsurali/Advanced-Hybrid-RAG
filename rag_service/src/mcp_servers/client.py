"""fastmcp.Client wrappers used by the router to turn MCP tool results into
Evidence. Every call has a timeout; any failure raises RouteUnavailable so
the executor can report the route as `unavailable` and carry on.

The servers are mounted in this same process, but they're called over HTTP
like any external MCP server, so GDRIVE_MCP_URL / WEBSEARCH_MCP_URL can point
at a separately hosted server without code changes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os

from fastmcp import Client

from chunking_techniques.text import chunk_text
from schemas import Evidence, QueryPlan

log = logging.getLogger("rag_service.mcp_client")

RAG_SERVICE_PORT = os.getenv("RAG_SERVICE_PORT", "8003")
GDRIVE_MCP_URL = os.getenv("GDRIVE_MCP_URL", f"http://127.0.0.1:{RAG_SERVICE_PORT}/mcp/gdrive/")
WEBSEARCH_MCP_URL = os.getenv("WEBSEARCH_MCP_URL", f"http://127.0.0.1:{RAG_SERVICE_PORT}/mcp/websearch/")
MCP_TOKEN = os.getenv("MCP_TOKEN", "")
MCP_TIMEOUT_SEC = float(os.getenv("MCP_TIMEOUT_SEC", "20"))
GDRIVE_MAX_FILES = int(os.getenv("GDRIVE_MAX_FILES", "3"))
WEB_MAX_PAGES = int(os.getenv("WEB_MAX_PAGES", "3"))


class RouteUnavailable(RuntimeError):
    pass


async def call_tool(url: str, name: str, args: dict):
    try:
        async with Client(url, auth=MCP_TOKEN or None, timeout=MCP_TIMEOUT_SEC) as client:
            result = await client.call_tool(name, args, timeout=MCP_TIMEOUT_SEC)
    except Exception as e:
        raise RouteUnavailable(f"{name} failed: {e}") from e
    if result.data is not None:
        return result.data
    if result.structured_content is not None:
        # Non-object results are wrapped as {"result": ...} by FastMCP.
        return result.structured_content.get("result", result.structured_content)
    return None


def _search_terms(plan: QueryPlan) -> str:
    """Drive full-text search matches phrases literally, so prefer the
    entities (names, titles, codes) over a long rewritten sentence."""
    if plan.entities:
        return " ".join(plan.entities[:3])
    return plan.rewritten or plan.standalone_question


def _chunks_to_evidence(prefix: str, text: str, **fields) -> list[Evidence]:
    return [Evidence(id=f"{prefix}:{i}", text=chunk, chunk_index=i, **fields)
            for i, chunk in enumerate(chunk_text(text))]


async def gdrive_evidence(plan: QueryPlan) -> list[Evidence]:
    files = await call_tool(GDRIVE_MCP_URL, "search_files",
                            {"query": _search_terms(plan), "limit": GDRIVE_MAX_FILES * 2})
    files = list(files or [])[:GDRIVE_MAX_FILES]

    async def _read(f: dict) -> list[Evidence]:
        try:
            doc = await call_tool(GDRIVE_MCP_URL, "read_file", {"file_id": f["id"]})
        except RouteUnavailable as e:
            log.warning("[gdrive] read %s failed: %s", f.get("name"), e)
            return []
        if not doc or not doc.get("text"):
            return []
        return _chunks_to_evidence(
            f"gdrive:{f['id']}", doc["text"], route="gdrive",
            source=f.get("name", "Google Drive file"), url=f.get("webViewLink"),
            date=f.get("modifiedTime"), owned=True)

    groups = await asyncio.gather(*(_read(f) for f in files))
    return [ev for g in groups for ev in g]


async def web_evidence(plan: QueryPlan) -> list[Evidence]:
    query = plan.rewritten or plan.standalone_question
    results = list(await call_tool(WEBSEARCH_MCP_URL, "search",
                                   {"query": query, "limit": WEB_MAX_PAGES + 2}) or [])

    async def _page(r: dict) -> list[Evidence]:
        key = hashlib.sha1(r["url"].encode()).hexdigest()[:12]
        fields = dict(route="web", source=r.get("title") or r["url"], url=r["url"],
                      date=r.get("published"), owned=False)
        try:
            page = await call_tool(WEBSEARCH_MCP_URL, "fetch_page", {"url": r["url"]})
            text = (page or {}).get("text") or ""
        except RouteUnavailable as e:
            log.info("[web] fetch %s failed, using snippet: %s", r["url"], e)
            text = ""
        if not text:
            text = r.get("snippet") or ""
        return _chunks_to_evidence(f"web:{key}", text, **fields) if text else []

    groups = await asyncio.gather(*(_page(r) for r in results[:WEB_MAX_PAGES]))
    return [ev for g in groups for ev in g]
