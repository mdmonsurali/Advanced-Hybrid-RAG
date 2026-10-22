"""FastMCP server: public web search.

Mounted by main.py at /mcp/websearch (streamable HTTP). Any MCP client can
use it with the MCP_TOKEN bearer token; rag_service's router calls it for
the `web` route.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Optional

from fastmcp import FastMCP
from pydantic import Field

from mcp_servers import websearch_providers as providers

mcp = FastMCP(
    "websearch",
    instructions="Search the public web and read pages as plain text.",
)


@mcp.tool
async def search(
    query: Annotated[str, Field(description="Search query")],
    limit: Annotated[int, Field(ge=1, le=10, description="Number of results")] = 5,
    recency_days: Annotated[
        Optional[int], Field(ge=1, description="Only results from the last N days")
    ] = None,
) -> list[dict]:
    """Search the web. Returns [{title, url, snippet, published}]."""
    return await asyncio.to_thread(providers.search, query, limit, recency_days)


@mcp.tool
async def fetch_page(
    url: Annotated[str, Field(description="http(s) URL to read")],
    max_chars: Annotated[int, Field(ge=500, le=50000)] = 15000,
) -> dict:
    """Download a web page and return its main text (boilerplate removed)."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("url must start with http:// or https://")
    return await asyncio.to_thread(providers.fetch_page, url, max_chars)
