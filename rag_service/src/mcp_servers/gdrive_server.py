"""FastMCP server: the user's Google Drive (read-only).

Mounted by main.py at /mcp/gdrive (streamable HTTP). Any MCP client can use
it with the MCP_TOKEN bearer token; rag_service's router calls it for the
`gdrive` route.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Optional

from fastmcp import FastMCP
from pydantic import Field

from mcp_servers import gdrive_api

mcp = FastMCP(
    "gdrive",
    instructions="Search and read files in the connected Google Drive (read-only).",
)


@mcp.tool
async def search_files(
    query: Annotated[str, Field(description="Words to find in file names or contents")],
    mime_types: Annotated[
        Optional[list[str]], Field(description="Restrict to these MIME types")
    ] = None,
    limit: Annotated[int, Field(ge=1, le=25)] = 10,
) -> list[dict]:
    """Full-text search across Drive. Returns [{id, name, mimeType, modifiedTime, webViewLink}]."""
    return await asyncio.to_thread(gdrive_api.search_files, query, mime_types, limit)


@mcp.tool
async def read_file(
    file_id: Annotated[str, Field(description="Drive file id from search_files")],
    max_chars: Annotated[int, Field(ge=500, le=100000)] = 20000,
) -> dict:
    """Read a file as text. Google Docs/Slides export as text, Sheets as CSV,
    PDFs via text extraction; other binary files return metadata only."""
    return await asyncio.to_thread(gdrive_api.read_file, file_id, max_chars)


@mcp.tool
async def get_file_metadata(
    file_id: Annotated[str, Field(description="Drive file id")],
) -> dict:
    """Name, type, modified time, owners and link of one file."""
    return await asyncio.to_thread(gdrive_api.get_file_metadata, file_id)
