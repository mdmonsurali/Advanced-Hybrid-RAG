"""MCP servers (FastMCP) and the client rag_service uses to call them.

Named `mcp_servers` rather than `mcp` so it doesn't shadow the official
`mcp` SDK package that fastmcp is built on.

    websearch_server.py     FastMCP("websearch"): search, fetch_page
    websearch_providers.py  ddgs (default, no key) | tavily | brave
    gdrive_server.py        FastMCP("gdrive"): search_files, read_file, get_file_metadata
    gdrive_api.py           Google Drive API v3 wrapper (OAuth token or service account)
    gdrive_auth.py          one-time host script: OAuth consent -> token.json
    client.py               fastmcp.Client wrappers -> Evidence for the router
"""
