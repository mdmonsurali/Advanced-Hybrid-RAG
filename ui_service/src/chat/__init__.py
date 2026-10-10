"""Chainlit chat UI (mounted at /chat by main.py, entry point chat_app.py).

    identity.py         session cookie -> fresh auth_service identity per message
    backend.py          httpx clients: orchestrator, ocr_service, rag_service, llm_service
    pipeline_stream.py  orchestrator SSE -> nested cl.Step tree + final answer
    citations.py        [n] citations -> side-panel source elements
    ingestion.py        files dropped in the chat -> Docling OCR -> indexing progress
    settings.py         chat settings + slash commands -> pipeline options
    mcp_tools.py        user-connected MCP servers -> tool calls -> extra evidence
    data_layer.py       chat history persisted in MinIO
"""
