"""Structure-aware chunking of DoclingDocuments with docling-core's
HybridChunker.

- Splits along the document hierarchy (sections / headings), so a chunk
  never mixes unrelated topics.
- Token-bounded with the embedder's own tokenizer (RAG_CHUNK_TOKENS), and
  undersized neighbours under the same headings are merged, so complete
  ideas stay together.
- Tables are serialized as Markdown and split by row groups with the header
  repeated when they exceed the budget; tables.py then attaches the caption
  and the explanatory paragraph that precedes them.
"""

from __future__ import annotations

import logging
import os

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DocItemLabel, DoclingDocument, RefItem, TableItem

from chunking_techniques import RawChunk
from chunking_techniques.tables import table_context

log = logging.getLogger("rag_service.chunking")

CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", "400"))

_LIST_LABELS = {DocItemLabel.LIST_ITEM}


class _MarkdownTableProvider(ChunkingSerializerProvider):
    """Serialize tables as Markdown (readable for the LLM and keeps rows
    intact) instead of the default row/column triplets."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())


def build_chunker(hf_tokenizer, max_tokens: int = CHUNK_TOKENS) -> HybridChunker:
    return HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=max_tokens),
        serializer_provider=_MarkdownTableProvider(),
        merge_peers=True,
        repeat_table_header=True,
    )


def _resolve(doc: DoclingDocument, doc_items) -> list:
    """Chunk metadata carries base-class DocItems; resolve each through its
    self_ref to the real document node (TableItem, TextItem, ...)."""
    out = []
    for it in doc_items:
        try:
            out.append(RefItem(cref=it.self_ref).resolve(doc))
        except Exception:
            out.append(it)
    return out


def _pages(doc_items) -> tuple[int | None, int | None]:
    pages = [p.page_no for item in doc_items for p in (getattr(item, "prov", None) or [])]
    return (min(pages), max(pages)) if pages else (None, None)


def _chunk_type(doc_items) -> str:
    if any(isinstance(it, TableItem) for it in doc_items):
        return "table"
    if doc_items and all(getattr(it, "label", None) in _LIST_LABELS for it in doc_items):
        return "list"
    return "text"


def chunk_document(doc: DoclingDocument, chunker: HybridChunker) -> list[RawChunk]:
    out: list[RawChunk] = []
    for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        body = chunk.text.strip()
        if not body:
            continue
        items = _resolve(doc, chunk.meta.doc_items or [])
        headings = list(chunk.meta.headings or [])
        ctype = _chunk_type(items)

        context_lines: list[str] = []
        if ctype == "table":
            for table in (it for it in items if isinstance(it, TableItem)):
                context_lines.extend(t for t in table_context(doc, table) if t not in body)

        text = "\n".join([*context_lines, body]) if context_lines else body
        heading_path = " > ".join(headings)
        embed_text = f"{heading_path}\n{text}" if heading_path else text
        start, end = _pages(items)
        out.append(RawChunk(index=len(out), text=text, embed_text=embed_text,
                            headings=headings, page_start=start, page_end=end,
                            chunk_type=ctype))
    log.info("[chunk] %d chunks (%d tables)", len(out),
             sum(1 for c in out if c.chunk_type == "table"))
    return out
