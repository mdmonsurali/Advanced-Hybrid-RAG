"""Structure-aware chunking of DoclingDocuments with docling-core's
HybridChunker, with tables and figures chunked separately.

- Splits along the document hierarchy (sections / headings), so a chunk
  never mixes unrelated topics.
- Token-bounded with the embedder's own tokenizer (RAG_CHUNK_TOKENS), and
  undersized neighbours under the same headings are merged, so complete
  ideas stay together.
- Tables and pictures (with their captions / footnotes) are left out of the
  HybridChunker pass, so they are never merged into a text chunk. Tables
  become row-group chunks (tables.py) and pictures figure chunks
  (figures.py), placed back at their position in reading order.
- Cross-page table continuations must already be merged (table_merge.py).
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
from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    PictureItem,
    RefItem,
    SectionHeaderItem,
    TableItem,
    TitleItem,
)
from pydantic import Field

from chunking_techniques import RawChunk
from chunking_techniques.figures import figure_chunk
from chunking_techniques.table_merge import RowPages
from chunking_techniques.tables import table_chunks

log = logging.getLogger("rag_service.chunking")

CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", "400"))

_LIST_LABELS = {DocItemLabel.LIST_ITEM}
_FLOATS = (TableItem, PictureItem)


class _TextOnlySerializer(ChunkingDocSerializer):
    """Serializes everything except tables, pictures and their captions /
    footnotes, which get their own chunks."""

    skip_refs: set[str] = Field(default_factory=set)

    def serialize(
        self, *, item: NodeItem | None = None, visited: set[str] | None = None, **kwargs
    ) -> SerializationResult:
        if item is not None and (isinstance(item, _FLOATS) or item.self_ref in self.skip_refs):
            if visited is not None:
                visited.add(item.self_ref)
            return create_ser_result()
        return super().serialize(item=item, visited=visited, **kwargs)


class _TextOnlyProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        skip = {
            ref.cref
            for item in [*doc.tables, *doc.pictures]
            for ref in [*item.captions, *item.footnotes]
        }
        return _TextOnlySerializer(doc=doc, skip_refs=skip)


def build_chunker(hf_tokenizer, max_tokens: int = CHUNK_TOKENS) -> HybridChunker:
    return HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=max_tokens),
        serializer_provider=_TextOnlyProvider(),
        merge_peers=True,
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
    if doc_items and all(getattr(it, "label", None) in _LIST_LABELS for it in doc_items):
        return "list"
    return "text"


def _layout(doc: DoclingDocument) -> tuple[dict[str, int], list[tuple[NodeItem, list[str]]]]:
    """Reading-order position of every item, and each table / picture with
    the heading path it sits under."""
    order: dict[str, int] = {}
    floats: list[tuple[NodeItem, list[str]]] = []
    heading_by_level: dict[int, str] = {}
    for item, _level in doc.iterate_items(with_groups=False):
        order[item.self_ref] = len(order)
        if isinstance(item, (TitleItem, SectionHeaderItem)):
            level = item.level if isinstance(item, SectionHeaderItem) else 0
            for k in [k for k in heading_by_level if k >= level]:
                del heading_by_level[k]
            heading_by_level[level] = item.text
        elif isinstance(item, _FLOATS):
            floats.append((item, [heading_by_level[k] for k in sorted(heading_by_level)]))
    return order, floats


def chunk_document(
    doc: DoclingDocument,
    chunker: HybridChunker,
    figure_descriptions: dict[str, str] | None = None,
    row_pages: RowPages | None = None,
) -> list[RawChunk]:
    """`figure_descriptions` maps picture self_ref -> generated description,
    `row_pages` is what table_merge returned for this document."""
    figure_descriptions = figure_descriptions or {}
    row_pages = row_pages or {}
    order, floats = _layout(doc)
    count_tokens = chunker.tokenizer.count_tokens
    max_tokens = chunker.tokenizer.get_max_tokens()
    keyed: list[tuple[tuple[int, int], RawChunk]] = []

    for chunk in chunker.chunk(dl_doc=doc):
        body = chunk.text.strip()
        if not body:
            continue
        items = _resolve(doc, chunk.meta.doc_items or [])
        headings = list(chunk.meta.headings or [])
        heading_path = " > ".join(headings)
        start, end = _pages(items)
        pos = min((order.get(it.self_ref, 0) for it in items), default=0)
        keyed.append(
            (
                (pos, 0),
                RawChunk(
                    index=0,
                    text=body,
                    embed_text=f"{heading_path}\n{body}" if heading_path else body,
                    headings=headings,
                    page_start=start,
                    page_end=end,
                    chunk_type=_chunk_type(items),
                ),
            )
        )

    for item, headings in floats:
        heading_path = " > ".join(headings)
        start, end = _pages([item])
        if isinstance(item, TableItem):
            pieces = table_chunks(
                doc, item, heading_path, count_tokens, max_tokens, row_pages.get(item.self_ref)
            )
            ctype = "table"
        else:
            piece = figure_chunk(
                doc, item, heading_path, figure_descriptions.get(item.self_ref, "")
            )
            pieces = [(*piece, None)] if piece else []
            ctype = "figure"
        for sub, (text, embed_text, pages) in enumerate(pieces):
            keyed.append(
                (
                    (order[item.self_ref], sub),
                    RawChunk(
                        index=0,
                        text=text,
                        embed_text=embed_text,
                        headings=headings,
                        page_start=pages[0] if pages else start,
                        page_end=pages[1] if pages else end,
                        chunk_type=ctype,
                    ),
                )
            )

    keyed.sort(key=lambda kc: kc[0])
    out = [c for _, c in keyed]
    for i, c in enumerate(out):
        c.index = i
    log.info(
        "[chunk] %d chunks (%d table, %d figure)",
        len(out),
        sum(1 for c in out if c.chunk_type == "table"),
        sum(1 for c in out if c.chunk_type == "figure"),
    )
    return out
