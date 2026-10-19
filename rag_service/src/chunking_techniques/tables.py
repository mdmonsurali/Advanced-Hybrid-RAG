"""Table chunks: searchable rows, readable Markdown, and their explanation.

A table on its own ("| 4.1 | 12 | yes |") is hard to retrieve and easy to
misread, and a Markdown grid embeds poorly against a question about one
value. So every table is cut into row groups (each within the token budget),
and each group gets two texts:

  embed_text  heading path + caption + one "Column: value; ..." line per row,
              with multi-level headers resolved to a path ("Leave > Sick")
              so every value carries its column name
  text        what the LLM sees: caption, the closest preceding paragraph in
              the same section (the sentence that usually says what the
              table shows), then a Markdown table with the full header

Run table_merge first so a table split over several pages is one table here.
"""

from __future__ import annotations

from typing import Callable

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    NodeItem,
    PictureItem,
    TableItem,
    TextItem,
)

_PARAGRAPH_LABELS = {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH}
_SECTION_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
# Look back at most this many items for the explanatory paragraph.
_MAX_LOOKBACK = 4
_MAX_CONTEXT_CHARS = 600


def preceding_paragraph(doc: DoclingDocument, target: NodeItem) -> str | None:
    """The closest paragraph before `target` in the same section."""
    previous: list = []
    for item, _level in doc.iterate_items(with_groups=False):
        if item.self_ref == target.self_ref:
            break
        previous.append(item)
    for item in reversed(previous[-_MAX_LOOKBACK:]):
        label = getattr(item, "label", None)
        if label in _SECTION_LABELS or isinstance(item, (TableItem, PictureItem)):
            return None                 # never borrow another section's / float's text
        if isinstance(item, TextItem) and label in _PARAGRAPH_LABELS and item.text.strip():
            text = item.text.strip()
            return text if len(text) <= _MAX_CONTEXT_CHARS else text[:_MAX_CONTEXT_CHARS] + "…"
    return None


def context_lines(doc: DoclingDocument, item: NodeItem, kind: str) -> list[str]:
    """The caption (first, prefixed with `kind` unless it already starts
    with it) plus the explanatory paragraph before the item."""
    lines: list[str] = []
    caption = item.caption_text(doc).strip()
    if caption:
        # "Table 2: ..." already says what it is.
        lines.append(caption if caption.lower().startswith(kind.lower()) else f"{kind}: {caption}")
    paragraph = preceding_paragraph(doc, item)
    if paragraph and paragraph != caption:
        lines.append(paragraph)
    return lines


def _header_row_count(grid) -> int:
    """Leading rows holding column-header cells; row 0 if none is flagged."""
    n = 0
    while n < len(grid) and any(c.column_header for c in grid[n]):
        n += 1
    return n if n else (1 if len(grid) > 1 else 0)


def _column_names(grid, header_rows: int, num_cols: int) -> list[str]:
    names = []
    for j in range(num_cols):
        parts: list[str] = []
        for i in range(header_rows):
            text = grid[i][j].text.strip()
            if text and text not in parts:
                parts.append(text)
        names.append(" > ".join(parts) or f"Column {j + 1}")
    return names


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _md_row(values: list[str]) -> str:
    return "| " + " | ".join(_md_cell(v) for v in values) + " |"


def _rows(grid, header_rows: int, names: list[str]) -> list[tuple[int, str, str]]:
    """(row index, key-value line, Markdown row) per body row."""
    out: list[tuple[int, str, str]] = []
    section = ""
    for r in range(header_rows, len(grid)):
        row = grid[r]
        # A cell spanning several columns appears once per column in the grid.
        firsts = [j == 0 or row[j] is not row[j - 1] for j in range(len(row))]
        values = [c.text.strip() if first else "" for c, first in zip(row, firsts)]
        spans_row = (row[0].end_col_offset_idx - row[0].start_col_offset_idx) >= len(names)
        if len(names) > 1 and values[0] and (spans_row or any(c.row_section for c in row)):
            # One cell across the whole row: a section label for the rows below.
            section = values[0]
            out.append((r, f"Section: {section}", _md_row(values)))
            continue
        pairs = [f"{name}: {value}" for name, value in zip(names, values) if value]
        if not pairs:
            continue
        kv = "; ".join(pairs)
        out.append((r, f"[{section}] {kv}" if section else kv, _md_row(values)))
    return out


def _page_of(row: int, row_pages: list[tuple[int, int]]) -> int:
    return next(page for first, page in reversed(row_pages) if first <= row)


def table_chunks(doc: DoclingDocument, table: TableItem, heading_path: str,
                 count_tokens: Callable[[str], int], max_tokens: int,
                 row_pages: list[tuple[int, int]] | None = None,
                 ) -> list[tuple[str, str, tuple[int, int] | None]]:
    """(text, embed_text, (first page, last page) or None) per row group of
    `table`. Pages come from `row_pages` (table_merge) for merged tables."""
    grid = table.data.grid
    if not grid:
        return []
    header_rows = _header_row_count(grid)
    names = _column_names(grid, header_rows, table.data.num_cols)
    rows = _rows(grid, header_rows, names)

    context = context_lines(doc, table, "Table")
    caption = context[0] if context and table.caption_text(doc).strip() else ""
    md_head = [_md_row(names), "| " + " | ".join("---" for _ in names) + " |"]
    embed_head = [l for l in (heading_path, caption) if l]
    if not rows:
        return [("\n".join([*context, *md_head]),
                 "\n".join([*embed_head, " | ".join(names)]), None)]

    budget = max_tokens - count_tokens("\n".join(embed_head))
    groups: list[list[tuple[int, str, str]]] = [[]]
    used = 0
    for row in rows:
        cost = count_tokens(row[1]) + 1
        if groups[-1] and used + cost > budget:
            groups.append([])
            used = 0
        groups[-1].append(row)
        used += cost

    out = []
    for group in groups:
        pages = None
        if row_pages:
            pages = (_page_of(group[0][0], row_pages), _page_of(group[-1][0], row_pages))
        out.append(("\n".join([*context, *md_head, *(md for _, _, md in group)]),
                    "\n".join([*embed_head, *(kv for _, kv, _ in group)]),
                    pages))
    return out
