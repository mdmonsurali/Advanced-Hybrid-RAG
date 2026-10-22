"""Stitch tables that continue across a page break back into one table.

Docling detects tables page by page, so a table running over three pages
arrives as three TableItems, and the continuations usually have no header
row. Chunked as-is, those parts are rows of numbers without column names.

A table is treated as the continuation of the previous one when, in reading
order:
  - nothing but captions / footnotes, page headers / footers, or items that
    actually sit above the first table on its page (Docling sometimes orders
    a page title after the table) comes between them,
  - it starts on the page right after the previous one ends,
  - it has the same number of columns, and
  - it has no caption of its own (a captioned table is a new table).

Its rows are appended to the previous table, a repeated header row is
dropped, and the continuation is removed from the document. Run this on the
DoclingDocument before chunking; the returned row -> page map lets each row
group of a merged table cite the page its rows are on.
"""

from __future__ import annotations

import logging

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    RichTableCell,
    TableItem,
)

log = logging.getLogger("rag_service.chunking")


_IGNORABLE_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.FOOTNOTE}


def _pages(item) -> list[int]:
    return [p.page_no for p in (item.prov or [])]


def _top(doc: DoclingDocument, prov) -> float:
    """Distance of a box from the top of its page."""
    page = doc.pages.get(prov.page_no)
    return prov.bbox.to_top_left_origin(page_height=page.size.height).t if page else 0.0


def _ignorable(doc: DoclingDocument, item, head: TableItem) -> bool:
    """An item between two table parts that doesn't break the table."""
    if getattr(item, "label", None) in _IGNORABLE_LABELS:
        return True
    last = head.prov[-1] if head.prov else None
    return (
        last is not None
        and len(item.prov or []) == 1
        and item.prov[0].page_no == last.page_no
        and _top(doc, item.prov[0]) < _top(doc, head.prov[-1])
    )


def _row_texts(table: TableItem, row: int) -> list[str]:
    return [c.text.strip().lower() for c in table.data.grid[row]]


def _header_rows(table: TableItem) -> int:
    """Leading rows that hold column-header cells."""
    grid = table.data.grid
    n = 0
    while n < len(grid) and any(c.column_header for c in grid[n]):
        n += 1
    return n


def _rows_to_skip(head: TableItem, cont: TableItem) -> int:
    """How many leading rows of `cont` repeat the header of `head`."""
    if not cont.data.num_rows:
        return 0
    head_rows = max(1, _header_rows(head))
    for n in range(min(head_rows, cont.data.num_rows), 0, -1):
        if all(_row_texts(cont, i) == _row_texts(head, i) for i in range(n)):
            return n
    return 0


def _is_continuation(head: TableItem, cont: TableItem) -> bool:
    head_pages, cont_pages = _pages(head), _pages(cont)
    return (
        bool(head_pages)
        and bool(cont_pages)
        and min(cont_pages) == max(head_pages) + 1
        and cont.data.num_cols == head.data.num_cols
        and not cont.captions
        and not any(isinstance(c, RichTableCell) for c in cont.data.table_cells)
    )


def _append(head: TableItem, cont: TableItem) -> None:
    skip = _rows_to_skip(head, cont)
    offset = head.data.num_rows - skip
    for cell in cont.data.table_cells:
        if cell.start_row_offset_idx < skip:
            continue
        head.data.table_cells.append(
            cell.model_copy(
                update={
                    "start_row_offset_idx": cell.start_row_offset_idx + offset,
                    "end_row_offset_idx": cell.end_row_offset_idx + offset,
                    # A continuation's "header" that doesn't repeat the real one is a
                    # data row TableFormer misread.
                    "column_header": False,
                }
            )
        )
    head.data.num_rows += cont.data.num_rows - skip
    head.prov.extend(cont.prov)


RowPages = dict[str, list[tuple[int, int]]]


def merge_cross_page_tables(doc: DoclingDocument) -> RowPages:
    """Merge continuation tables in place. Returns, per merged table
    self_ref, (first row, page) for each page the table now spans."""
    merged: list[TableItem] = []
    starts: dict[int, tuple[TableItem, list[tuple[int, int]]]] = {}
    head: TableItem | None = None
    for item, _level in doc.iterate_items(with_groups=False):
        parent = item.parent.resolve(doc) if item.parent else None
        if isinstance(parent, (TableItem, PictureItem)):
            continue  # caption / footnote of a float
        if isinstance(item, TableItem):
            if head is not None and _is_continuation(head, item):
                _, pages = starts.setdefault(id(head), (head, [(0, min(_pages(head)))]))
                pages.append((head.data.num_rows, min(_pages(item))))
                _append(head, item)
                merged.append(item)
                continue
            head = item
        elif head is not None and not _ignorable(doc, item, head):
            head = None
    if merged:
        doc.delete_items(node_items=merged)  # renumbers the remaining refs
        log.info("[chunk] merged %d cross-page table continuation(s)", len(merged))
    return {table.self_ref: pages for table, pages in starts.values()}
