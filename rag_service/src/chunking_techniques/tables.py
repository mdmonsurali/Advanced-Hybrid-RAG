"""Keep tables with their explanations.

A table on its own ("| 4.1 | 12 | yes |") is hard to retrieve and easy to
misread. For every table chunk we attach:
  - the table caption, and
  - the closest preceding paragraph in the same section (the sentence that
    usually says what the table shows),
so the chunk carries its own explanation.
"""

from __future__ import annotations

from docling_core.types.doc import DocItemLabel, DoclingDocument, TableItem, TextItem

_PARAGRAPH_LABELS = {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH}
_SECTION_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
# Look back at most this many items for the explanatory paragraph.
_MAX_LOOKBACK = 4
_MAX_CONTEXT_CHARS = 600


def _preceding_paragraph(doc: DoclingDocument, table: TableItem) -> str | None:
    previous: list = []
    for item, _level in doc.iterate_items(with_groups=False):
        if item.self_ref == table.self_ref:
            break
        previous.append(item)
    for item in reversed(previous[-_MAX_LOOKBACK:]):
        label = getattr(item, "label", None)
        if label in _SECTION_LABELS:
            return None                 # never borrow text from another section
        if isinstance(item, TextItem) and label in _PARAGRAPH_LABELS and item.text.strip():
            text = item.text.strip()
            return text if len(text) <= _MAX_CONTEXT_CHARS else text[:_MAX_CONTEXT_CHARS] + "…"
    return None


def table_context(doc: DoclingDocument, table: TableItem) -> list[str]:
    lines: list[str] = []
    caption = table.caption_text(doc).strip()
    if caption:
        lines.append(f"Table: {caption}")
    paragraph = _preceding_paragraph(doc, table)
    if paragraph and paragraph != caption:
        lines.append(paragraph)
    return lines
