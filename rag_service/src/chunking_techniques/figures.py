"""Figure chunks: make charts, diagrams and photos searchable.

Without this a figure only contributes its caption, so a question answered
by a chart ("which region grew fastest?") finds nothing. Each picture becomes
one chunk with:
  - the caption,
  - a description of what the image shows (generated at ingest by
    ingestion/figures.py with the vision-capable LLM, or Docling's own
    picture description if the OCR step produced one),
  - text Docling found inside the picture (axis labels, diagram boxes), and
  - the closest preceding paragraph in the same section.

Pictures with none of these (logos, decorations) produce no chunk.
"""

from __future__ import annotations

from docling_core.types.doc import DoclingDocument, PictureItem, TextItem

from chunking_techniques.tables import context_lines

_MAX_INNER_CHARS = 800


def _inner_text(doc: DoclingDocument, picture: PictureItem) -> str:
    """Text items nested inside the picture (not its caption / footnotes)."""
    skip = {r.cref for r in [*picture.captions, *picture.footnotes]}
    texts = [
        item.text.strip()
        for item, _ in doc.iterate_items(root=picture, traverse_pictures=True)
        if isinstance(item, TextItem) and item.self_ref not in skip and item.text.strip()
    ]
    joined = " · ".join(texts)
    return joined if len(joined) <= _MAX_INNER_CHARS else joined[:_MAX_INNER_CHARS] + "…"


def picture_description(picture: PictureItem) -> str:
    """Description Docling attached during conversion, if any."""
    meta = picture.meta
    if meta is not None and meta.description is not None:
        return meta.description.text.strip()
    return ""


def figure_chunk(
    doc: DoclingDocument, picture: PictureItem, heading_path: str, description: str
) -> tuple[str, str] | None:
    """(text, embed_text) for one picture, or None if it carries nothing."""
    description = description or picture_description(picture)
    inner = _inner_text(doc, picture)
    context = context_lines(doc, picture, "Figure")
    has_caption = bool(picture.caption_text(doc).strip())
    if not (description or inner or has_caption):
        return None
    lines = list(context[:1] if has_caption else [])
    if description:
        lines.append(f"Figure content: {description}")
    if inner:
        lines.append(f"Text in figure: {inner}")
    lines.extend(context[1:] if has_caption else context)
    text = "\n".join(lines)
    return text, f"{heading_path}\n{text}" if heading_path else text
