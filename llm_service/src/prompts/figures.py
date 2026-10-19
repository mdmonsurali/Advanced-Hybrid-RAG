"""Prompt for STEP 1: describing a figure so its content can be retrieved."""

from __future__ import annotations

from prompts import registry

FIGURE_SYSTEM = registry.register("rag/figure-description", """\
You describe figures from documents for a search index. Look at the image and \
write a factual description someone could search for:
- what kind of figure it is (bar chart, line chart, table image, diagram, \
flowchart, photo, map, screenshot, ...) and what it is about,
- for charts: the axes, the series, the key values and the trend or comparison \
it shows,
- for diagrams: the main parts and how they connect,
- any legible labels, names, numbers and units.
Only state what is visible; do not guess. Write plain text, at most 120 words, \
in the language of the caption / context (English if there is none). If the \
image is only a logo, icon or decoration, answer with an empty string.""")


def figure_messages(image_b64: str, mime: str, caption: str, context: str) -> list[dict]:
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    if context:
        parts.append(f"Surrounding text: {context}")
    user = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        {"type": "text", "text": "\n".join(parts) or "No caption."},
    ]
    return registry.messages("rag/figure-description", [{"role": "user", "content": user}])
