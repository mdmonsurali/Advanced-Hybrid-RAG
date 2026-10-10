"""Prompt for STEP 2: document-level metadata attached to every chunk."""

from __future__ import annotations

DOC_METADATA_SYSTEM = """\
You catalogue documents. From the file name and the opening text, return:
- doc_type: one or two lowercase words, e.g. invoice, contract, policy, \
manual, report, form, letter, presentation, spreadsheet, paper, other.
- topic: a short noun phrase (max 8 words) naming what the document is about.
- summary: 2-3 sentences describing the content, in the document's language.
- language: ISO 639-1 code of the document's main language."""


def doc_metadata_messages(name: str, text: str) -> list[dict]:
    return [
        {"role": "system", "content": DOC_METADATA_SYSTEM},
        {"role": "user", "content": f"File name: {name}\n\nOpening text:\n{text}"},
    ]
