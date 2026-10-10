"""Sentence-aware chunking for plain text (Google Drive files, web pages).

Splits on paragraph boundaries first, then sentences, and packs pieces up to
RAG_TEXT_CHUNK_CHARS with a small overlap so an idea cut at a boundary still
appears whole in one of the two chunks.
"""

from __future__ import annotations

import os
import re

TEXT_CHUNK_CHARS = int(os.getenv("RAG_TEXT_CHUNK_CHARS", "1500"))
TEXT_CHUNK_OVERLAP = int(os.getenv("RAG_TEXT_CHUNK_OVERLAP", "200"))

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _pieces(text: str, max_chars: int) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            out.append(para)
            continue
        for sent in _SENTENCE_RE.split(para):
            sent = sent.strip()
            while len(sent) > max_chars:          # pathological run-on text
                out.append(sent[:max_chars])
                sent = sent[max_chars:]
            if sent:
                out.append(sent)
    return out


def chunk_text(text: str, max_chars: int = TEXT_CHUNK_CHARS,
               overlap: int = TEXT_CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in _pieces(text, max_chars):
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # Start the overlap at a word boundary.
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            current = f"{tail} {piece}".strip()
        else:
            current = f"{current}\n{piece}".strip() if current else piece
    if current:
        chunks.append(current)
    return chunks
