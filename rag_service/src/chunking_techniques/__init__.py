"""Chunking strategies (STEP 1).

    hybrid.py  structure-aware chunking of DoclingDocuments (OCR'd uploads)
    tables.py  keeps tables together with their caption + explanation
    text.py    sentence-aware chunking of plain text (Drive files, web pages)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RawChunk:
    index: int
    text: str                       # what the LLM sees (context lines + body)
    embed_text: str                 # what gets embedded (heading path + body)
    headings: list[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_type: str = "text"        # text | table | list
