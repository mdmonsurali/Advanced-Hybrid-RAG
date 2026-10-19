"""STEP 8 — context compression.

Goal: the smallest useful context that fits the window.
  1. drop weak chunks (far below the best reranker score)
  2. remove near-duplicates (word-shingle Jaccard)
  3. merge adjacent chunks from the same document page
  4. fit the token budget; long chunks that don't fit are extractively
     compressed by the LLM (citation id kept), the rest dropped
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

from llm_client.client import LLM_CONTEXT_LEN, chat_json, count_tokens
from prompts import registry
from prompts.answer import extract_messages
from schemas import CompressResponse, Evidence, ExtractOutput, QueryPlan

log = logging.getLogger("llm_service.compression")

ANSWER_TOKENS = int(os.getenv("RAG_ANSWER_TOKENS", "1024"))
RELATIVE_MIN_SCORE = float(os.getenv("COMPRESS_RELATIVE_MIN_SCORE", "0.25"))
DUP_JACCARD = float(os.getenv("COMPRESS_DUP_JACCARD", "0.9"))
LONG_CHUNK_TOKENS = int(os.getenv("COMPRESS_LONG_CHUNK_TOKENS", "350"))
# Room for the question, format constraints and chat-template tokens.
PROMPT_OVERHEAD_TOKENS = 384

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _drop_weak(evidence: list[Evidence]) -> list[Evidence]:
    if not evidence:
        return evidence
    top = max(e.score for e in evidence)
    if top <= 0:
        return evidence
    return [e for e in evidence if e.score >= top * RELATIVE_MIN_SCORE]


def _dedupe(evidence: list[Evidence]) -> list[Evidence]:
    kept: list[tuple[Evidence, set]] = []
    for ev in evidence:                       # already score-sorted
        sh = _shingles(ev.text)
        if any(_jaccard(sh, other) >= DUP_JACCARD for _, other in kept):
            continue
        kept.append((ev, sh))
    return [ev for ev, _ in kept]


def _drop_repeated_head(prev: str, text: str) -> str:
    """Leading lines of `text` already in `prev` (the caption, explanation and
    header row every row group of a table repeats)."""
    seen = {l.strip() for l in prev.splitlines()}
    lines = text.strip().splitlines()
    i = 0
    while i < len(lines) - 1 and lines[i].strip() in seen:
        i += 1
    return "\n".join(lines[i:])


def _merge_adjacent(evidence: list[Evidence]) -> tuple[list[Evidence], int]:
    """Merge chunks of the same doc + page whose chunk_index is consecutive,
    keeping the best score and the earlier chunk's id for citations."""
    merged = 0
    by_key: dict[tuple, Evidence] = {}
    out: list[Evidence] = []
    ordered = sorted(
        evidence,
        key=lambda e: (e.doc_id or e.id, e.page or 0,
                       e.chunk_index if e.chunk_index is not None else 10**9),
    )
    for ev in ordered:
        key = (ev.doc_id, ev.page)
        prev = by_key.get(key)
        if (ev.doc_id and prev is not None and prev.chunk_index is not None
                and ev.chunk_index is not None and ev.chunk_index == prev.chunk_index + 1):
            prev.text = f"{prev.text.rstrip()}\n{_drop_repeated_head(prev.text, ev.text)}"
            prev.score = max(prev.score, ev.score)
            prev.chunk_index = ev.chunk_index
            merged += 1
            continue
        copy = ev.model_copy()
        by_key[key] = copy
        out.append(copy)
    out.sort(key=lambda e: e.score, reverse=True)
    return out, merged


async def _extract(question: str, ev: Evidence) -> Evidence | None:
    try:
        res = await chat_json(extract_messages(question, ev.text), ExtractOutput,
                              max_tokens=512)
    except Exception as e:
        log.warning("[compress] extraction failed for %s: %s", ev.id, e)
        return None
    text = res.relevant_text.strip()
    if not text:
        return None
    return ev.model_copy(update={"text": text})


async def compress(plan: QueryPlan, evidence: list[Evidence]) -> CompressResponse:
    before = len(evidence)
    ranked = sorted(evidence, key=lambda e: e.score, reverse=True)
    ranked = _dedupe(_drop_weak(ranked))
    ranked, merged = _merge_adjacent(ranked)
    dropped = before - len(ranked) - merged

    system_tokens = count_tokens(registry.text("rag/answer", language=plan.language)) + count_tokens(plan.standalone_question)
    budget = LLM_CONTEXT_LEN - ANSWER_TOKENS - PROMPT_OVERHEAD_TOKENS - system_tokens

    # First pass: keep everything that fits; queue long chunks that don't.
    kept: list[Evidence] = []
    used = 0
    overflow: list[Evidence] = []
    for ev in ranked:
        n = count_tokens(ev.text) + 24          # header line per block
        if used + n <= budget:
            kept.append(ev)
            used += n
        elif n > LONG_CHUNK_TOKENS:
            overflow.append(ev)
        else:
            dropped += 1

    # Second pass: extractive compression of the long overflow chunks.
    extracted = 0
    if overflow:
        results = await asyncio.gather(*(_extract(plan.standalone_question, ev)
                                         for ev in overflow))
        for res in results:
            if res is None:
                dropped += 1
                continue
            n = count_tokens(res.text) + 24
            if used + n <= budget:
                kept.append(res)
                used += n
                extracted += 1
            else:
                dropped += 1

    kept.sort(key=lambda e: e.score, reverse=True)
    log.info("[compress] %d -> %d evidence (dropped=%d merged=%d extracted=%d, %d tokens)",
             before, len(kept), dropped, merged, extracted, used)
    return CompressResponse(evidence=kept, dropped=max(0, dropped), merged=merged,
                            extracted=extracted, tokens=used)
