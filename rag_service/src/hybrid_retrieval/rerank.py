"""STEP 7 — reranking.

Initial retrieval only gives candidates. A cross-encoder reads each
(query, chunk) pair together and scores how well the chunk answers that
specific query — this is where precision improves.

Candidates from every route (docs, Drive, web) are pooled; each is scored
against the query it was retrieved for (its sub-question, or the standalone
question). We keep the top RAG_RERANK_TOP_K above RAG_RERANK_MIN_SCORE, and
make sure every sub-question keeps its best chunk so decomposed questions
stay answerable.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict

from embeddings import reranker
from schemas import Evidence

log = logging.getLogger("rag_service.rerank")

RERANK_TOP_K = int(os.getenv("RAG_RERANK_TOP_K", "8"))
RERANK_MIN_SCORE = float(os.getenv("RAG_RERANK_MIN_SCORE", "0.15"))
RERANK_MAX_CANDIDATES = int(os.getenv("RAG_RERANK_MAX_CANDIDATES", "40"))


async def rerank(candidates: list[tuple[Evidence, str]],
                 top_k: int = RERANK_TOP_K) -> list[Evidence]:
    if not candidates:
        return []
    # Cap the CPU cost: keep the best first-stage candidates per query.
    by_query: dict[str, list[Evidence]] = defaultdict(list)
    for ev, q in candidates:
        by_query[q].append(ev)
    per_query_cap = max(5, RERANK_MAX_CANDIDATES // max(1, len(by_query)))
    for q in by_query:
        by_query[q] = sorted(by_query[q], key=lambda e: e.score, reverse=True)[:per_query_cap]

    async def _score(q: str, evs: list[Evidence]) -> list[tuple[Evidence, str]]:
        scores = await asyncio.to_thread(reranker.score, q, [e.text for e in evs])
        return [(e.model_copy(update={"score": s}), q) for e, s in zip(evs, scores)]

    scored = [pair for group in await asyncio.gather(
        *(_score(q, evs) for q, evs in by_query.items())) for pair in group]

    # Same chunk retrieved for several queries: keep its best score.
    best: dict[str, tuple[Evidence, str]] = {}
    for ev, q in scored:
        if ev.id not in best or ev.score > best[ev.id][0].score:
            best[ev.id] = (ev, q)

    ranked = sorted(best.values(), key=lambda p: p[0].score, reverse=True)
    kept = [ev for ev, _ in ranked if ev.score >= RERANK_MIN_SCORE][:top_k]
    kept_ids = {e.id for e in kept}

    # Every sub-question keeps its best chunk (if it is at all relevant).
    for q in by_query:
        top = next(((ev, qq) for ev, qq in ranked if qq == q), None)
        if top and top[0].id not in kept_ids and top[0].score >= RERANK_MIN_SCORE / 2:
            kept.append(top[0])
            kept_ids.add(top[0].id)

    log.info("[rerank] %d candidates -> %d kept (top=%.3f)", len(best), len(kept),
             ranked[0][0].score if ranked else 0.0)
    return sorted(kept, key=lambda e: e.score, reverse=True)
