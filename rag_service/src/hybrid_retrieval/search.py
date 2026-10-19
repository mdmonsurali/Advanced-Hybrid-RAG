"""STEP 6 — hybrid retrieval over the uploaded documents.

Vector search finds semantic matches, BM25 catches exact terms (codes,
names, numbers); Qdrant runs both as prefetches and fuses them with RRF.

Per search query (each sub-question, or the rewritten question):
  dense  : embedding of the query (with the embedder's query instruction),
           plus the HyDE passage (embedded as a document) for the main query
  sparse : BM25 of the query (+ expansions for the main query)
  filter : ACL (owner OR shared) + doc_type / date filters from STEP 3,
           filters dropped and K doubled on a `broaden` retry
"""

from __future__ import annotations

import asyncio
import logging
import os

from qdrant_client import models

from embeddings import dense, sparse
from identity import Identity
from schemas import Evidence, QueryPlan
from vectordb import qdrant_store

log = logging.getLogger("rag_service.retrieval")

RETRIEVE_K = int(os.getenv("RAG_RETRIEVE_K", "30"))


def _acl_filter(identity: Identity) -> models.Filter:
    return models.Filter(should=[
        models.FieldCondition(key="owner_id", match=models.MatchValue(value=str(identity.id))),
        models.FieldCondition(key="access_level", match=models.MatchValue(value="shared")),
    ])


def build_filter(plan: QueryPlan, identity: Identity, broaden: bool) -> models.Filter:
    must: list = [_acl_filter(identity)]
    if not broaden:
        f = plan.filters
        if f.doc_type:
            must.append(models.FieldCondition(key="doc_type",
                                              match=models.MatchValue(value=f.doc_type.lower())))
        if f.date_from or f.date_to:
            must.append(models.FieldCondition(key="timestamp", range=models.DatetimeRange(
                gte=f.date_from or None, lte=f.date_to or None)))
    return models.Filter(must=must)


def search_queries(plan: QueryPlan) -> list[tuple[str, bool]]:
    """(query, is_main) pairs: sub-questions when decomposed, else the
    rewritten question."""
    if plan.sub_questions:
        return [(q, False) for q in plan.sub_questions] + [(plan.rewritten or plan.standalone_question, True)]
    return [(plan.rewritten or plan.standalone_question, True)]


def _evidence(point: models.ScoredPoint, identity: Identity) -> Evidence:
    p = point.payload or {}
    return Evidence(
        id=str(point.id),
        route="docs",
        text=p.get("text", ""),
        score=float(point.score or 0.0),
        source=p.get("source", ""),
        doc_id=p.get("doc_id"),
        page=p.get("page_start"),
        date=p.get("timestamp"),
        headings=p.get("headings") or [],
        chunk_index=p.get("chunk_index"),
        owned=p.get("owner_id") == str(identity.id),
    )


async def search_docs(plan: QueryPlan, identity: Identity,
                      broaden: bool = False) -> list[tuple[Evidence, str]]:
    """Returns (evidence, query it was retrieved for) pairs, deduplicated by
    chunk id — the query is what the reranker scores the chunk against."""
    flt = build_filter(plan, identity, broaden)
    limit = RETRIEVE_K * (2 if broaden else 1)
    queries = search_queries(plan)

    async def _one(query: str, is_main: bool) -> list[tuple[Evidence, str]]:
        sparse_input = " ".join([query, *plan.expansions]) if is_main else query
        dense_vecs = await asyncio.to_thread(dense.embed_query, [query])
        if is_main and plan.hyde_passage:
            # A HyDE passage is a hypothetical answer: embed it like a document.
            dense_vecs += await asyncio.to_thread(dense.embed, [plan.hyde_passage])
        sparse_vec = await asyncio.to_thread(sparse.embed_query, sparse_input)
        points = await qdrant_store.hybrid_query(dense_vecs, sparse_vec, flt, limit)
        rerank_query = plan.standalone_question if is_main else query
        return [(_evidence(p, identity), rerank_query) for p in points]

    results = await asyncio.gather(*(_one(q, main) for q, main in queries))
    best: dict[str, tuple[Evidence, str]] = {}
    for hits in results:
        for ev, q in hits:
            if ev.id not in best or ev.score > best[ev.id][0].score:
                best[ev.id] = (ev, q)
    log.info("[retrieve] %d queries -> %d unique chunks (broaden=%s)",
             len(queries), len(best), broaden)
    return list(best.values())
