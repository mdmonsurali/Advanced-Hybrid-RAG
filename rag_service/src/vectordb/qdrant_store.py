"""Qdrant collection: one point per chunk with two named vectors —
`dense` (the EMBED_MODEL embedder, cosine) and `bm25` (sparse, IDF applied by
Qdrant) — and the chunk metadata as payload. Payload indexes back the ACL /
doc_type / time filters used at query time. The collection records which
embedder filled it, so switching models never mixes vectors from two models."""

from __future__ import annotations

import logging
import os
import uuid as _uuid
from typing import Optional

from qdrant_client import AsyncQdrantClient, models

log = logging.getLogger("rag_service.qdrant")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks")

DENSE = "dense"
SPARSE = "bm25"

_KEYWORD_FIELDS = ("owner_id", "access_level", "doc_id", "doc_type", "chunk_type", "language")

client: Optional[AsyncQdrantClient] = None


def init_client(location: str | None = None) -> None:
    """`location=":memory:"` gives an in-process store (tests)."""
    global client
    client = (AsyncQdrantClient(location=location) if location
              else AsyncQdrantClient(url=QDRANT_URL))


async def ensure_collection(dense_dim: int, embed_model: str) -> None:
    assert client is not None
    if await client.collection_exists(QDRANT_COLLECTION):
        info = await client.get_collection(QDRANT_COLLECTION)
        size = info.config.params.vectors[DENSE].size
        if size != dense_dim:
            raise RuntimeError(
                f"collection {QDRANT_COLLECTION} has dense size {size}, model gives "
                f"{dense_dim}; change QDRANT_COLLECTION or drop the collection")
        filled_by = (info.config.metadata or {}).get("embed_model")
        if filled_by and filled_by != embed_model:
            raise RuntimeError(
                f"collection {QDRANT_COLLECTION} holds {filled_by} vectors, EMBED_MODEL is "
                f"{embed_model}; change QDRANT_COLLECTION or drop the collection")
        return
    await client.create_collection(
        QDRANT_COLLECTION,
        vectors_config={DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        metadata={"embed_model": embed_model},
    )
    for field in _KEYWORD_FIELDS:
        await client.create_payload_index(QDRANT_COLLECTION, field,
                                          models.PayloadSchemaType.KEYWORD)
    await client.create_payload_index(QDRANT_COLLECTION, "timestamp",
                                      models.PayloadSchemaType.DATETIME)
    log.info("created qdrant collection %s (%s dense=%d + bm25)", QDRANT_COLLECTION,
             embed_model, dense_dim)


async def health() -> None:
    assert client is not None
    await client.get_collections()


def point_id(doc_id: str, chunk_index: int) -> str:
    return str(_uuid.uuid5(_uuid.NAMESPACE_URL, f"{doc_id}#{chunk_index}"))


def _doc_filter(doc_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="doc_id",
                                                     match=models.MatchValue(value=doc_id))])


async def delete_document(doc_id: str) -> None:
    assert client is not None
    await client.delete(QDRANT_COLLECTION,
                        points_selector=models.FilterSelector(filter=_doc_filter(doc_id)))


async def upsert_chunks(doc_id: str, dense: list[list[float]],
                        sparse: list[models.SparseVector], payloads: list[dict],
                        batch: int = 64) -> None:
    assert client is not None
    points = [
        models.PointStruct(
            id=point_id(doc_id, p["chunk_index"]),
            vector={DENSE: d, SPARSE: s},
            payload=p,
        )
        for d, s, p in zip(dense, sparse, payloads)
    ]
    for i in range(0, len(points), batch):
        await client.upsert(QDRANT_COLLECTION, points=points[i:i + batch], wait=True)


async def set_access(doc_id: str, access_level: str) -> None:
    assert client is not None
    await client.set_payload(QDRANT_COLLECTION, payload={"access_level": access_level},
                             points=_doc_filter(doc_id), wait=True)


async def count_document(doc_id: str) -> int:
    assert client is not None
    res = await client.count(QDRANT_COLLECTION, count_filter=_doc_filter(doc_id), exact=True)
    return res.count


async def hybrid_query(dense_vectors: list[list[float]], sparse_vector: models.SparseVector,
                       query_filter: models.Filter, limit: int) -> list[models.ScoredPoint]:
    """Dense prefetch per vector (e.g. rewritten query + HyDE passage) plus
    one BM25 prefetch, fused with Reciprocal Rank Fusion."""
    assert client is not None
    prefetch = [models.Prefetch(query=v, using=DENSE, filter=query_filter, limit=limit)
                for v in dense_vectors]
    if sparse_vector.indices:
        prefetch.append(models.Prefetch(query=sparse_vector, using=SPARSE,
                                        filter=query_filter, limit=limit))
    res = await client.query_points(
        QDRANT_COLLECTION,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return res.points
