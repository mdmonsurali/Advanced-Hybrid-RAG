"""Sparse BM25 vectors — fastembed `Qdrant/bm25`.

Documents get term-frequency weights; queries get plain term ids. Qdrant
applies the IDF part server-side (collection uses `modifier=idf`), so the
scores stay correct as the corpus grows.
"""

from __future__ import annotations

import logging
import os
import threading

from qdrant_client import models

log = logging.getLogger("rag_service.embeddings")

BM25_MODEL = os.getenv("BM25_MODEL", "Qdrant/bm25")
# Snowball stemmer language for BM25 tokens (e.g. english, german, french).
BM25_LANGUAGE = os.getenv("BM25_LANGUAGE", "english")

_model = None
_lock = threading.Lock()


def load() -> None:
    global _model
    if _model is None:
        from fastembed import SparseTextEmbedding

        _model = SparseTextEmbedding(BM25_MODEL, language=BM25_LANGUAGE)
        log.info("sparse model ready: %s (%s)", BM25_MODEL, BM25_LANGUAGE)


def is_ready() -> bool:
    return _model is not None


def _to_qdrant(emb) -> models.SparseVector:
    return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


def embed_documents(texts: list[str]) -> list[models.SparseVector]:
    assert _model is not None, "sparse.load() not called"
    with _lock:
        return [_to_qdrant(e) for e in _model.embed(texts)]


def embed_query(text: str) -> models.SparseVector:
    assert _model is not None, "sparse.load() not called"
    with _lock:
        return _to_qdrant(next(iter(_model.query_embed(text))))
