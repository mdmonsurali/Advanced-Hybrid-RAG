"""Dense embeddings — BAAI/bge-m3 (multilingual, 1024-d) on CPU.

Loaded once at startup; encode() is blocking, so async callers wrap it in
asyncio.to_thread. A lock serializes encodes so concurrent requests don't
oversubscribe the CPU.
"""

from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("rag_service.embeddings")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))
EMBED_MAX_SEQ = int(os.getenv("EMBED_MAX_SEQ", "1024"))

_model = None
_lock = threading.Lock()


def load() -> None:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
        _model.max_seq_length = min(EMBED_MAX_SEQ, _model.max_seq_length or EMBED_MAX_SEQ)
        log.info("dense model ready: %s (dim=%d)", EMBED_MODEL, dim())


def is_ready() -> bool:
    return _model is not None


def dim() -> int:
    assert _model is not None, "dense.load() not called"
    getter = (getattr(_model, "get_embedding_dimension", None)
              or _model.get_sentence_embedding_dimension)
    return int(getter())


def tokenizer():
    """The embedder's HF tokenizer (used by the chunker to size chunks)."""
    assert _model is not None, "dense.load() not called"
    return _model.tokenizer


def embed(texts: list[str]) -> list[list[float]]:
    assert _model is not None, "dense.load() not called"
    if not texts:
        return []
    with _lock:
        vecs = _model.encode(texts, batch_size=EMBED_BATCH, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
    return vecs.tolist()
