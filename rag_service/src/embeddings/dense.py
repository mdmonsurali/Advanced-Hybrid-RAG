"""Dense embeddings — Qwen/Qwen3-Embedding-0.6B (multilingual, 1024-d,
instruction-aware) on CPU.

Queries and documents are embedded differently: queries get the model's
"query" instruction prompt, documents (chunks, HyDE passages) none. Models
without saved prompts (e.g. bge-m3) embed both the same way.

Weights are loaded as float32 (EMBED_DTYPE): Qwen3 checkpoints are stored
in bfloat16, which most CPUs compute ~4x slower than float32.

Loaded once at startup; encode() is blocking, so async callers wrap it in
asyncio.to_thread. A lock serializes encodes so concurrent requests don't
oversubscribe the CPU.
"""

from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("rag_service.embeddings")

EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))
EMBED_MAX_SEQ = int(os.getenv("EMBED_MAX_SEQ", "1024"))
EMBED_DTYPE = os.getenv("EMBED_DTYPE", "float32")

_model = None
_lock = threading.Lock()


def load() -> None:
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE,
                                     model_kwargs={"dtype": getattr(torch, EMBED_DTYPE)})
        _model.max_seq_length = min(EMBED_MAX_SEQ, _model.max_seq_length or EMBED_MAX_SEQ)
        log.info("dense model ready: %s (dim=%d, query prompt=%s)", EMBED_MODEL, dim(),
                 "query" in (_model.prompts or {}))


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


def _encode(texts: list[str], prompt_name: str | None) -> list[list[float]]:
    assert _model is not None, "dense.load() not called"
    if not texts:
        return []
    if prompt_name not in (_model.prompts or {}):
        prompt_name = None
    with _lock:
        vecs = _model.encode(texts, batch_size=EMBED_BATCH, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False,
                             prompt_name=prompt_name)
    return vecs.tolist()


def embed(texts: list[str]) -> list[list[float]]:
    """Documents: chunks at ingest, HyDE passages at query time."""
    return _encode(texts, "document")


def embed_query(texts: list[str]) -> list[list[float]]:
    """Search queries (with the model's retrieval instruction)."""
    return _encode(texts, "query")
