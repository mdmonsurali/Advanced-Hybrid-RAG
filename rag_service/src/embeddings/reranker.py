"""Cross-encoder reranker — BAAI/bge-reranker-v2-m3 on CPU.

Scores are passed through a sigmoid so they land in 0..1 regardless of the
model's raw logit scale, which keeps RAG_RERANK_MIN_SCORE meaningful.
"""

from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("rag_service.embeddings")

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "8"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))

_model = None
_lock = threading.Lock()


def load() -> None:
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE, max_length=RERANK_MAX_LENGTH)
        log.info("reranker ready: %s", RERANK_MODEL)


def is_ready() -> bool:
    return _model is not None


def score(query: str, texts: list[str]) -> list[float]:
    assert _model is not None, "reranker.load() not called"
    if not texts:
        return []
    import torch
    with _lock:
        scores = _model.predict([(query, t) for t in texts], batch_size=RERANK_BATCH,
                                activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
    return [float(s) for s in scores]
