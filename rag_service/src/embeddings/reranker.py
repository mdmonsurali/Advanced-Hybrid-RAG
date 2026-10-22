"""Reranker — Qwen/Qwen3-Reranker-0.6B on CPU.

Qwen3-Reranker is a causal LM that answers "yes" / "no" to "does this
document answer the query?" through its chat template; sentence-transformers'
CrossEncoder loads it with that template and scores each pair as the
yes-vs-no logit. Sequence-classification cross-encoders (e.g.
bge-reranker-v2-m3) load the same way.

Weights are loaded as float32 (RERANK_DTYPE): the checkpoint is bfloat16,
which most CPUs compute ~4x slower.

Scores are passed through a sigmoid so they land in 0..1 (for Qwen3: the
probability of "yes") regardless of the model's raw logit scale, which keeps
RAG_RERANK_MIN_SCORE meaningful.
"""

from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("rag_service.embeddings")

RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cpu")
RERANK_BATCH = int(os.getenv("RERANK_BATCH", "8"))
# Query + chunk (+ the reranker's prompt template) in tokens.
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "1024"))
RERANK_DTYPE = os.getenv("RERANK_DTYPE", "float32")

_model = None
_lock = threading.Lock()


def load() -> None:
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(
            RERANK_MODEL, device=RERANK_DEVICE, model_kwargs={"dtype": getattr(torch, RERANK_DTYPE)}
        )
        _model.max_seq_length = min(RERANK_MAX_LENGTH, _model.max_seq_length or RERANK_MAX_LENGTH)
        log.info("reranker ready: %s", RERANK_MODEL)


def is_ready() -> bool:
    return _model is not None


def score(query: str, texts: list[str]) -> list[float]:
    assert _model is not None, "reranker.load() not called"
    if not texts:
        return []
    import torch

    with _lock:
        scores = _model.predict(
            [(query, t) for t in texts],
            batch_size=RERANK_BATCH,
            activation_fn=torch.nn.Sigmoid(),
            show_progress_bar=False,
        )
    return [float(s) for s in scores]
