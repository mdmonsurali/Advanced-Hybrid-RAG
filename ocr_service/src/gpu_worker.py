"""Docling on CUDA in a child process, switched on and off by the GPU swap.

The orchestrator's gpu_swap gives the GPU to OCR or to the LLM, never both.
When OCR gets it, POST /gpu/enable spawns the worker and loads the Docling
models on CUDA; POST /gpu/disable lets the worker exit. Exiting the process
(not just dropping the models) is what returns every byte of VRAM, CUDA
context included, so SGLang can start again with its full memory fraction.

Conversions run under an `ocr` lease taken from the orchestrator
(`ocr_lease`), renewed while they run. Any failure of the GPU path itself
raises GpuUnavailable so docling_engine falls back to the CPU converter.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
from collections.abc import AsyncIterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from docling_core.types.doc import DoclingDocument

log = logging.getLogger("ocr_service.gpu")

ORCHESTRATOR_SERVICE_URL = os.getenv("ORCHESTRATOR_SERVICE_URL", "http://orchestrator_service:8004")
# A lease request waits for the swap (LLM stop + model load), or for chat to
# finish first, so it can take a while.
GPU_ACQUIRE_TIMEOUT = float(os.getenv("GPU_ACQUIRE_TIMEOUT", "1800"))
GPU_LEASE_TTL = float(os.getenv("GPU_LEASE_TTL_SEC", "120"))
OCR_GPU_WORKERS = int(os.getenv("OCR_GPU_WORKERS", "1"))


class GpuUnavailable(RuntimeError):
    pass


# ── child process ──────────────────────────────────────────────────────────

_child_converter = None


def _child_init() -> None:
    global _child_converter
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s[gpu-worker]: %(message)s"
    )
    import docling_engine

    _child_converter = docling_engine.build_converter(device="cuda")
    logging.getLogger("ocr_service.gpu").info("[gpu-worker] Docling models loaded on CUDA")


def _child_ping() -> str:
    import torch

    return torch.cuda.get_device_name(0)


def _child_convert(path: str) -> dict:
    import docling_engine

    if _child_converter is None:  # set by _child_init when the worker process starts
        raise RuntimeError("GPU worker not initialized")
    return docling_engine.convert_with(_child_converter, Path(path)).export_to_dict()


# ── parent side ────────────────────────────────────────────────────────────

_pool: ProcessPoolExecutor | None = None
_lock = asyncio.Lock()


def enabled() -> bool:
    return _pool is not None


async def enable() -> dict:
    """Spawn the worker(s) and wait until the models are on the GPU."""
    global _pool
    async with _lock:
        if _pool is None:
            pool = ProcessPoolExecutor(
                max_workers=max(1, OCR_GPU_WORKERS),
                mp_context=mp.get_context("spawn"),
                initializer=_child_init,
            )
            loop = asyncio.get_running_loop()
            try:
                names = await asyncio.gather(
                    *(
                        loop.run_in_executor(pool, _child_ping)
                        for _ in range(max(1, OCR_GPU_WORKERS))
                    )
                )
            except Exception:
                await asyncio.to_thread(pool.shutdown, True, cancel_futures=True)
                raise
            _pool = pool
            log.info("[gpu] worker ready on %s", names[0])
    return {"enabled": True}


async def disable() -> dict:
    """Let the worker(s) exit; returns once their VRAM is released."""
    global _pool
    async with _lock:
        pool, _pool = _pool, None
        if pool is not None:
            await asyncio.to_thread(pool.shutdown, True, cancel_futures=True)
            log.info("[gpu] worker stopped, GPU released")
    return {"enabled": False}


async def convert(path: Path) -> DoclingDocument:
    pool = _pool
    if pool is None:
        raise GpuUnavailable("GPU worker is not running")
    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(pool, _child_convert, str(path))
    except BrokenProcessPool as e:
        await disable()
        raise GpuUnavailable(f"GPU worker crashed: {e}") from e
    return DoclingDocument.model_validate(data)


@asynccontextmanager
async def ocr_lease() -> AsyncIterator[str]:
    """Hold an `ocr` GPU lease from the orchestrator while the body runs."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(GPU_ACQUIRE_TIMEOUT, connect=10)
        ) as client:
            resp = await client.post(
                f"{ORCHESTRATOR_SERVICE_URL}/gpu/acquire/ocr", params={"ttl": GPU_LEASE_TTL}
            )
        resp.raise_for_status()
        lease_id = resp.json()["lease_id"]
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise GpuUnavailable(f"no GPU lease from the orchestrator: {e}") from e

    async def _renew() -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            while True:
                await asyncio.sleep(GPU_LEASE_TTL / 3)
                try:
                    await client.post(
                        f"{ORCHESTRATOR_SERVICE_URL}/gpu/renew/{lease_id}",
                        params={"ttl": GPU_LEASE_TTL},
                    )
                except httpx.HTTPError as e:
                    log.warning("[gpu] lease renew failed: %s", e)

    renewer = asyncio.create_task(_renew()) if lease_id != "noop" else None
    try:
        yield lease_id
    finally:
        if renewer:
            renewer.cancel()
        if lease_id != "noop":
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(f"{ORCHESTRATOR_SERVICE_URL}/gpu/release/{lease_id}")
            except httpx.HTTPError as e:
                log.warning("[gpu] lease release failed (it expires on its own): %s", e)
