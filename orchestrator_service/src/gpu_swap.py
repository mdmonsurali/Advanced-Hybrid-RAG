"""GPU swap: OCR (Docling on CUDA) and the LLM (SGLang) take turns on the GPU.

The 6 GB card can't hold both, so exactly one tenant owns it at a time:

    llm   the llm_service container (stopped / started through the docker socket)
    ocr   ocr_service's Docling GPU worker process (POST /gpu/enable | /gpu/disable;
          ocr_service itself stays up because it also serves the document store)

Callers take a lease on the tenant they need (`acquire` / `release`, or the
`lease()` context manager in-process). Policy, tuned for a ~6 min SGLang cold
start:

  * llm -> ocr  as soon as OCR is waiting and no LLM lease is held. New LLM
                requests queue behind a waiting OCR so ingestion can't starve.
  * ocr -> llm  once OCR holds no lease and has been idle GPU_SWAP_IDLE_SEC,
                whether or not anyone is waiting, so chat is warm by default.
                Back-to-back uploads therefore share one OCR window.
  * After every OCR window rag_service re-indexes the documents it indexed
    without the LLM (heuristic metadata, undescribed figures).

Leases expire after their TTL unless renewed, so a crashed caller can't pin
the GPU. Only container names from env reach the docker client.

When GPU_SWAP=false every operation is a no-op and leases are granted at once.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Optional

import docker
import httpx
from docker.errors import APIError, DockerException, NotFound

log = logging.getLogger("orchestrator.gpu_swap")

GPU_SWAP_ENABLED = os.environ.get("GPU_SWAP", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
LLM_CONTAINER_NAME = os.environ.get("LLM_CONTAINER_NAME", "llm_service")
OCR_SERVICE_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr_service:8001")
RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://rag_service:8003")
HEALTH_TIMEOUT_SEC = int(os.environ.get("SWAP_HEALTH_TIMEOUT_SEC", "900"))
IDLE_SEC = float(os.environ.get("GPU_SWAP_IDLE_SEC", "30"))
LEASE_TTL_SEC = float(os.environ.get("GPU_LEASE_TTL_SEC", "120"))
OCR_ENABLE_TIMEOUT_SEC = float(os.environ.get("OCR_GPU_ENABLE_TIMEOUT_SEC", "300"))
RETRY_SEC = 15

TENANTS = ("llm", "ocr")

EmitStatus = Callable[[str, dict], Awaitable[None]]

# Docker SDK is synchronous; calls go through to_thread so the event loop
# never blocks on a stop/start.
_client: Optional[docker.DockerClient] = None


def _get_client() -> docker.DockerClient:
    global _client
    if _client is None:
        _client = docker.from_env()
    return _client


# ── Low-level state probes ────

def _container_state(name: str) -> dict:
    """Return {"running": bool, "health": str|None} or {"exists": False}."""
    try:
        c = _get_client().containers.get(name)
    except NotFound:
        return {"exists": False, "running": False, "health": None}
    c.reload()
    state = c.attrs.get("State") or {}
    health = (state.get("Health") or {}).get("Status")
    return {"exists": True, "running": state.get("Running", False),
            "health": health, "status": state.get("Status")}


async def state(name: str) -> dict:
    return await asyncio.to_thread(_container_state, name)


async def is_running(name: str) -> bool:
    return (await state(name)).get("running", False)


async def is_healthy(name: str) -> bool:
    s = await state(name)
    return s.get("running", False) and s.get("health") == "healthy"


# ── Lifecycle operations ────

def _stop_sync(name: str, timeout: int) -> None:
    c = _get_client().containers.get(name)
    c.stop(timeout=timeout)


def _start_sync(name: str) -> None:
    c = _get_client().containers.get(name)
    c.start()


async def stop(name: str, timeout: int = 15) -> None:
    log.info("[gpu_swap] stop  %s  (graceful timeout=%ds)", name, timeout)
    try:
        await asyncio.to_thread(_stop_sync, name, timeout)
        log.info("[gpu_swap] stop  %s  OK", name)
    except NotFound:
        log.warning("[gpu_swap] stop  %s  NOT FOUND (treating as already stopped)", name)
    except APIError as e:
        if "is not running" in str(e).lower():
            log.info("[gpu_swap] stop  %s  already not running", name)
            return
        log.error("[gpu_swap] stop  %s  ERROR: %s", name, e)
        raise


async def start(name: str) -> None:
    log.info("[gpu_swap] start %s", name)
    try:
        await asyncio.to_thread(_start_sync, name)
        log.info("[gpu_swap] start %s  OK (container running, waiting for app healthcheck)", name)
    except APIError as e:
        if "already started" in str(e).lower() or "is already" in str(e).lower():
            log.info("[gpu_swap] start %s  already running", name)
            return
        log.error("[gpu_swap] start %s  ERROR: %s", name, e)
        raise


async def wait_healthy(name: str, timeout_s: int = HEALTH_TIMEOUT_SEC,
                       emit: Optional[EmitStatus] = None) -> None:
    """Poll Inspect.State.Health.Status until 'healthy' or the deadline.

    If the container has no healthcheck, fall back to status=='running'.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    last_emit = 0.0
    while True:
        s = await state(name)
        if not s.get("exists", True):
            raise RuntimeError(f"container {name} disappeared while waiting")
        h = s.get("health")
        if h == "healthy":
            return
        if h is None and s.get("running"):
            return      # no healthcheck declared: 'running' means ready
        if h == "unhealthy":
            raise RuntimeError(f"container {name} became unhealthy")
        if not s.get("running"):
            raise RuntimeError(f"container {name} exited while starting ({s.get('status')})")

        now = loop.time()
        if emit is not None and now - last_emit > 5:
            await emit("status", {"phase": "waiting-healthy", "container": name,
                                  "message": f"Waiting for {name} healthcheck ({h or 'starting'})"})
            last_emit = now
        if now >= deadline:
            raise TimeoutError(f"{name} did not become healthy within {timeout_s}s (last health={h})")
        await asyncio.sleep(2)


# ── OCR GPU worker (inside ocr_service) ────

async def _ocr_gpu(action: str) -> None:
    timeout = OCR_ENABLE_TIMEOUT_SEC if action == "enable" else 120
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{OCR_SERVICE_URL}/gpu/{action}")
    if resp.status_code >= 400:
        raise RuntimeError(f"ocr_service /gpu/{action} -> {resp.status_code}: {resp.text[:200]}")


async def _upgrade_heuristic_docs() -> None:
    """Re-index what was indexed while the LLM was off (fire-and-forget)."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{RAG_SERVICE_URL}/ingest/sync",
                                     params={"upgrade_heuristic": "true"})
        log.info("[gpu_swap] rag upgrade sync -> %d %s", resp.status_code, resp.text[:200])
    except httpx.HTTPError as e:
        log.warning("[gpu_swap] rag upgrade sync failed: %s", e)


# ── High-level swaps ──────

async def _emit(emit: Optional[EmitStatus], payload: dict) -> None:
    if emit is not None:
        await emit("status", payload)


async def swap_to_ocr(emit: Optional[EmitStatus] = None) -> None:
    """Stop the LLM container (frees its VRAM), then start the OCR GPU worker."""
    if await is_running(LLM_CONTAINER_NAME):
        await _emit(emit, {"phase": "swap-to-ocr", "message": f"Stopping {LLM_CONTAINER_NAME} to free GPU VRAM"})
        await stop(LLM_CONTAINER_NAME)
        await asyncio.sleep(3)      # let the driver release the context
    await _emit(emit, {"phase": "swap-to-ocr", "message": "Loading Docling models on the GPU"})
    await _ocr_gpu("enable")
    await _emit(emit, {"phase": "swap-to-ocr", "message": "OCR is ready"})


async def swap_to_llm(emit: Optional[EmitStatus] = None) -> None:
    """Stop the OCR GPU worker (its process exit frees all VRAM), then start
    the LLM container and wait for its healthcheck."""
    try:
        await _ocr_gpu("disable")
    except (httpx.HTTPError, RuntimeError) as e:
        # ocr_service down means its worker is gone too.
        log.warning("[gpu_swap] ocr /gpu/disable failed (continuing): %s", e)
    await asyncio.sleep(2)
    if not await is_running(LLM_CONTAINER_NAME):
        await _emit(emit, {"phase": "swap-to-llm", "message": f"Starting {LLM_CONTAINER_NAME}"})
        await start(LLM_CONTAINER_NAME)
    await _emit(emit, {"phase": "swap-to-llm",
                       "message": "Waiting for the LLM healthcheck (SGLang cold start takes several minutes)"})
    await wait_healthy(LLM_CONTAINER_NAME, emit=emit)
    await _emit(emit, {"phase": "swap-to-llm", "message": "LLM is ready"})


# ── Arbiter: leases + swap decisions ──────

@dataclass
class _Lease:
    tenant: str
    expires: float


class _Arbiter:
    def __init__(self) -> None:
        self.tenant: Optional[str] = None   # who owns the GPU (None: unknown / failed)
        self.ready = False                  # False while a swap runs
        self.swapping_to: Optional[str] = None
        self.message = "starting"
        self.error: Optional[str] = None
        self.leases: dict[str, _Lease] = {}
        self.waiting = {t: 0 for t in TENANTS}
        self.last_used = {t: 0.0 for t in TENANTS}
        self._changed = asyncio.Condition()
        self._task: Optional[asyncio.Task] = None
        self._side_tasks: set[asyncio.Task] = set()

    # helpers
    def held(self, tenant: str) -> int:
        return sum(1 for l in self.leases.values() if l.tenant == tenant)

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    def _grantable(self, tenant: str) -> bool:
        if not (self.ready and self.tenant == tenant):
            return False
        # New LLM work queues behind waiting OCR so an ingestion batch gets its turn.
        return not (tenant == "llm" and self.waiting["ocr"] > 0)

    def _target(self) -> Optional[str]:
        now = time.monotonic()
        if self.tenant is None:
            return "ocr" if self.waiting["ocr"] or self.held("ocr") else "llm"
        if self.tenant == "llm":
            if self.waiting["ocr"] and not self.held("llm"):
                return "ocr"
        elif self.tenant == "ocr":
            if not self.held("ocr") and not self.waiting["ocr"] \
                    and now - self.last_used["ocr"] >= IDLE_SEC:
                return "llm"
        return None

    # lifecycle
    def start(self) -> None:
        if GPU_SWAP_ENABLED and self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _reconcile(self) -> None:
        """Startup: the LLM is the default tenant."""
        llm = await state(LLM_CONTAINER_NAME)
        if llm.get("running") and llm.get("health") in {"healthy", "starting", None}:
            try:
                await _ocr_gpu("disable")
            except (httpx.HTTPError, RuntimeError) as e:
                log.info("[gpu_swap] reconcile: ocr /gpu/disable skipped: %s", e)
            self.tenant, self.ready = "llm", False
            self.message = "LLM starting"
            await wait_healthy(LLM_CONTAINER_NAME)
            self.ready, self.message = True, "LLM ready"
        else:
            await self._swap("llm")

    async def _swap(self, target: str) -> None:
        self.ready, self.swapping_to, self.error = False, target, None
        prev = self.tenant
        await self._notify()

        async def emit(_event: str, payload: dict) -> None:
            self.message = payload.get("message", self.message)
            await self._notify()

        log.info("[gpu_swap] swap %s -> %s", prev, target)
        t0 = time.monotonic()
        try:
            await (swap_to_ocr if target == "ocr" else swap_to_llm)(emit)
        except (DockerException, httpx.HTTPError, RuntimeError, TimeoutError) as e:
            log.error("[gpu_swap] swap to %s failed: %s", target, e)
            self.tenant, self.error = None, f"swap to {target} failed: {e}"
            self.message = self.error
            self.swapping_to = None
            await self._notify()
            await asyncio.sleep(RETRY_SEC)
            return
        self.tenant, self.ready, self.swapping_to = target, True, None
        self.last_used[target] = time.monotonic()
        self.message = f"{target.upper()} ready"
        log.info("[gpu_swap] swap %s -> %s done in %.1fs", prev, target, time.monotonic() - t0)
        await self._notify()
        if prev == "ocr" and target == "llm":
            t = asyncio.create_task(_upgrade_heuristic_docs())
            self._side_tasks.add(t)
            t.add_done_callback(self._side_tasks.discard)

    async def _run(self) -> None:
        try:
            await self._reconcile()
        except Exception as e:      # keep the manager alive; the loop retries
            log.exception("[gpu_swap] reconcile failed")
            self.tenant, self.ready, self.error = None, False, str(e)
        while True:
            now = time.monotonic()
            for lid, l in list(self.leases.items()):
                if l.expires < now:
                    log.warning("[gpu_swap] lease %s (%s) expired", lid, l.tenant)
                    self.leases.pop(lid)
                    self.last_used[l.tenant] = now
            target = self._target()
            if target is not None:
                await self._swap(target)
                continue
            async with self._changed:
                try:
                    await asyncio.wait_for(self._changed.wait(), timeout=2)
                except asyncio.TimeoutError:
                    pass

    # leases
    async def acquire(self, tenant: str, ttl: float = LEASE_TTL_SEC,
                      emit: Optional[EmitStatus] = None) -> str:
        if tenant not in TENANTS:
            raise ValueError(f"unknown tenant {tenant!r}")
        if not GPU_SWAP_ENABLED:
            return "noop"
        self.waiting[tenant] += 1
        await self._notify()
        last_msg, last_emit = None, 0.0
        try:
            async with self._changed:
                while not self._grantable(tenant):
                    # On change, and every 30 s as a heartbeat for streaming callers.
                    if emit is not None and (self.message != last_msg
                                             or time.monotonic() - last_emit > 30):
                        last_msg, last_emit = self.message, time.monotonic()
                        await emit("status", {"phase": "waiting", "tenant": tenant,
                                              "message": self._wait_message(tenant)})
                    try:
                        await asyncio.wait_for(self._changed.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        pass
                lid = uuid.uuid4().hex
                self.leases[lid] = _Lease(tenant, time.monotonic() + ttl)
                self.last_used[tenant] = time.monotonic()
                return lid
        finally:
            self.waiting[tenant] -= 1
            await self._notify()

    def _wait_message(self, tenant: str) -> str:
        if tenant == "llm" and (self.tenant == "ocr" or self.swapping_to == "ocr" or self.waiting["ocr"]):
            return f"GPU busy with document ingestion; the LLM restarts when it finishes. {self.message}"
        return f"Waiting for the GPU ({tenant}): {self.message}"

    def renew(self, lease_id: str, ttl: float = LEASE_TTL_SEC) -> bool:
        if lease_id == "noop":
            return True
        lease = self.leases.get(lease_id)
        if lease is None:
            return False
        lease.expires = time.monotonic() + ttl
        self.last_used[lease.tenant] = time.monotonic()
        return True

    async def release(self, lease_id: str) -> None:
        lease = self.leases.pop(lease_id, None)
        if lease is not None:
            self.last_used[lease.tenant] = time.monotonic()
            await self._notify()

    def status(self) -> dict:
        return {"tenant": self.tenant, "ready": self.ready, "swapping_to": self.swapping_to,
                "message": self.message, "error": self.error,
                "leases": {t: self.held(t) for t in TENANTS},
                "waiting": dict(self.waiting), "idle_sec": IDLE_SEC}


arbiter = _Arbiter()


async def wait_lease(tenant: str) -> AsyncIterator[tuple[str, dict]]:
    """Acquire a lease, yielding ("status", payload) while waiting and finally
    ("lease", {"id": lease_id}). Lets a streaming caller show the wait."""
    queue: asyncio.Queue[dict] = asyncio.Queue()

    async def emit(_event: str, payload: dict) -> None:
        await queue.put(payload)

    acq = asyncio.create_task(arbiter.acquire(tenant, emit=emit))
    try:
        while not acq.done():
            get = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({acq, get}, return_when=asyncio.FIRST_COMPLETED)
            if get in done:
                yield "status", get.result()
            else:
                get.cancel()
        yield "lease", {"id": acq.result()}
    finally:
        if not acq.done():
            acq.cancel()


@asynccontextmanager
async def renewing(lease_id: str) -> AsyncIterator[str]:
    """Keep a lease alive while the body runs, release it afterwards."""
    async def _renew() -> None:
        while True:
            await asyncio.sleep(LEASE_TTL_SEC / 3)
            arbiter.renew(lease_id)

    renewer = asyncio.create_task(_renew()) if lease_id != "noop" else None
    try:
        yield lease_id
    finally:
        if renewer:
            renewer.cancel()
        await arbiter.release(lease_id)


@asynccontextmanager
async def lease(tenant: str, emit: Optional[EmitStatus] = None) -> AsyncIterator[str]:
    """In-process lease, renewed in the background while the body runs."""
    lid = await arbiter.acquire(tenant, emit=emit)
    async with renewing(lid):
        yield lid


# ── Read-only snapshot for /gpu/status ──────

async def snapshot() -> dict:
    out = {"gpu_swap_enabled": GPU_SWAP_ENABLED}
    if not GPU_SWAP_ENABLED:
        return out
    try:
        llm = await state(LLM_CONTAINER_NAME)
    except DockerException as e:
        llm = {"error": str(e)}
    return {**out, **arbiter.status(),
            "llm": {"name": LLM_CONTAINER_NAME, "running": llm.get("running", False),
                    "health": llm.get("health")}}


def docker_reachable() -> bool:
    """Cheap probe for /health: can we even talk to the docker socket?"""
    try:
        _get_client().ping()
        return True
    except Exception as e:
        log.warning("docker ping failed: %s", e)
        return False
