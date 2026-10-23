"""GPU arbiter policy: who should own the GPU next, and lease bookkeeping."""

from __future__ import annotations

import asyncio
import time

import gpu_swap
import pytest

pytestmark = pytest.mark.anyio


def arbiter(tenant: str | None, ready: bool = True) -> gpu_swap._Arbiter:
    a = gpu_swap._Arbiter()
    a.tenant, a.ready = tenant, ready
    return a


def add_lease(a: gpu_swap._Arbiter, tenant: str) -> str:
    lid = f"lease-{len(a.leases)}"
    a.leases[lid] = gpu_swap._Lease(tenant, time.monotonic() + 60)
    return lid


def test_unknown_owner_defaults_to_llm_unless_ocr_waits():
    assert arbiter(None)._target() == "llm"
    a = arbiter(None)
    a.waiting["ocr"] = 1
    assert a._target() == "ocr"


def test_llm_hands_over_only_when_no_llm_lease_is_held():
    a = arbiter("llm")
    assert a._target() is None
    a.waiting["ocr"] = 1
    assert a._target() == "ocr"
    add_lease(a, "llm")
    assert a._target() is None  # an answer is still being generated


def test_ocr_returns_gpu_after_idle_period(monkeypatch):
    monkeypatch.setattr(gpu_swap, "IDLE_SEC", 30)
    a = arbiter("ocr")
    a.last_used["ocr"] = time.monotonic()
    assert a._target() is None  # just used: keep the OCR window open
    a.last_used["ocr"] = time.monotonic() - 31
    assert a._target() == "llm"
    add_lease(a, "ocr")
    assert a._target() is None


def test_new_llm_work_queues_behind_waiting_ocr():
    a = arbiter("llm")
    assert a._grantable("llm")
    a.waiting["ocr"] = 1
    assert not a._grantable("llm")
    assert not arbiter("llm", ready=False)._grantable("llm")  # mid-swap


async def test_acquire_and_release_when_swap_is_on(monkeypatch):
    monkeypatch.setattr(gpu_swap, "GPU_SWAP_ENABLED", True)
    a = arbiter("llm")
    lid = await a.acquire("llm", ttl=60)
    assert a.held("llm") == 1
    assert a.waiting["llm"] == 0
    assert a.renew(lid)
    await a.release(lid)
    assert a.held("llm") == 0
    assert not a.renew(lid)


async def test_acquire_waits_until_the_gpu_is_swapped(monkeypatch):
    monkeypatch.setattr(gpu_swap, "GPU_SWAP_ENABLED", True)
    a = arbiter("llm")
    statuses: list[dict] = []

    async def emit(_event, payload):
        statuses.append(payload)

    task = asyncio.create_task(a.acquire("ocr", emit=emit))
    await asyncio.sleep(0.05)
    assert not task.done()
    assert a.waiting["ocr"] == 1
    assert statuses and statuses[0]["phase"] == "waiting"

    a.tenant = "ocr"  # what a finished swap does
    await a._notify()
    lid = await asyncio.wait_for(task, timeout=2)
    assert a.leases[lid].tenant == "ocr"


async def test_unknown_tenant_is_rejected():
    with pytest.raises(ValueError):
        await arbiter("llm").acquire("tpu")


async def test_failed_swap_records_the_error(monkeypatch):
    async def broken(emit=None):
        raise RuntimeError("ocr_service /gpu/enable -> 500")

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(gpu_swap, "swap_to_ocr", broken)
    monkeypatch.setattr(gpu_swap.asyncio, "sleep", no_sleep)
    a = arbiter("llm")
    await a._swap("ocr")
    assert a.tenant is None
    assert a.ready is False
    assert "swap to ocr failed" in (a.error or "")


async def test_successful_swap_back_to_llm_triggers_reindex(monkeypatch):
    calls: list[str] = []

    async def ok(emit=None):
        calls.append("swap_to_llm")

    async def upgrade():
        calls.append("reindex")

    monkeypatch.setattr(gpu_swap, "swap_to_llm", ok)
    monkeypatch.setattr(gpu_swap, "_upgrade_heuristic_docs", upgrade)
    a = arbiter("ocr")
    await a._swap("llm")
    await asyncio.sleep(0)  # let the fire-and-forget task run
    assert (a.tenant, a.ready) == ("llm", True)
    assert calls == ["swap_to_llm", "reindex"]


async def test_wait_lease_yields_lease_immediately_when_swap_is_off():
    events = [e async for e in gpu_swap.wait_lease("llm")]
    assert events == [("lease", {"id": "noop"})]
