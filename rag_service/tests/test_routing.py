"""STEP 5: rule-based routing, user overrides, LLM fallback, executors."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from conftest import ADMIN, USER, plan
from mcp_servers import gdrive_api
from query_route import executors, router
from schemas import Evidence

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def drive_configured(monkeypatch):
    monkeypatch.setattr(gdrive_api, "is_configured", lambda: True)


@pytest.fixture
def llm_route(monkeypatch):
    calls: list[list[str]] = []

    async def fake(plan_dict, available):
        calls.append(list(available))
        return {"routes": ["web"], "reason": "llm says web"}

    monkeypatch.setattr(router.llm_api, "route", fake)
    return calls


def test_drive_is_only_offered_to_allowed_roles():
    assert "gdrive" not in router.available_routes(USER)
    assert "gdrive" in router.available_routes(ADMIN)


@pytest.mark.parametrize(
    ("question", "intent", "expected"),
    [
        ("hi there", "chitchat", ["direct"]),
        ("how many invoices do I have", "catalog", ["catalog"]),
        ("latest news on interest rates", "fact", ["web", "docs"]),
        ("find it in my google drive", "fact", ["docs"]),  # USER may not use Drive
        ("what does clause 4 say", "fact", None),  # no rule: the LLM decides
    ],
)
def test_rules(question, intent, expected):
    out = router._rules(plan(question, intent=intent), router.available_routes(USER))
    assert (out[0] if out else None) == expected


async def test_admin_drive_question_goes_to_drive(llm_route):
    out = await router.decide(plan("search my google drive for the budget", intent="fact"), ADMIN)
    assert out.routes == ["gdrive"]
    assert out.decided_by == "rules"


async def test_user_forced_route_wins(llm_route):
    out = await router.decide(plan(), USER, route_hint=["web"])
    assert (out.routes, out.decided_by) == (["web"], "user")


async def test_forcing_a_forbidden_route_falls_back_to_docs(llm_route):
    out = await router.decide(plan(), USER, route_hint=["gdrive"])
    assert (out.routes, out.decided_by) == (["docs"], "default")


async def test_denied_routes_are_not_offered_to_the_llm(llm_route):
    out = await router.decide(
        plan("what does clause 4 say", intent="fact"), USER, deny_routes=["web"]
    )
    assert "web" not in llm_route[0]
    assert out.routes == ["docs"]  # the LLM's "web" is not available any more
    assert out.decided_by == "llm"


async def test_llm_outage_defaults_to_docs(monkeypatch):
    async def down(*args):
        raise httpx.ConnectError("llm_service down")

    monkeypatch.setattr(router.llm_api, "route", down)
    out = await router.decide(plan("what does clause 4 say", intent="fact"), USER)
    assert (out.routes, out.decided_by) == (["docs"], "default")


# ── executors ────────────────────────────────────────────────────────────


def hit(id: str, route: str = "docs") -> Evidence:
    return Evidence(id=id, route=route, text=f"text {id}", score=0.5)


@pytest.fixture
def sources(monkeypatch):
    async def docs(p, identity, broaden):
        return [(hit("d1"), "q"), (hit("d2"), "q")]

    async def web(p):
        raise asyncio.CancelledError()  # not an Exception subclass (Step 9 fix)

    async def gdrive(p):
        return []

    async def rerank(candidates):
        return [ev for ev, _ in candidates]

    monkeypatch.setattr(executors, "search_docs", docs)
    monkeypatch.setattr(executors, "web_evidence", web)
    monkeypatch.setattr(executors, "gdrive_evidence", gdrive)
    monkeypatch.setattr(executors, "rerank", rerank)


async def test_a_failing_route_is_reported_not_fatal(sources):
    out = await executors.execute(plan(), ["docs", "web", "gdrive", "direct"], USER)
    assert out.route_status == {
        "docs": "ok",
        "web": "unavailable",
        "gdrive": "empty",
        "direct": "skipped",
    }
    assert [e.id for e in out.evidence] == ["d1", "d2"]


async def test_external_tool_output_is_chunked_for_the_reranker(sources):
    long = Evidence(id="tool", route="mcp", text="Sentence one. " * 300)
    out = await executors.execute(plan(), [], USER, external=[long])
    ids = [e.id for e in out.evidence]
    assert len(ids) > 1
    assert ids[0] == "tool:0"
    assert out.route_status == {"mcp": "ok"}
