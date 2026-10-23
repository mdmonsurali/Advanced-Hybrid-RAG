"""Login gate: the auth middleware, login redirects and the validate cache."""

from __future__ import annotations

import auth_client
import main
import pytest
from conftest import USER
from fastapi.testclient import TestClient
from middleware import AuthMiddleware
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

COOKIE = auth_client.SESSION_COOKIE_NAME


@pytest.fixture
def client():
    return TestClient(main.app, follow_redirects=False)


def test_public_pages_need_no_login(client, session):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/login").status_code == 200


def test_pages_redirect_to_login_and_keep_the_target(client, session):
    resp = client.get("/history?page=2")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login?next=/history%3Fpage%3D2"


def test_chat_api_calls_get_401_instead_of_a_redirect(client, session):
    resp = client.post("/chat/project/settings")
    assert resp.status_code == 401
    assert resp.json() == {"detail": "Not authenticated"}


def test_forced_password_change(client, session):
    session["sid"] = auth_client.CurrentUser("u-1", "alice", "user", must_change_password=True)
    client.cookies.set(COOKIE, "sid")
    resp = client.get("/history")
    assert (resp.status_code, resp.headers["location"]) == (303, "/change-password")
    assert client.get("/change-password").status_code == 200


def test_logged_in_pages_are_never_cached(client, session):
    session["sid"] = USER
    client.cookies.set(COOKIE, "sid")
    resp = client.get("/me")
    assert resp.status_code == 200
    assert "no-store" in resp.headers["cache-control"]


def test_admin_pages_need_an_admin(client, session):
    session["sid"] = USER
    client.cookies.set(COOKIE, "sid")
    assert client.get("/admin/users").status_code == 403


def test_inbound_identity_headers_are_stripped():
    """A browser must not be able to pretend to be another user to the backends."""

    async def echo(request):
        return JSONResponse({k: v for k, v in request.headers.items() if k.startswith("x-")})

    app = Starlette(routes=[Route("/health", echo)])
    app.add_middleware(AuthMiddleware)
    resp = TestClient(app).get(
        "/health",
        headers={
            "X-User-Id": "admin",
            "X-User-Role": "master",
            "X-Langfuse-Trace-Id": "1",
            "X-Ok": "1",
        },
    )
    assert resp.json() == {"x-ok": "1"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "/"),
        ("/history", "/history"),
        ("//evil.example", "/"),
        ("https://evil.example", "/"),
        ("javascript:alert(1)", "/"),
    ],
)
def test_login_redirect_target_stays_on_this_site(raw, expected):
    assert main._safe_next(raw) == expected


def test_login_sets_cookie_and_redirects(client, monkeypatch):
    async def login(username, password):
        return "new-sid", {"must_change_password": False}

    monkeypatch.setattr(auth_client, "login", login)
    resp = client.post("/login", data={"username": "a", "password": "b", "next": "/history"})
    assert (resp.status_code, resp.headers["location"]) == (303, "/history")
    assert f"{COOKIE}=new-sid" in resp.headers["set-cookie"]
    assert "httponly" in resp.headers["set-cookie"].lower()


def test_failed_login_shows_the_error(client, monkeypatch):
    async def login(username, password):
        return None, {"detail": "Invalid credentials"}

    monkeypatch.setattr(auth_client, "login", login)
    resp = client.post("/login", data={"username": "a", "password": "b"})
    assert resp.headers["location"] == "/login?err=Invalid%20credentials"


# ── validate cache ───────────────────────────────────────────────────────


class FakeAuthService:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.calls = 0

    def client(self, *args, **kwargs):
        service = self

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, cookies=None):
                service.calls += 1
                body = {"user_id": "u-1", "username": "alice", "role": "user"}
                return type("R", (), {"status_code": service.status, "json": lambda _: body})()

        return _Client()


@pytest.mark.anyio
async def test_validate_caches_successful_lookups(monkeypatch):
    service = FakeAuthService()
    monkeypatch.setattr(auth_client.httpx, "AsyncClient", service.client)
    auth_client._cache.clear()

    first = await auth_client.validate("sid-1")
    second = await auth_client.validate("sid-1")
    assert first == second == auth_client.CurrentUser("u-1", "alice", "user", False)
    assert service.calls == 1
    assert await auth_client.validate("") is None


@pytest.mark.anyio
async def test_rejected_sessions_are_not_cached(monkeypatch):
    service = FakeAuthService(status=401)
    monkeypatch.setattr(auth_client.httpx, "AsyncClient", service.client)
    auth_client._cache.clear()

    assert await auth_client.validate("sid-2") is None
    assert await auth_client.validate("sid-2") is None
    assert service.calls == 2
