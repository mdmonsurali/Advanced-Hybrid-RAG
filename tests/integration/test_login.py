"""auth_service on its own, and ui_service checking sessions against it."""

from __future__ import annotations

import uuid

import httpx
from conftest import AUTH_URL, MASTER, UI_URL


def test_master_logs_in_and_creates_a_user():
    with httpx.Client(base_url=AUTH_URL) as master:
        resp = master.post("/auth/login", json={"username": MASTER[0], "password": MASTER[1]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["role"] == "master"

        name = f"ci-{uuid.uuid4().hex[:8]}"
        created = master.post(
            "/auth/admin/users", json={"username": name, "password": "ci-password-1"}
        )
        assert created.status_code == 201, created.text

    with httpx.Client(base_url=AUTH_URL) as user:
        login = user.post("/auth/login", json={"username": name, "password": "ci-password-1"})
        assert login.json()["must_change_password"] is True
        assert user.get("/auth/me").json()["username"] == name
        assert user.get("/auth/admin/users").status_code == 403


def test_ui_sends_anonymous_visitors_to_the_login_page():
    resp = httpx.get(f"{UI_URL}/history", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/login")


def test_ui_login_goes_through_auth_service():
    with httpx.Client(base_url=UI_URL, follow_redirects=False) as browser:
        resp = browser.post(
            "/login", data={"username": MASTER[0], "password": MASTER[1], "next": "/history"}
        )
        assert resp.status_code == 303
        assert resp.headers["location"] == "/history"
        assert browser.cookies.get("montashi_session")

        me = browser.get("/me")
        assert me.status_code == 200
        assert me.json()["username"] == MASTER[0]

        assert browser.post("/logout").status_code == 200
        browser.cookies.clear()
        assert browser.get("/me").status_code == 303


def test_ui_rejects_a_wrong_password():
    resp = httpx.post(
        f"{UI_URL}/login", data={"username": MASTER[0], "password": "wrong"}, follow_redirects=False
    )
    assert resp.headers["location"].startswith("/login?err=")
