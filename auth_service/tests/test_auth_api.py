"""Login, sessions and self-service endpoints."""

from __future__ import annotations

import db
from conftest import MASTER, create_user, login
from fastapi.testclient import TestClient
from security import SESSION_COOKIE_NAME


def test_health(client):
    assert client.get("/auth/health").json() == {"status": "ok"}


def test_login_sets_session_cookie(client):
    body = login(client, *MASTER)
    assert body["username"] == "master"
    assert body["role"] == "master"
    assert client.cookies.get(SESSION_COOKIE_NAME)


def test_login_rejects_wrong_password_and_unknown_user(client):
    for username, password in [("master", "wrong-password"), ("nobody", "whatever-123")]:
        resp = client.post("/auth/login", json={"username": username, "password": password})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid credentials"


def test_login_is_case_insensitive_on_username(client):
    assert login(client, "MASTER", MASTER[1])["username"] == "master"


def test_disabled_user_cannot_log_in(client):
    login(client, *MASTER)
    user = create_user(client, "alice")
    client.patch(f"/auth/admin/users/{user['id']}", json={"disabled": True})
    client.cookies.clear()
    resp = client.post("/auth/login", json={"username": "alice", "password": "alice-pass-123"})
    assert resp.status_code == 401


def test_me_requires_a_valid_session(client):
    assert client.get("/auth/me").status_code == 401
    client.cookies.set(SESSION_COOKIE_NAME, "not-a-uuid")
    assert client.get("/auth/me").json()["detail"] == "Invalid session"
    client.cookies.set(SESSION_COOKIE_NAME, "00000000-0000-0000-0000-000000000000")
    assert client.get("/auth/me").json()["detail"] == "Session expired"


def test_me_and_validate_return_the_session_user(client):
    login(client, *MASTER)
    for path in ("/auth/me", "/auth/validate"):
        body = client.get(path).json()
        assert body["username"] == "master"
        assert body["role"] == "master"
        assert body["must_change_password"] is False


def test_logout_ends_the_session(client):
    login(client, *MASTER)
    assert client.post("/auth/logout").status_code == 200
    assert not db._sessions
    assert client.get("/auth/me").status_code == 401


def test_new_users_must_change_password(client):
    login(client, *MASTER)
    create_user(client, "alice")
    client.cookies.clear()
    assert login(client, "alice", "alice-pass-123")["must_change_password"] is True


def test_change_password(client):
    login(client, *MASTER)
    create_user(client, "alice")
    client.cookies.clear()
    login(client, "alice", "alice-pass-123")

    wrong = client.post(
        "/auth/change-password",
        json={"current_password": "nope-nope-1", "new_password": "brand-new-pass"},
    )
    assert wrong.status_code == 400
    same = client.post(
        "/auth/change-password",
        json={"current_password": "alice-pass-123", "new_password": "alice-pass-123"},
    )
    assert same.status_code == 400

    ok = client.post(
        "/auth/change-password",
        json={"current_password": "alice-pass-123", "new_password": "brand-new-pass"},
    )
    assert ok.status_code == 200
    assert client.get("/auth/me").json()["must_change_password"] is False
    client.cookies.clear()
    assert login(client, "alice", "brand-new-pass")["username"] == "alice"


def test_change_password_revokes_other_sessions(client):
    login(client, *MASTER)
    create_user(client, "alice")
    other = TestClient(client.app)
    client.cookies.clear()
    login(client, "alice", "alice-pass-123")
    login(other, "alice", "alice-pass-123")

    client.post(
        "/auth/change-password",
        json={"current_password": "alice-pass-123", "new_password": "brand-new-pass"},
    )
    assert client.get("/auth/me").status_code == 200  # the session that changed it stays
    assert other.get("/auth/me").status_code == 401


def test_master_cannot_delete_itself(client):
    login(client, *MASTER)
    assert client.delete("/auth/me").status_code == 400


def test_user_can_delete_own_account(client):
    login(client, *MASTER)
    alice = create_user(client, "alice")
    client.cookies.clear()
    login(client, "alice", "alice-pass-123")

    assert client.delete("/auth/me").status_code == 204
    assert client.wiped == [(alice["id"], "user")]
    resp = client.post("/auth/login", json={"username": "alice", "password": "alice-pass-123"})
    assert resp.status_code == 401
