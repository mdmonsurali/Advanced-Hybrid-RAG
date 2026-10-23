"""Admin user management and the master / admin / user role rules."""

from __future__ import annotations

import uuid

import db
import main
from conftest import MASTER, create_user, login
from fastapi.testclient import TestClient


def test_regular_users_get_403_on_admin_routes(client):
    login(client, *MASTER)
    create_user(client, "alice")
    client.cookies.clear()
    login(client, "alice", "alice-pass-123")
    assert client.get("/auth/admin/users").status_code == 403


def test_create_user_validates_input(client):
    login(client, *MASTER)
    bad = [
        {"username": "a", "password": "long-enough-1"},  # too short
        {"username": "has space", "password": "long-enough-1"},
        {"username": "alice", "password": "short"},
        {"username": "alice", "password": "long-enough-1", "role": "master"},
    ]
    for body in bad:
        assert client.post("/auth/admin/users", json=body).status_code == 422, body


def test_duplicate_username_is_rejected(client):
    login(client, *MASTER)
    create_user(client, "alice")
    resp = client.post("/auth/admin/users", json={"username": "ALICE", "password": "long-enough-1"})
    assert resp.status_code == 409


def test_only_master_creates_admins(client):
    login(client, *MASTER)
    create_user(client, "boss", role="admin")
    client.cookies.clear()
    login(client, "boss", "boss-pass-123")

    resp = client.post(
        "/auth/admin/users",
        json={"username": "boss2", "password": "long-enough-1", "role": "admin"},
    )
    assert resp.status_code == 403
    assert create_user(client, "alice")["role"] == "user"  # admins can create users


def test_admins_only_see_users(client):
    login(client, *MASTER)
    create_user(client, "boss", role="admin")
    create_user(client, "alice")
    master_view = {u["username"] for u in client.get("/auth/admin/users").json()}
    assert master_view == {"boss", "alice"}  # master never lists itself

    client.cookies.clear()
    login(client, "boss", "boss-pass-123")
    assert {u["username"] for u in client.get("/auth/admin/users").json()} == {"alice"}


def test_admin_cannot_touch_other_admins(client):
    login(client, *MASTER)
    boss = create_user(client, "boss", role="admin")
    create_user(client, "boss2", role="admin")
    client.cookies.clear()
    login(client, "boss2", "boss2-pass-123")

    # 404, not 403: admins must not learn that other admins exist.
    assert (
        client.patch(f"/auth/admin/users/{boss['id']}", json={"disabled": True}).status_code == 404
    )
    assert client.delete(f"/auth/admin/users/{boss['id']}").status_code == 404


def test_update_user(client):
    login(client, *MASTER)
    alice = create_user(client, "alice")
    create_user(client, "bob")

    assert (
        client.patch(f"/auth/admin/users/{alice['id']}", json={"username": "bob"}).status_code
        == 409
    )
    renamed = client.patch(f"/auth/admin/users/{alice['id']}", json={"username": "alicia"})
    assert renamed.json()["username"] == "alicia"
    promoted = client.patch(f"/auth/admin/users/{alice['id']}", json={"role": "admin"})
    assert promoted.json()["role"] == "admin"


def test_password_reset_forces_change_and_revokes_sessions(client):
    login(client, *MASTER)
    alice = create_user(client, "alice")
    alice_client = TestClient(client.app)
    login(alice_client, "alice", "alice-pass-123")
    db._users[uuid.UUID(alice["id"])]["must_change_password"] = False

    resp = client.patch(f"/auth/admin/users/{alice['id']}", json={"password": "reset-pass-123"})
    assert resp.json()["must_change_password"] is True
    assert alice_client.get("/auth/me").status_code == 401


def test_last_admin_cannot_be_demoted_disabled_or_deleted(client):
    login(client, *MASTER)
    boss = create_user(client, "boss", role="admin")
    url = f"/auth/admin/users/{boss['id']}"
    assert client.patch(url, json={"role": "user"}).status_code == 400
    assert client.patch(url, json={"disabled": True}).status_code == 400
    assert client.delete(url).status_code == 400


def test_only_master_changes_roles(client):
    login(client, *MASTER)
    create_user(client, "boss", role="admin")
    alice = create_user(client, "alice")
    client.cookies.clear()
    login(client, "boss", "boss-pass-123")
    assert (
        client.patch(f"/auth/admin/users/{alice['id']}", json={"role": "admin"}).status_code == 403
    )


def test_update_returns_404_when_user_vanishes_mid_request(client, monkeypatch):
    """The user is deleted between the lookup and the update (Step 9 fix)."""
    login(client, *MASTER)
    alice = create_user(client, "alice")

    async def gone(*args, **kwargs):
        return None

    monkeypatch.setattr(main.db, "update_user", gone)
    resp = client.patch(f"/auth/admin/users/{alice['id']}", json={"disabled": True})
    assert resp.status_code == 404


def test_invalid_user_id(client):
    login(client, *MASTER)
    assert client.patch("/auth/admin/users/not-a-uuid", json={}).status_code == 400


def test_delete_user_wipes_documents_and_sessions(client):
    login(client, *MASTER)
    alice = create_user(client, "alice")

    assert client.delete(f"/auth/admin/users/{alice['id']}").status_code == 204
    assert client.wiped == [(alice["id"], "user")]
    assert client.delete(f"/auth/admin/users/{alice['id']}").status_code == 404


def test_admin_cannot_delete_itself(client):
    master = login(client, *MASTER)
    assert client.delete(f"/auth/admin/users/{master['user_id']}").status_code == 400
