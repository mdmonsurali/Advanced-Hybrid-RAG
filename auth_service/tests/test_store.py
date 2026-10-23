"""The JSON-file store and the master bootstrap (no HTTP)."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import bootstrap
import db
import pytest
from security import hash_password, verify_password


def run(coro):
    return asyncio.run(coro)


def test_store_survives_a_restart():
    user = run(db.create_user("alice", hash_password("pw-123456"), "user"))
    sess = run(db.create_session(user["id"], timedelta(hours=1)))

    saved = json.loads(db.DATA_FILE.read_text(encoding="utf-8"))
    assert saved["users"][0]["username"] == "alice"
    assert isinstance(saved["users"][0]["created_at"], str)  # datetimes stored as ISO text

    db._users.clear()
    db._sessions.clear()
    run(db.init_store())

    reloaded = run(db.fetch_session_with_user(sess["session_id"]))
    assert reloaded is not None
    assert reloaded["username"] == "alice"
    assert reloaded["user_id"] == user["id"]  # UUIDs come back as UUIDs


def test_expired_sessions_are_rejected_and_cleaned_up():
    user = run(db.create_user("alice", "hash", "user"))
    sess = run(db.create_session(user["id"], timedelta(seconds=-1)))

    assert run(db.fetch_session_with_user(sess["session_id"])) is None
    assert run(db.cleanup_expired_sessions()) == 1
    assert not db._sessions


def test_deleting_a_user_deletes_their_sessions():
    user = run(db.create_user("alice", "hash", "user"))
    run(db.create_session(user["id"], timedelta(hours=1)))
    assert run(db.delete_user(user["id"])) is True
    assert not db._sessions
    assert run(db.delete_user(user["id"])) is False


def test_health_needs_an_initialised_store():
    with pytest.raises(RuntimeError):
        run(db.health())
    run(db.init_store())
    run(db.health())


def test_password_hashing():
    hashed = hash_password("secret-123")
    assert verify_password("secret-123", hashed)
    assert not verify_password("wrong", hashed)
    assert not verify_password("secret-123", "not-a-bcrypt-hash")  # never raises


# ── bootstrap ────────────────────────────────────────────────────────────


def test_bootstrap_creates_master_once():
    run(bootstrap.bootstrap_admin())
    run(bootstrap.bootstrap_admin())
    masters = [u for u in db._users.values() if u["role"] == "master"]
    assert len(masters) == 1
    assert masters[0]["username"] == "master"
    assert masters[0]["must_change_password"] is False


def test_bootstrap_re_enables_a_disabled_master():
    run(bootstrap.bootstrap_admin())
    master = next(iter(db._users.values()))
    master["disabled"] = True
    run(bootstrap.bootstrap_admin())
    assert master["disabled"] is False


def test_bootstrap_promotes_a_legacy_admin():
    legacy = run(db.create_user("oldadmin", "hash", "admin"))
    run(bootstrap.bootstrap_admin())
    row = db._users[legacy["id"]]
    assert row["role"] == "master"
    assert row["username"] == "master"
    assert len(db._users) == 1


def test_bootstrap_needs_a_password(monkeypatch):
    monkeypatch.delenv("MASTER_PASSWORD")
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="MASTER_PASSWORD"):
        bootstrap._read_credentials()


def test_bootstrap_accepts_admin_fallback_names(monkeypatch):
    monkeypatch.delenv("MASTER_USERNAME")
    monkeypatch.delenv("MASTER_PASSWORD")
    monkeypatch.setenv("ADMIN_USERNAME", "root")
    monkeypatch.setenv("ADMIN_PASSWORD", "root-pass-1")
    assert bootstrap._read_credentials() == ("root", "root-pass-1")
