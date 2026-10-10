"""File-backed user / session store for auth_service.

Users and sessions live in memory and are persisted to one JSON file
(AUTH_DATA_FILE, on a docker volume) after every change. Writes go to a temp
file and are renamed into place, so a crash never leaves a half-written file.
auth_service is a single process, so an asyncio lock is enough to serialize
mutations.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("auth_service.db")

DATA_FILE = Path(os.getenv("AUTH_DATA_FILE", "/data/auth.json"))

USER_FIELDS = ("id", "username", "role", "must_change_password", "disabled",
               "created_at", "updated_at")

_users: dict[_uuid.UUID, dict] = {}
_sessions: dict[_uuid.UUID, dict] = {}
_lock = asyncio.Lock()
_loaded = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Persistence ──────────────────────────────────────────────────────────
_DT_FIELDS = {"created_at", "updated_at", "expires_at", "last_seen_at"}
_UUID_FIELDS = {"id", "user_id", "session_id"}


def _encode(row: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime)
                else str(v) if isinstance(v, _uuid.UUID) else v)
            for k, v in row.items()}


def _decode(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k in _DT_FIELDS and v is not None:
            v = datetime.fromisoformat(v)
        elif k in _UUID_FIELDS and v is not None:
            v = _uuid.UUID(v)
        out[k] = v
    return out


def _save() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DATA_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "users": [_encode(u) for u in _users.values()],
        "sessions": [_encode(s) for s in _sessions.values()],
    }, indent=2), encoding="utf-8")
    os.replace(tmp, DATA_FILE)


# ── Lifecycle ────────────────────────────────────────────────────────────
async def init_store() -> None:
    global _loaded
    if DATA_FILE.exists():
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        for u in data.get("users", []):
            u = _decode(u)
            _users[u["id"]] = u
        for s in data.get("sessions", []):
            s = _decode(s)
            _sessions[s["session_id"]] = s
    _loaded = True
    log.info("auth store ready: %s (%d users, %d sessions)",
             DATA_FILE, len(_users), len(_sessions))


async def health() -> None:
    if not _loaded:
        raise RuntimeError("auth store not initialised")
    if not os.access(DATA_FILE.parent, os.W_OK):
        raise RuntimeError(f"{DATA_FILE.parent} is not writable")


# ── Users ────────────────────────────────────────────────────────────────
def _public(user: dict, with_hash: bool = False) -> dict:
    out = {k: user[k] for k in USER_FIELDS}
    if with_hash:
        out["password_hash"] = user["password_hash"]
    return out


def _find_by_username(username: str) -> Optional[dict]:
    needle = username.lower()
    return next((u for u in _users.values() if u["username"].lower() == needle), None)


async def get_user_by_username(username: str) -> Optional[dict]:
    user = _find_by_username(username)
    return _public(user, with_hash=True) if user else None


async def get_user_by_id(user_id: _uuid.UUID) -> Optional[dict]:
    user = _users.get(user_id)
    return _public(user) if user else None


async def create_user(username: str, password_hash: str, role: str,
                      must_change_password: bool = True) -> dict:
    async with _lock:
        if _find_by_username(username):
            raise ValueError(f"username already exists: {username}")
        now = _now()
        user = {
            "id": _uuid.uuid4(),
            "username": username,
            "password_hash": password_hash,
            "role": role,
            "must_change_password": must_change_password,
            "disabled": False,
            "created_at": now,
            "updated_at": now,
        }
        _users[user["id"]] = user
        _save()
    return _public(user)


async def list_users() -> list[dict]:
    return [_public(u) for u in sorted(_users.values(), key=lambda u: u["created_at"])]


async def count_admins() -> int:
    return sum(1 for u in _users.values() if u["role"] == "admin" and not u["disabled"])


async def find_first_user(role: str) -> Optional[dict]:
    """Oldest user with `role` (used by the master bootstrap)."""
    matches = sorted((u for u in _users.values() if u["role"] == role),
                     key=lambda u: u["created_at"])
    return _public(matches[0]) if matches else None


_USER_UPDATE_COLS = {"username", "password_hash", "role", "must_change_password", "disabled"}


async def update_user(user_id: _uuid.UUID, **fields) -> Optional[dict]:
    async with _lock:
        user = _users.get(user_id)
        if user is None:
            return None
        cols = {k: v for k, v in fields.items() if k in _USER_UPDATE_COLS}
        if cols:
            user.update(cols)
            user["updated_at"] = _now()
            _save()
    return _public(user)


async def delete_user(user_id: _uuid.UUID) -> bool:
    async with _lock:
        if _users.pop(user_id, None) is None:
            return False
        for sid in [s for s, row in _sessions.items() if row["user_id"] == user_id]:
            del _sessions[sid]
        _save()
    return True


# ── Sessions ─────────────────────────────────────────────────────────────
async def create_session(user_id: _uuid.UUID, ttl: timedelta) -> dict:
    now = _now()
    sess = {
        "session_id": _uuid.uuid4(),
        "user_id": user_id,
        "expires_at": now + ttl,
        "created_at": now,
        "last_seen_at": now,
    }
    async with _lock:
        _sessions[sess["session_id"]] = sess
        _save()
    return dict(sess)


async def fetch_session_with_user(session_id: _uuid.UUID) -> Optional[dict]:
    """Return {session_id, user_id, expires_at, username, role, must_change_password,
    disabled} or None if expired / missing / user disabled."""
    sess = _sessions.get(session_id)
    if sess is None or sess["expires_at"] <= _now():
        return None
    user = _users.get(sess["user_id"])
    if user is None or user["disabled"]:
        return None
    return {
        "session_id": sess["session_id"],
        "user_id": sess["user_id"],
        "expires_at": sess["expires_at"],
        "username": user["username"],
        "role": user["role"],
        "must_change_password": user["must_change_password"],
        "disabled": user["disabled"],
    }


async def touch_session(session_id: _uuid.UUID) -> None:
    # In memory only: persisted with the next write, so /me polling doesn't
    # rewrite the file on every request.
    sess = _sessions.get(session_id)
    if sess is not None:
        sess["last_seen_at"] = _now()


async def delete_session(session_id: _uuid.UUID) -> None:
    async with _lock:
        if _sessions.pop(session_id, None) is not None:
            _save()


async def delete_user_sessions(user_id: _uuid.UUID,
                               except_session: Optional[_uuid.UUID] = None) -> None:
    async with _lock:
        doomed = [sid for sid, s in _sessions.items()
                  if s["user_id"] == user_id and sid != except_session]
        for sid in doomed:
            del _sessions[sid]
        if doomed:
            _save()


async def cleanup_expired_sessions() -> int:
    async with _lock:
        now = _now()
        doomed = [sid for sid, s in _sessions.items() if s["expires_at"] < now]
        for sid in doomed:
            del _sessions[sid]
        if doomed:
            _save()
    return len(doomed)
