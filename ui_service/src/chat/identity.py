"""Who is chatting.

Chainlit signs its own JWT at login, but the source of truth stays
auth_service: every message re-validates the `montashi_session` cookie from
the websocket handshake, so logouts, disabled accounts and role changes
apply immediately instead of when Chainlit's JWT expires.
"""

from __future__ import annotations

from http.cookies import SimpleCookie

import auth_client
import chainlit as cl


class SessionExpired(Exception):
    pass


def session_cookie_from(cookie_header: str) -> str | None:
    jar = SimpleCookie()
    try:
        jar.load(cookie_header or "")
    except Exception:
        return None
    morsel = jar.get(auth_client.SESSION_COOKIE_NAME)
    return morsel.value if morsel else None


def to_chainlit_user(user: auth_client.CurrentUser) -> cl.User:
    # identifier = auth user id (stable across renames); the username is
    # only the display name.
    return cl.User(
        identifier=user.id,
        display_name=user.username,
        metadata={"id": user.id, "role": user.role, "username": user.username},
    )


async def current_user() -> auth_client.CurrentUser:
    environ = getattr(cl.context.session, "environ", None) or {}
    sid = session_cookie_from(environ.get("HTTP_COOKIE", ""))
    user = await auth_client.validate(sid or "")
    if user is None:
        raise SessionExpired()
    return user


def headers_for(user: auth_client.CurrentUser) -> dict[str, str]:
    return {"X-User-Id": user.id, "X-User-Role": user.role}
