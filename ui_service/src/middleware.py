"""Auth middleware: every request must carry a valid session cookie,
otherwise redirect to /login. Forces /change-password when the user has
must_change_password = True."""

from __future__ import annotations

from urllib.parse import quote

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

import auth_client

_PUBLIC_PREFIXES = ("/static/",)
_PUBLIC_EXACT = {"/login", "/logout", "/health", "/favicon.ico"}
_FORCE_CHANGE_ALLOWED = {"/change-password", "/logout"}


def _is_chat_api(request: Request) -> bool:
    """A Chainlit XHR/fetch call (anything under /chat/ that isn't a page load)."""
    path = request.url.path
    if not path.startswith("/chat/"):
        return False
    return not (request.method == "GET" and "text/html" in request.headers.get("accept", ""))


def _is_public(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIXES)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Drop any X-User-* / X-Langfuse-* headers from the inbound request;
        # only the middleware/proxy layer and the backends set them.
        scope_headers = [
            (k, v)
            for (k, v) in request.scope["headers"]
            if not k.lower().startswith((b"x-user-", b"x-langfuse-"))
        ]
        request.scope["headers"] = scope_headers

        if _is_public(path):
            return await call_next(request)

        session_id = request.cookies.get(auth_client.SESSION_COOKIE_NAME)
        user = await auth_client.validate(session_id or "")
        if user is None:
            if _is_chat_api(request):
                # Chainlit's frontend calls these with fetch(); a redirect to the
                # login page would arrive as HTML. A 401 makes it show its
                # login state, which custom.js sends to /login.
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)
            next_url = quote(
                request.url.path + (("?" + request.url.query) if request.url.query else "")
            )
            return RedirectResponse(url=f"/login?next={next_url}", status_code=303)

        if user.must_change_password and path not in _FORCE_CHANGE_ALLOWED:
            return RedirectResponse(url="/change-password", status_code=303)

        request.state.user = user
        request.state.session_id = session_id
        response = await call_next(request)
        # Prevent the browser back-button (bfcache) from revealing an
        # authenticated page after logout. Belt-and-braces: no-store +
        # explicit no-cache directives.
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        return response
