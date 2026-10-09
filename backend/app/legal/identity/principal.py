"""Who is calling? Resolution of the authenticated principal in the Law edition.

HTTP: the opaque session cookie, or, only on the voice RAG routes, an on-behalf-of bearer token.
WebSocket: the session cookie plus an Origin check (cross-site WebSocket hijacking), and a
periodic re-check that closes the socket (code 4401) once the session is revoked or expires.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ...core.config import settings
from . import SESSION_COOKIE, obo, sessions

logger = logging.getLogger(__name__)

# Routes a service may call with an on-behalf-of token instead of a browser session.
OBO_PATHS = frozenset({"/api/chat/ask-voice", "/api/chat/ask-voice-stream"})
# Routes a partial (password-only, MFA pending) session may call.
PARTIAL_PATHS = frozenset({"/api/auth/mfa/verify", "/api/auth/mfa/enroll", "/api/auth/mfa/confirm"})
WS_CLOSE_UNAUTHORIZED = 4401


def _principal(s: dict, auth: str) -> dict:
    return {"sub": s["user_id"], "username": s["username"], "role": s["role"], "tenant": s["tenant"], "org_id": s["org_id"],
            "sid": s["token_hash"], "mfa_verified": s["mfa_verified"], "auth": auth, "session": s}


def from_request(request, allow_partial: bool = False) -> Optional[dict]:
    """The principal for an HTTP request, cached on the request for the middlewares and route."""
    cache = request.scope.setdefault("law_principal", {})
    if allow_partial in cache:
        return cache[allow_partial]
    principal = None
    authz = request.headers.get("authorization") or ""
    if authz[:7].lower() == "bearer ":
        if request.url.path in OBO_PATHS:
            v = obo.verify(authz[7:].strip(), audience="voice")
            principal = _principal(v["session"], "obo") if v else None
    else:
        s = sessions.lookup(request.cookies.get(SESSION_COOKIE, ""))
        if s and (s["mfa_verified"] or allow_partial):
            principal = _principal(s, "session")
    cache[allow_partial] = principal
    return principal


def origin_allowed(origin: Optional[str]) -> bool:
    """Browsers always send Origin on WebSocket handshakes; it must be one of our own origins.
    Non-browser clients (no Origin) cannot be driven cross-site, so they are judged by cookie alone."""
    if not origin:
        return True
    allowed = {o.strip() for o in (settings.CORS_ALLOW_ORIGINS or "").split(",") if o.strip()}
    return origin in allowed


def from_websocket(ws) -> Optional[dict]:
    if not origin_allowed(ws.headers.get("origin")):
        return None
    s = sessions.lookup(ws.cookies.get(SESSION_COOKIE, ""))
    return _principal(s, "session") if s and s["mfa_verified"] else None


def watch_websocket(ws, principal: dict) -> asyncio.Task:
    """Close the socket with 4401 as soon as the session behind it is no longer live."""
    async def _watch():
        try:
            while True:
                await asyncio.sleep(settings.LEGAL_WS_RECHECK_SEC)
                s = await asyncio.to_thread(sessions.lookup_hash, principal["sid"], True)
                if not s or not s["mfa_verified"]:
                    logger.info("Law WS: session ended, closing socket")
                    try:
                        await ws.close(code=WS_CLOSE_UNAUTHORIZED)
                    except Exception:
                        pass
                    return
        except asyncio.CancelledError:
            return
    return asyncio.create_task(_watch())
