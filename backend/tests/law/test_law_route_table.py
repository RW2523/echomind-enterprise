"""S1.1 / SEC-01 — deny by default: every HTTP route and WebSocket in the Law app refuses an
anonymous caller, except a short public allow-list. Generated from the live route table, so a
newly added route is covered automatically."""
from __future__ import annotations

import re
import threading

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from starlette.websockets import WebSocketDisconnect

from conftest import make_user, sign_in

# Starlette's TestClient opens ws:// unless given an absolute URL; the session cookie is Secure, so a
# browser (and the cookie jar) only sends it over wss://.
WSS = "wss://testserver"

PUBLIC = {
    ("GET", "/health"), ("GET", "/api/version"), ("GET", "/api/auth/config"),
    ("POST", "/api/auth/login"), ("POST", "/api/auth/logout"),
}
# FastAPI's own /docs, /redoc, /openapi.json are disabled at import in the Law edition; that is
# proven in a fresh interpreter by test_law_edition_posture.py (this process may have imported
# the app in the legacy edition first).
_FASTAPI_DOCS = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


def test_every_http_route_denies_anonymous_callers(law_app, client):
    checked, leaks = 0, []
    for route in law_app.routes:
        if not isinstance(route, APIRoute) or route.path in _FASTAPI_DOCS:
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            if (method, route.path) in PUBLIC:
                continue
            r = client.request(method, _concrete(route.path), json={})
            checked += 1
            if r.status_code != 401:
                leaks.append(f"{method} {route.path} -> {r.status_code}")
    assert checked > 60, checked
    assert not leaks, "routes reachable without a session:\n" + "\n".join(leaks)


def test_every_websocket_denies_anonymous_callers(law_app, client):
    ws_routes = [r for r in law_app.routes if isinstance(r, APIWebSocketRoute)]
    assert ws_routes
    for route in ws_routes:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(f"{WSS}{_concrete(route.path)}") as ws:
                ws.receive_text()
        assert exc.value.code == 1008, route.path


@pytest.fixture
def stub_ws_handler(monkeypatch):
    """Replace the ASR-backed live-transcript handler; we test the gate, not speech recognition."""
    from app.api.routes import transcribe

    async def handler(ws):
        await ws.accept()
        await ws.send_text("ready")
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return

    monkeypatch.setattr(transcribe, "ws_handler", handler)


def test_live_transcript_ws_accepts_a_session_and_closes_on_revocation(client, stub_ws_handler, monkeypatch):
    from app.core.config import settings
    from app.legal.identity import sessions
    monkeypatch.setattr(settings, "LEGAL_WS_RECHECK_SEC", 0.1)
    sign_in(client, make_user("a.rahman", mfa=True))
    token = client.cookies.get("__Host-echomind_s")
    with client.websocket_connect(f"{WSS}/api/transcribe/ws", headers={"Origin": "https://testserver"}) as ws:
        assert ws.receive_text() == "ready"
        threading.Timer(0.2, sessions.revoke, args=(token, "test")).start()
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4401


def test_live_transcript_ws_refuses_a_foreign_origin(client, stub_ws_handler):
    sign_in(client, make_user("a.rahman", mfa=True))
    with client.websocket_connect(f"{WSS}/api/transcribe/ws", headers={"Origin": "https://testserver"}) as ok:
        assert ok.receive_text() == "ready"                      # same session, own origin: accepted
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"{WSS}/api/transcribe/ws", headers={"Origin": "https://evil.example"}) as ws:
            ws.receive_text()
    assert exc.value.code == 1008
