"""S1.1 / SEC-01 — deny by default: every HTTP route and WebSocket in the Law app refuses an
anonymous caller, except a short public allow-list. Generated from the live route table, so a
newly added route is covered automatically."""
from __future__ import annotations

import re
import threading

import pytest
from fastapi import WebSocket
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


LAW_API = ("/api/auth/", "/api/legal/")


def test_every_http_route_denies_anonymous_callers(law_app, client):
    """Law routes answer 401 without a session; every other /api route is closed in the Law
    edition (S1.7, D-025) and answers 404 to everyone."""
    checked, leaks = 0, []
    for route in law_app.routes:
        if not isinstance(route, APIRoute) or route.path in _FASTAPI_DOCS:
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            if (method, route.path) in PUBLIC:
                continue
            r = client.request(method, _concrete(route.path), json={})
            checked += 1
            expected = 401 if route.path.startswith(LAW_API) or not route.path.startswith("/api/") else 404
            if r.status_code != expected:
                leaks.append(f"{method} {route.path} -> {r.status_code} (expected {expected})")
    assert checked > 60, checked
    assert not leaks, "routes reachable without a session:\n" + "\n".join(leaks)


def test_with_intake_links_on_only_the_link_routes_answer_anonymous_callers_and_only_to_a_real_link(law_app, client, monkeypatch):
    """S4.1 (D-054): enquirer links are the Law edition's only routes without a session. A made-up token gets 404,
    and every other route still answers 401 (Law) or 404 (closed)."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_INTAKE", True)
    monkeypatch.setattr(settings, "LEGAL_INTAKE_LINKS", True)
    link_routes, leaks = 0, []
    for route in law_app.routes:
        if not isinstance(route, APIRoute) or route.path in _FASTAPI_DOCS:
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            if (method, route.path) in PUBLIC:
                continue
            r = client.request(method, _concrete(route.path).replace("/x", "/" + "t" * 43), json={"answers": {}, "version": "x"})
            if route.path.startswith("/api/intake-link/"):
                link_routes += 1
                expected = 404
            else:
                expected = 401 if route.path.startswith(LAW_API) or not route.path.startswith("/api/") else 404
            if r.status_code != expected:
                leaks.append(f"{method} {route.path} -> {r.status_code} (expected {expected})")
    assert link_routes == 3
    assert not leaks, "\n".join(leaks)


def test_every_websocket_denies_anonymous_callers(law_app, client):
    ws_routes = [r for r in law_app.routes if isinstance(r, APIWebSocketRoute)]
    assert ws_routes
    for route in ws_routes:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(f"{WSS}{_concrete(route.path)}") as ws:
                ws.receive_text()
        # The only backend socket is live capture, closed in Law until S1.8 adds consent (4404).
        assert exc.value.code == (4404 if route.path == "/api/transcribe/ws" else 1008), route.path


@pytest.fixture
def probe_ws(law_app):
    """The WebSocket gate (`from_websocket` + `watch_websocket`) that consent-gated capture will use
    in S1.8, mounted on a probe socket so it stays tested while live capture is closed."""
    from app.legal.identity.principal import from_websocket, watch_websocket
    path = "/api/legal/_probe/ws"
    if not any(getattr(r, "path", None) == path for r in law_app.routes):
        @law_app.websocket(path)
        async def _probe(ws: WebSocket):
            principal = from_websocket(ws)
            if not principal:
                await ws.close(code=1008)
                return
            watcher = watch_websocket(ws, principal)
            await ws.accept()
            await ws.send_text("ready")
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        return
            finally:
                watcher.cancel()
    yield path
    law_app.router.routes[:] = [r for r in law_app.router.routes if getattr(r, "path", None) != path]


def test_the_socket_gate_accepts_a_session_and_closes_on_revocation(client, probe_ws, monkeypatch):
    from app.core.config import settings
    from app.legal.identity import sessions
    monkeypatch.setattr(settings, "LEGAL_WS_RECHECK_SEC", 0.1)
    sign_in(client, make_user("a.rahman", mfa=True))
    token = client.cookies.get("__Host-echomind_s")
    with client.websocket_connect(f"{WSS}{probe_ws}", headers={"Origin": "https://testserver"}) as ws:
        assert ws.receive_text() == "ready"
        threading.Timer(0.2, sessions.revoke, args=(token, "test")).start()
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4401


def test_the_socket_gate_refuses_a_foreign_origin(client, probe_ws):
    sign_in(client, make_user("a.rahman", mfa=True))
    with client.websocket_connect(f"{WSS}{probe_ws}", headers={"Origin": "https://testserver"}) as ok:
        assert ok.receive_text() == "ready"                      # same session, own origin: accepted
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"{WSS}{probe_ws}", headers={"Origin": "https://evil.example"}) as ws:
            ws.receive_text()
    assert exc.value.code == 1008
