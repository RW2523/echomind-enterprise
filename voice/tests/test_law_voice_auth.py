"""S1.1 / SEC-01, D-013 — voice service in the Law edition.

The voice WebSocket accepts only a browser whose backend session is live and whose Origin is
allowed; it exchanges that session for short-lived on-behalf-of tokens used on every RAG call;
and it closes the socket (4401) as soon as the session ends. A small in-process HTTP server plays
the backend's /api/auth/me and /api/auth/obo-token endpoints.
"""
from __future__ import annotations

import http.server
import json
import threading
import time
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect

COOKIE = "__Host-echomind_s=sess-abc; __Host-echomind_csrf=csrf-xyz"


class _Backend(http.server.BaseHTTPRequestHandler):
    state = {"session_live": True, "obo_calls": 0, "expires_in": 300, "last_csrf": None}

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _session_ok(self):
        return self.state["session_live"] and "__Host-echomind_s=sess-abc" in (self.headers.get("Cookie") or "")

    def do_GET(self):
        if self.path == "/api/auth/me" and self._session_ok():
            return self._send(200, {"user": {"id": "usr_1", "username": "a.rahman"}})
        return self._send(401, {"detail": "Not authenticated"})

    def do_POST(self):
        self.state["last_csrf"] = self.headers.get("X-CSRF-Token")
        if self.path == "/api/auth/obo-token" and self._session_ok() and self.headers.get("X-CSRF-Token") == "csrf-xyz":
            self.state["obo_calls"] += 1
            return self._send(200, {"token": f"obo.t{self.state['obo_calls']}.sig", "expires_in": self.state["expires_in"]})
        return self._send(401, {"detail": "Not authenticated"})


@pytest.fixture
def backend(monkeypatch):
    _Backend.state.update(session_live=True, obo_calls=0, expires_in=300, last_csrf=None)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Backend)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from app import auth_check
    monkeypatch.setattr(auth_check, "_BACKEND", f"http://127.0.0.1:{srv.server_address[1]}")
    yield _Backend.state
    srv.shutdown()


def test_session_user_comes_from_the_backend(backend):
    from app.auth_check import law_session_user
    assert law_session_user(COOKIE)["username"] == "a.rahman"
    backend["session_live"] = False
    assert law_session_user(COOKIE) is None
    assert law_session_user("") is None


def test_backend_auth_mints_and_refreshes_obo_tokens(backend):
    from app.auth_check import BackendAuth
    ba = BackendAuth(COOKIE)
    assert ba.headers() == {"Authorization": "Bearer obo.t1.sig"}
    assert backend["last_csrf"] == "csrf-xyz"               # the browser's CSRF token is echoed back
    assert ba.headers() == {"Authorization": "Bearer obo.t1.sig"}  # cached while fresh
    ba._exp = time.time() + 30                                # within a minute of expiry → refresh
    assert ba.headers() == {"Authorization": "Bearer obo.t2.sig"}


def test_backend_auth_fails_closed(backend, monkeypatch):
    from app import auth_check
    ba = auth_check.BackendAuth(COOKIE)
    backend["session_live"] = False
    assert ba.refresh() is False and ba.headers() == {} and ba.still_valid() is False
    monkeypatch.setattr(auth_check, "_BACKEND", "http://127.0.0.1:9")   # nothing listening
    assert auth_check.BackendAuth(COOKIE).headers() == {}


def test_rag_calls_carry_the_obo_token():
    from app.session import OmniSessionA
    fake = SimpleNamespace(backend_auth=SimpleNamespace(headers=lambda: {"Authorization": "Bearer obo.x.y"}))
    assert OmniSessionA._backend_headers(fake) == {"Content-Type": "application/json", "Authorization": "Bearer obo.x.y"}
    assert OmniSessionA._backend_headers(SimpleNamespace(backend_auth=None)) == {"Content-Type": "application/json"}


# ── The WebSocket gate itself ────────────────────────────────────────────────────
@pytest.fixture
def law_server(monkeypatch, backend):
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    monkeypatch.setenv("VOICE_AUTH_ENABLED", "1")     # the import-time posture check must pass
    monkeypatch.setenv("VOICE_OFFLINE", "1")
    monkeypatch.setenv("VOICE_ALLOWED_ORIGINS", "https://law.example")
    monkeypatch.setenv("VOICE_WS_RECHECK_SEC", "0.1")
    from app import server

    class _StubSession:                      # the real session would load STT/TTS models
        def __init__(self, ws):
            self.ws, self.backend_auth = ws, None

        async def start(self, sid):
            await self.ws.send_text(json.dumps({"type": "hello", "obo": bool(self.backend_auth and self.backend_auth.headers())}))

        async def on_control(self, data):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(server, "OmniSessionA", _StubSession)
    from fastapi.testclient import TestClient
    return TestClient(server.app)


def _connect(client, cookie=COOKIE, origin="https://law.example"):
    headers = {"origin": origin}
    if cookie:
        headers["cookie"] = cookie
    return client.websocket_connect("/ws", headers=headers)


def test_law_voice_ws_refuses_anonymous_and_foreign_origins(law_server):
    for kwargs in ({"cookie": ""}, {"origin": "https://evil.example"}):
        with pytest.raises(WebSocketDisconnect) as exc:
            with _connect(law_server, **kwargs) as ws:
                ws.receive_text()
        assert exc.value.code == 1008


def test_law_voice_ws_accepts_a_live_session_and_closes_when_it_ends(law_server, backend):
    with _connect(law_server) as ws:
        hello = json.loads(ws.receive_text())
        assert hello == {"type": "hello", "obo": True}
        backend["session_live"] = False
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4401
