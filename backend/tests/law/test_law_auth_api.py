"""S1.1 / SEC-01 — sign-in, MFA, cookies, CSRF, lockout, sessions and admin controls over HTTP,
against the real app in the Law edition."""
from __future__ import annotations

import sqlite3

from conftest import csrf_headers, current_code, make_user, sign_in

GENERIC = "Sign-in failed"


def _set_cookie_headers(r) -> dict:
    out = {}
    for raw in r.headers.get_list("set-cookie"):
        name = raw.split("=", 1)[0]
        out[name] = raw.lower()
    return out


# ── Sign-in and MFA ──────────────────────────────────────────────────────────────
def test_login_sets_hardened_cookies_and_returns_no_token(client):
    u = make_user("p.okafor")
    r = client.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "mfa_enrollment_required" and "token" not in body and "user" not in body
    cookies = _set_cookie_headers(r)
    s, c = cookies["__Host-echomind_s"], cookies["__Host-echomind_csrf"]
    for attr in ("secure", "samesite=strict", "path=/"):
        assert attr in s and attr in c
    assert "httponly" in s and "httponly" not in c


def test_partial_session_cannot_use_the_app(client):
    u = make_user("p.okafor")
    client.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    assert client.get("/api/docs/list").status_code == 401
    assert client.get("/api/auth/me").status_code == 401


def test_first_login_enrols_mfa_and_rotates_the_session(client, fresh_db):
    from app.legal.identity import mfa
    u = make_user("p.okafor")
    link, _ = mfa.issue_enrollment_token(u["id"], created_by="test-admin")
    client.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    partial = client.cookies.get("__Host-echomind_s")
    r = client.post("/api/auth/mfa/enroll", json={"enrollment_token": link}, headers=csrf_headers(client))
    assert r.status_code == 200
    secret, uri = r.json()["secret"], r.json()["otpauth_uri"]
    assert uri.startswith("otpauth://totp/EchoMind%20Law:p.okafor?")
    bad = client.post("/api/auth/mfa/confirm", json={"code": "000000", "enrollment_token": link}, headers=csrf_headers(client))
    assert bad.status_code == 401
    r = client.post("/api/auth/mfa/confirm", json={"code": current_code(secret), "enrollment_token": link}, headers=csrf_headers(client))
    assert r.status_code == 200 and len(r.json()["recovery_codes"]) == 10
    assert client.cookies.get("__Host-echomind_s") != partial                       # rotated
    assert client.get("/api/auth/me").json()["user"]["username"] == "p.okafor"
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT revoke_reason FROM sessions WHERE revoked_at IS NOT NULL").fetchone() == ("rotated",)


def test_enrolled_user_needs_a_current_code(client):
    u = make_user("a.rahman", mfa=True)
    r = client.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    assert r.json()["status"] == "mfa_required"
    assert client.post("/api/auth/mfa/verify", json={"code": current_code(u["secret"], -5)},
                       headers=csrf_headers(client)).status_code == 401
    code = current_code(u["secret"])
    assert client.post("/api/auth/mfa/verify", json={"code": code}, headers=csrf_headers(client)).status_code == 200
    assert client.get("/api/auth/me").status_code == 200
    # the same code cannot be replayed on a second sign-in
    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    assert client.post("/api/auth/mfa/verify", json={"code": code}, headers=csrf_headers(client)).status_code == 401


def test_recovery_code_signs_in_once(client):
    from app.legal.identity import mfa
    u = make_user("a.rahman", mfa=True)
    codes = mfa.issue_recovery_codes(u["id"])
    for expected in (200, 401):
        client.post("/api/auth/logout")
        client.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
        r = client.post("/api/auth/mfa/verify", json={"code": codes[3]}, headers=csrf_headers(client))
        assert r.status_code == expected


def test_mfa_can_be_optional_by_firm_policy(client, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_MFA_REQUIRED", False)
    u = make_user("p.okafor")
    r = client.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    assert r.json()["status"] == "ok" and r.json()["user"]["username"] == "p.okafor"
    assert client.get("/api/auth/me").status_code == 200


# ── Failure handling ─────────────────────────────────────────────────────────────
def test_unknown_user_and_wrong_password_look_identical(client):
    make_user("a.rahman", mfa=True)
    r1 = client.post("/api/auth/login", json={"username": "nobody", "password": "whatever-whatever"})
    r2 = client.post("/api/auth/login", json={"username": "a.rahman", "password": "wrong password here"})
    assert r1.status_code == r2.status_code == 401
    assert r1.json() == r2.json() and GENERIC in r1.json()["detail"]


def test_lockout_after_repeated_failures(client):
    u = make_user("a.rahman", mfa=True)
    for _ in range(5):
        client.post("/api/auth/login", json={"username": u["username"], "password": "wrong password here"})
    r = client.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    assert r.status_code == 401 and GENERIC in r.json()["detail"]


def test_mfa_guessing_counts_towards_lockout(client):
    u = make_user("a.rahman", mfa=True)
    client.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    for _ in range(5):
        client.post("/api/auth/mfa/verify", json={"code": "000000"}, headers=csrf_headers(client))
    from app.legal.identity import accounts
    assert accounts.is_locked(accounts.get(u["id"]))


def test_disabled_user_cannot_sign_in(client):
    from app.legal.identity import accounts
    u = make_user("a.rahman", mfa=True)
    accounts.set_status("a.rahman", "disabled")
    r = client.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    assert r.status_code == 401


def test_legacy_jwt_cookie_is_ignored(client):
    from app.core import auth
    u = make_user("a.rahman", mfa=True)
    client.cookies.set("echomind_token", auth.make_token(u["id"], "admin", u["username"]))
    assert client.get("/api/docs/list").status_code == 401


# ── CSRF, logout, sessions ───────────────────────────────────────────────────────
def test_mutations_require_the_csrf_token(client):
    sign_in(client, make_user("a.rahman", mfa=True))
    assert client.post("/api/chat/create", json={"title": "x"}).status_code == 403
    assert client.post("/api/chat/create", json={"title": "x"}, headers={"X-CSRF-Token": "forged"}).status_code == 403
    assert client.post("/api/chat/create", json={"title": "x"}, headers=csrf_headers(client)).status_code == 200


def test_logout_revokes_server_side(client):
    sign_in(client, make_user("a.rahman", mfa=True))
    stolen = client.cookies.get("__Host-echomind_s")
    client.post("/api/auth/logout")
    client.cookies.set("__Host-echomind_s", stolen)
    assert client.get("/api/auth/me").status_code == 401


def test_list_and_revoke_own_sessions(law_app, client):
    from fastapi.testclient import TestClient
    u = make_user("a.rahman", mfa=True)
    sign_in(client, u)
    laptop = TestClient(law_app, base_url="https://testserver")
    from app.legal.identity import mfa
    laptop.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    codes = mfa.issue_recovery_codes(u["id"])
    assert laptop.post("/api/auth/mfa/verify", json={"code": codes[0]}, headers=csrf_headers(laptop)).status_code == 200
    listed = client.get("/api/auth/sessions").json()["sessions"]
    assert len(listed) == 2 and sum(s["current"] for s in listed) == 1
    other = next(s for s in listed if not s["current"])
    assert client.delete(f"/api/auth/sessions/{other['id']}", headers=csrf_headers(client)).status_code == 200
    assert laptop.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me").status_code == 200


def test_password_change_enforces_policy_and_signs_out_elsewhere(law_app, client):
    from fastapi.testclient import TestClient
    from app.legal.identity import sessions
    u = make_user("a.rahman", mfa=True)
    sign_in(client, u)
    elsewhere, _ = sessions.create(u["id"], mfa_verified=True)
    weak = client.post("/api/auth/password", json={"current_password": u["password"], "new_password": "short"},
                       headers=csrf_headers(client))
    assert weak.status_code == 400 and "at least 12 characters" in weak.json()["detail"]
    wrong = client.post("/api/auth/password", json={"current_password": "not it at all", "new_password": "a brand new passphrase"},
                        headers=csrf_headers(client))
    assert wrong.status_code == 400
    ok = client.post("/api/auth/password", json={"current_password": u["password"], "new_password": "a brand new passphrase"},
                     headers=csrf_headers(client))
    assert ok.status_code == 200 and ok.json()["revoked_sessions"] == 1
    assert sessions.lookup(elsewhere) is None and client.get("/api/auth/me").status_code == 200


# ── Admin controls ───────────────────────────────────────────────────────────────
def test_admin_can_disable_and_sign_out_a_user(law_app, client):
    from fastapi.testclient import TestClient
    admin = make_user("firm.admin", role="admin", mfa=True)
    target = make_user("a.rahman", mfa=True)
    sign_in(client, admin)
    victim = TestClient(law_app, base_url="https://testserver")
    sign_in(victim, target)
    assert client.post("/api/auth/users/a.rahman/revoke-sessions", headers=csrf_headers(client)).status_code == 200
    assert victim.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/users/a.rahman/disable", headers=csrf_headers(client)).status_code == 200
    r = victim.post("/api/auth/login", json={"username": "a.rahman", "password": target["password"]})
    assert r.status_code == 401
    assert client.post("/api/auth/users/a.rahman/enable", headers=csrf_headers(client)).status_code == 200


def test_non_admin_cannot_manage_users(client):
    sign_in(client, make_user("a.rahman", mfa=True))
    make_user("b.berg")
    assert client.post("/api/auth/users/b.berg/disable", headers=csrf_headers(client)).status_code == 403


def test_new_users_must_meet_the_password_policy(client):
    sign_in(client, make_user("firm.admin", role="admin", mfa=True))
    weak = client.post("/api/auth/users", json={"username": "j.doe", "password": "password1", "role": "user"},
                       headers=csrf_headers(client))
    assert weak.status_code == 400
    ok = client.post("/api/auth/users", json={"username": "j.doe", "password": "a perfectly long phrase", "role": "user"},
                     headers=csrf_headers(client))
    assert ok.status_code == 200


# ── Service-to-service (voice) tokens ────────────────────────────────────────────
def test_obo_token_works_only_on_voice_rag_routes(law_app, client):
    from fastapi.testclient import TestClient
    sign_in(client, make_user("a.rahman", mfa=True))
    r = client.post("/api/auth/obo-token", json={"audience": "voice"}, headers=csrf_headers(client))
    assert r.status_code == 200 and r.json()["expires_in"] == 300
    bearer = {"Authorization": f"Bearer {r.json()['token']}"}
    svc = TestClient(law_app, base_url="https://testserver")  # no cookies: a service
    assert svc.get("/api/docs/list", headers=bearer).status_code == 401
    assert svc.post("/api/chat/ask-voice", json={}, headers=bearer).status_code == 422  # authenticated; body invalid
    client.post("/api/auth/logout")
    assert svc.post("/api/chat/ask-voice", json={}, headers=bearer).status_code == 401  # session gone → token dead


def test_obo_token_cannot_mint_more_tokens(law_app, client):
    from fastapi.testclient import TestClient
    sign_in(client, make_user("a.rahman", mfa=True))
    tok = client.post("/api/auth/obo-token", json={"audience": "voice"}, headers=csrf_headers(client)).json()["token"]
    svc = TestClient(law_app, base_url="https://testserver")
    assert svc.post("/api/auth/obo-token", json={"audience": "voice"},
                    headers={"Authorization": f"Bearer {tok}"}).status_code == 401


def test_client_error_reports_need_a_session(client):
    assert client.post("/api/client-error", json={"message": "x"}).status_code == 401


def test_config_tells_the_frontend_which_edition_to_load(client, monkeypatch):
    assert client.get("/api/auth/config").json() == {"auth_enabled": True, "edition": "law"}
    monkeypatch.delenv("ECHOMIND_EDITION")
    assert client.get("/api/auth/config").json()["edition"] == "standard"
