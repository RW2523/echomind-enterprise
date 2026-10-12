"""The client's address and a per-address sign-in throttle (D-075).

Found on the pilot: behind the Law frontend every request reaches the backend from nginx, so the audit trail recorded
nginx's address for every event (4,789 of them), the intake links' per-address limit was one limit for everyone, and
there was no per-address sign-in throttle at all (only per-account lockout). nginx passes the client's address in
X-Real-IP; it is believed only from an address listed in LEGAL_TRUSTED_PROXIES, so a client cannot choose its own.
After LEGAL_SIGNIN_IP_MAX_FAILURES failed sign-in steps from one address within LEGAL_SIGNIN_IP_WINDOW_MIN minutes,
that address is refused (429) until the window passes; other addresses are not affected.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from conftest import LAW_TEST_ORIGIN, sign_in

A, B = {"X-Real-IP": "203.0.113.7"}, {"X-Real-IP": "198.51.100.23"}


@pytest.fixture
def thr(world, monkeypatch):
    from app.core.config import settings
    from app.legal import throttle
    monkeypatch.setattr(settings, "LEGAL_TRUSTED_PROXIES", "testclient")        # Starlette's TestClient connects as "testclient"
    monkeypatch.setattr(settings, "LEGAL_SIGNIN_IP_MAX_FAILURES", 3)
    throttle.reset()
    return world


def client(w):
    from fastapi.testclient import TestClient
    return TestClient(w["app"], base_url=LAW_TEST_ORIGIN)


def login(c, user, password, headers):
    return c.post("/api/auth/login", json={"username": user["username"], "password": password}, headers=headers)


def test_the_audit_trail_records_the_client_s_address_passed_by_the_proxy(thr, fresh_db):
    c = client(thr)
    assert login(c, thr["users"]["alice"], "wrong password!", A).status_code == 401
    with sqlite3.connect(fresh_db) as db:
        ips = [r[0] for r in db.execute("SELECT ip FROM audit_events WHERE action LIKE 'auth.%' ORDER BY seq")]
    assert ips[-1] == "203.0.113.7"


def test_the_header_is_believed_only_from_a_trusted_proxy(thr, fresh_db, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_TRUSTED_PROXIES", "")                    # nothing trusted: the header is the client's own
    c = client(thr)
    login(c, thr["users"]["alice"], "wrong password!", A)
    with sqlite3.connect(fresh_db) as db:
        assert db.execute("SELECT ip FROM audit_events WHERE action LIKE 'auth.%' ORDER BY seq DESC").fetchone()[0] == "testclient"


def test_an_address_with_too_many_failed_sign_ins_is_refused_and_others_are_not(thr, fresh_db):
    c, alice, bob = client(thr), thr["users"]["alice"], thr["users"]["bob"]
    for who in (alice, bob, alice):                                               # different accounts, one address
        assert login(c, who, "wrong password!", A).status_code == 401
    r = login(c, alice, alice["password"], A)                                     # even the right password, from there
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
    assert "Too many failed sign-ins" in r.json()["detail"]
    other = client(thr)
    assert login(other, bob, bob["password"], B).status_code == 200               # another address signs in
    with sqlite3.connect(fresh_db) as db:
        rows = db.execute("SELECT outcome, ip, details_json FROM audit_events WHERE action='auth.signin.throttled'").fetchall()
    assert rows and rows[0][:2] == ("deny", "203.0.113.7") and json.loads(rows[0][2]) == {"failures": 3}


def test_wrong_authenticator_codes_count_too(thr):
    c, alice = client(thr), thr["users"]["alice"]
    assert login(c, alice, alice["password"], A).json()["status"] == "mfa_required"
    from conftest import csrf_headers
    for _ in range(3):
        assert c.post("/api/auth/mfa/verify", json={"code": "000000"}, headers={**csrf_headers(c), **A}).status_code in (400, 401)
    assert c.post("/api/auth/mfa/verify", json={"code": "000000"}, headers={**csrf_headers(c), **A}).status_code == 429


def test_the_window_passes(thr, monkeypatch):
    from app.legal import throttle
    c, alice = client(thr), thr["users"]["alice"]
    for _ in range(3):
        login(c, alice, "wrong password!", A)
    assert login(c, alice, alice["password"], A).status_code == 429
    later = throttle._now() + 16 * 60
    monkeypatch.setattr(throttle, "_now", lambda: later)
    assert login(c, alice, alice["password"], A).status_code == 200


def test_successful_sign_ins_are_never_throttled(thr):
    for name in ("alice", "bob", "carol", "dave", "mp"):                          # five sign-ins from one office address
        c = client(thr)                                                           # (one code each: codes are single-use)
        c.headers.update(A)
        sign_in(c, thr["users"][name])


def test_behind_a_proxy_that_is_not_trusted_nothing_is_counted_per_address(thr, monkeypatch):
    """Otherwise every client would share the proxy's address, and one person's failures would lock out everyone."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_TRUSTED_PROXIES", "")
    c, alice = client(thr), thr["users"]["alice"]
    for _ in range(4):
        assert login(c, alice, "wrong password!", A).status_code == 401
    assert login(c, thr["users"]["bob"], thr["users"]["bob"]["password"], B).status_code == 200

