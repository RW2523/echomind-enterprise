"""S1.3 / decision D-021 — onboarding hardening.

1. Binding an authenticator needs a one-time enrolment link issued by an administrator, so a
   leaked initial password is not enough to take over a new account.
2. Accounts created by an administrator must change their password at first sign-in.
3. Administrators can reset a lost authenticator (signs the person out, issues a new link).
"""
from __future__ import annotations

import time

from conftest import csrf_headers, current_code, make_user, sign_in


def _login(c, u):
    r = c.post("/api/auth/login", json={"username": u["username"], "password": u["password"]})
    assert r.status_code == 200, r.text
    return r.json()["status"]


def test_enrolment_needs_a_valid_link(client):
    from app.legal.identity import mfa
    u = make_user("n.new")
    other = make_user("o.other")
    assert _login(client, u) == "mfa_enrollment_required"
    assert client.post("/api/auth/mfa/enroll", json={}, headers=csrf_headers(client)).status_code == 403
    assert client.post("/api/auth/mfa/enroll", json={"enrollment_token": "guess"}, headers=csrf_headers(client)).status_code == 403
    theirs, _ = mfa.issue_enrollment_token(other["id"], created_by="admin")
    assert client.post("/api/auth/mfa/enroll", json={"enrollment_token": theirs}, headers=csrf_headers(client)).status_code == 403
    mine, _ = mfa.issue_enrollment_token(u["id"], created_by="admin")
    r = client.post("/api/auth/mfa/enroll", json={"enrollment_token": mine}, headers=csrf_headers(client))
    assert r.status_code == 200
    ok = client.post("/api/auth/mfa/confirm", json={"code": current_code(r.json()["secret"]), "enrollment_token": mine},
                     headers=csrf_headers(client))
    assert ok.status_code == 200


def test_links_are_single_use_and_expire(client, monkeypatch):
    from app.legal.identity import mfa
    u = make_user("n.new")
    link, expires_at = mfa.issue_enrollment_token(u["id"], created_by="admin")
    assert expires_at > time.time() + 70 * 3600                                   # default 72 h
    assert mfa.consume_enrollment_token(u["id"], link) is True
    assert mfa.consume_enrollment_token(u["id"], link) is False                   # used
    late, _ = mfa.issue_enrollment_token(u["id"], created_by="admin")
    real = time.time()
    monkeypatch.setattr(mfa.time, "time", lambda: real + 73 * 3600)
    assert mfa.enrollment_token_valid(u["id"], late) is False                     # expired


def test_a_new_link_replaces_older_ones(client):
    from app.legal.identity import mfa
    u = make_user("n.new")
    first, _ = mfa.issue_enrollment_token(u["id"], created_by="admin")
    second, _ = mfa.issue_enrollment_token(u["id"], created_by="admin")
    assert not mfa.enrollment_token_valid(u["id"], first) and mfa.enrollment_token_valid(u["id"], second)


def test_link_requirement_can_be_relaxed_by_firm_policy(client, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_ENROLLMENT_LINK_REQUIRED", False)
    u = make_user("n.new")
    _login(client, u)
    assert client.post("/api/auth/mfa/enroll", json={}, headers=csrf_headers(client)).status_code == 200


def test_admin_created_accounts_get_a_link_and_must_change_password(law_app, client):
    from fastapi.testclient import TestClient
    sign_in(client, make_user("firm.admin", role="admin", mfa=True))
    r = client.post("/api/auth/users", json={"username": "j.doe", "password": "an initial passphrase", "role": "attorney"},
                    headers=csrf_headers(client))
    assert r.status_code == 200, r.text
    link = r.json()["enrollment"]["token"]
    assert r.json()["enrollment"]["path"] == f"/signin?enrol={link}"
    new = TestClient(law_app, base_url="https://testserver")
    assert _login(new, {"username": "j.doe", "password": "an initial passphrase"}) == "mfa_enrollment_required"
    secret = new.post("/api/auth/mfa/enroll", json={"enrollment_token": link}, headers=csrf_headers(new)).json()["secret"]
    assert new.post("/api/auth/mfa/confirm", json={"code": current_code(secret), "enrollment_token": link},
                    headers=csrf_headers(new)).status_code == 200
    me = new.get("/api/auth/me").json()
    assert me["session"]["must_change_password"] is True
    blocked = new.get("/api/legal/matters")
    assert blocked.status_code == 403 and blocked.json()["detail"] == "password_change_required"
    assert new.post("/api/auth/password", json={"current_password": "an initial passphrase", "new_password": "a phrase only i know"},
                    headers=csrf_headers(new)).status_code == 200
    assert new.get("/api/legal/matters").status_code == 200
    assert new.get("/api/auth/me").json()["session"]["must_change_password"] is False


def test_admin_resets_a_lost_authenticator(law_app, client):
    from fastapi.testclient import TestClient
    sign_in(client, make_user("firm.admin", role="admin", mfa=True))
    victim = make_user("a.rahman", mfa=True)
    other = TestClient(law_app, base_url="https://testserver")
    sign_in(other, victim)
    r = client.post("/api/auth/users/a.rahman/reset-mfa", headers=csrf_headers(client))
    assert r.status_code == 200 and r.json()["enrollment"]["token"]
    assert other.get("/api/auth/me").status_code == 401                            # signed out everywhere
    assert _login(other, victim) == "mfa_enrollment_required"                       # old authenticator gone


def test_enrolment_link_is_refused_for_an_already_enrolled_account(client):
    sign_in(client, make_user("firm.admin", role="admin", mfa=True))
    make_user("a.rahman", mfa=True)
    assert client.post("/api/auth/users/a.rahman/enrollment-link", headers=csrf_headers(client)).status_code == 409


def test_bootstrap_admin_gets_a_private_enrolment_link(fresh_db, law_edition, monkeypatch):
    import os
    import stat
    from app.core import auth
    from app.core.config import settings
    from app.core.db import init_db
    from app.legal.identity import mfa
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_SECRET", "s" * 48)
    monkeypatch.setattr(settings, "AUTH_ADMIN_PASSWORD", "seven brave otters swim")
    init_db()
    auth.seed_admin()
    path = os.path.join(settings.DATA_DIR, "legal", "bootstrap_enrollment_token")
    token = open(path).read().strip()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert mfa.enrollment_token_valid(auth.get_user_by_username("admin")["id"], token)
