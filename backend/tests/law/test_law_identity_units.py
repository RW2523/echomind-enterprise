"""S1.1 / SEC-01 — identity building blocks: TOTP, password policy, server-side sessions,
recovery codes and on-behalf-of (OBO) tokens. Pure unit tests against a fresh Law database."""
from __future__ import annotations

import sqlite3
import time

import pytest

STRONG_SECRET = "k" * 48


@pytest.fixture
def law_db(fresh_db, law_edition, monkeypatch):
    from app.core.config import settings
    from app.core.db import init_db
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_SECRET", STRONG_SECRET)
    init_db()
    return fresh_db


@pytest.fixture
def user(law_db):
    from app.core import auth
    return auth.create_user("a.rahman", "correct horse battery staple", role="user")


# ── TOTP (RFC 6238, SHA-1, 30 s, 6 digits) ─────────────────────────────────────
RFC_SECRET_B32 = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"  # base32("12345678901234567890")


@pytest.mark.parametrize("t, code", [(59, "287082"), (1111111109, "081804"), (1111111111, "050471"),
                                     (1234567890, "005924"), (2000000000, "279037")])
def test_totp_matches_rfc6238_vectors(t, code):
    from app.legal.identity import mfa
    assert mfa.totp(RFC_SECRET_B32, t) == code


def test_totp_accepts_adjacent_step_and_rejects_replay():
    from app.legal.identity import mfa
    now = 1_700_000_000
    prev_code = mfa.totp(RFC_SECRET_B32, now - 30)
    step = mfa.verify_totp(RFC_SECRET_B32, prev_code, now=now, last_used_step=None)
    assert step == (now - 30) // 30
    assert mfa.verify_totp(RFC_SECRET_B32, prev_code, now=now, last_used_step=step) is None  # replay
    assert mfa.verify_totp(RFC_SECRET_B32, mfa.totp(RFC_SECRET_B32, now - 120), now=now, last_used_step=None) is None
    assert mfa.verify_totp(RFC_SECRET_B32, "12345", now=now, last_used_step=None) is None
    assert mfa.verify_totp(RFC_SECRET_B32, "abcdef", now=now, last_used_step=None) is None


def test_generated_secrets_are_base32_and_unique():
    from app.legal.identity import mfa
    a, b = mfa.new_secret(), mfa.new_secret()
    assert a != b and len(a) == 32 and set(a) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
    uri = mfa.otpauth_uri(a, "a.rahman")
    assert uri.startswith("otpauth://totp/EchoMind%20Law:a.rahman?") and f"secret={a}" in uri


# ── Password policy ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("pw, problem", [
    ("short1!", "length"),
    ("a.rahman-a.rahman", "username"),
    ("passwordpassword", "common"),
    ("aaaaaaaaaaaaaaaa", "repetitive"),
])
def test_password_policy_rejects(pw, problem):
    from app.legal.identity import passwords
    assert problem in passwords.problems(pw, "a.rahman")


def test_password_policy_accepts_a_passphrase():
    from app.legal.identity import passwords
    assert passwords.problems("correct horse battery staple", "a.rahman") == []


# ── Sessions ────────────────────────────────────────────────────────────────────
def test_session_roundtrip_and_storage_is_hashed(user, law_db):
    from app.legal.identity import sessions
    token, csrf = sessions.create(user["id"], mfa_verified=True, ip="10.0.0.5", user_agent="pytest")
    s = sessions.lookup(token)
    assert s and s["user_id"] == user["id"] and s["mfa_verified"] and sessions.csrf_ok(s, csrf)
    assert not sessions.csrf_ok(s, "forged")
    with sqlite3.connect(law_db) as c:
        stored = [r[0] for r in c.execute("SELECT token_hash FROM sessions")]
    assert token not in stored and len(stored[0]) == 64


def test_session_idle_and_absolute_expiry(user, monkeypatch):
    from app.legal.identity import sessions
    token, _ = sessions.create(user["id"], mfa_verified=True)
    real = time.time()
    monkeypatch.setattr(sessions, "_now", lambda: real + 31 * 60)          # idle 30 min
    assert sessions.lookup(token) is None
    token2, _ = sessions.create(user["id"], mfa_verified=True)
    t = real
    for _ in range(25):                                                     # active every 25 min …
        t += 25 * 60
        monkeypatch.setattr(sessions, "_now", lambda t=t: t)
        if sessions.lookup(token2) is None:
            break
    assert (t - real) <= 10 * 3600 + 25 * 60                                # … but never past 10 h


def test_revoke_and_rotate(user):
    from app.legal.identity import sessions
    token, _ = sessions.create(user["id"], mfa_verified=False)
    new_token, new_csrf = sessions.rotate(token, mfa_verified=True)
    assert sessions.lookup(token) is None and sessions.lookup(new_token)["mfa_verified"]
    other, _ = sessions.create(user["id"], mfa_verified=True)
    assert sessions.revoke_all_for_user(user["id"], reason="password_changed", keep_token=new_token) == 1
    assert sessions.lookup(other) is None and sessions.lookup(new_token)


def test_disabled_user_sessions_are_dead(user):
    from app.legal.identity import sessions, accounts
    token, _ = sessions.create(user["id"], mfa_verified=True)
    accounts.set_status(user["username"], "disabled")
    assert sessions.lookup(token) is None


# ── Lockout ─────────────────────────────────────────────────────────────────────
def test_lockout_after_threshold_and_reset_on_success(user, monkeypatch):
    from app.legal.identity import accounts
    for _ in range(4):
        accounts.register_failure(user["id"])
    assert not accounts.is_locked(accounts.get(user["id"]))
    accounts.register_failure(user["id"])
    assert accounts.is_locked(accounts.get(user["id"]))
    accounts.register_success(user["id"])
    assert not accounts.is_locked(accounts.get(user["id"]))


# ── Recovery codes ──────────────────────────────────────────────────────────────
def test_recovery_codes_are_single_use_and_hashed(user, law_db):
    from app.legal.identity import mfa
    codes = mfa.issue_recovery_codes(user["id"])
    assert len(codes) == 10 and len(set(codes)) == 10
    assert mfa.use_recovery_code(user["id"], codes[0]) is True
    assert mfa.use_recovery_code(user["id"], codes[0]) is False
    assert mfa.use_recovery_code(user["id"], "nope-nope") is False
    with sqlite3.connect(law_db) as c:
        stored = " ".join(r[0] for r in c.execute("SELECT code_hash FROM mfa_recovery_codes"))
    assert codes[1] not in stored


# ── On-behalf-of tokens (voice → backend) ───────────────────────────────────────
def test_obo_token_is_bound_to_audience_and_live_session(user, monkeypatch):
    from app.legal.identity import obo, sessions
    token, _ = sessions.create(user["id"], mfa_verified=True)
    t = obo.issue(sessions.lookup(token), audience="voice")
    p = obo.verify(t, audience="voice")
    assert p and p["user_id"] == user["id"]
    assert obo.verify(t, audience="other") is None
    assert obo.verify(t + "x", audience="voice") is None
    sessions.revoke(token, reason="logout")
    assert obo.verify(t, audience="voice") is None                         # revocation propagates


def test_obo_token_expires(user, monkeypatch):
    from app.legal.identity import obo, sessions
    token, _ = sessions.create(user["id"], mfa_verified=True)
    t = obo.issue(sessions.lookup(token), audience="voice")
    real = time.time()
    monkeypatch.setattr(obo, "_now", lambda: real + 301)
    assert obo.verify(t, audience="voice") is None


def test_obo_tokens_are_not_legacy_jwts(user):
    """A legacy HS256 JWT signed with the same AUTH_SECRET must not pass as an OBO token."""
    from app.core import auth
    from app.legal.identity import obo
    legacy = auth.make_token(user["id"], "user", user["username"])
    assert obo.verify(legacy, audience="voice") is None
