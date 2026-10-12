"""Single sign-on with the firm's identity provider (OIDC; SEC-01, P-011; decision D-076).

Off unless LEGAL_OIDC_ISSUER is set. The authorisation-code flow with PKCE, a state bound to the browser by its own
cookie, and a nonce. The ID token is checked: signature (the provider's published keys), issuer, audience, expiry,
nonce. It signs in only an account that already exists here (matched on a configured claim); nothing is created, and
roles and matter access stay this server's. This server's own two-step sign-in is still asked for unless the firm
chooses to rely on the provider's (LEGAL_OIDC_TRUST_MFA, with the token saying MFA was used). Every outcome is
audited without the token or its claims. A stand-in provider (a key made here) answers discovery, keys and tokens.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from conftest import LAW_TEST_ORIGIN, csrf_headers, current_code

ISSUER = "https://idp.example.test/tenant"
CLIENT = "echomind-law"


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class Provider:
    """The identity provider, standing in: discovery, keys, and a token endpoint that returns what the test sets."""
    def __init__(self):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.claims, self.posted, self.sign_with, self.alg = {}, [], self.key, "RS256"

    def jwks(self) -> dict:
        n = self.key.public_key().public_numbers()
        return {"keys": [{"kty": "RSA", "kid": "k1", "use": "sig", "alg": "RS256", "n": b64(n.n.to_bytes((n.n.bit_length() + 7) // 8, "big")),
                          "e": b64(n.e.to_bytes(3, "big"))}]}

    def id_token(self, claims: dict) -> str:
        head = b64(json.dumps({"alg": self.alg, "kid": "k1", "typ": "JWT"}).encode())
        body = b64(json.dumps(claims).encode())
        if self.alg == "none":
            return f"{head}.{body}."
        sig = self.sign_with.sign(f"{head}.{body}".encode(), padding.PKCS1v15(), hashes.SHA256())
        return f"{head}.{body}.{b64(sig)}"

    def get(self, url: str) -> dict:
        if url == f"{ISSUER}/.well-known/openid-configuration":
            return {"issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/authorize", "token_endpoint": f"{ISSUER}/token",
                    "jwks_uri": f"{ISSUER}/keys"}
        if url == f"{ISSUER}/keys":
            return self.jwks()
        raise AssertionError(url)

    def post(self, url: str, form: dict) -> dict:
        assert url == f"{ISSUER}/token"
        self.posted.append(form)
        return {"id_token": self.id_token(self.claims), "access_token": "not-used", "token_type": "Bearer"}


@pytest.fixture
def idp(world, monkeypatch, tmp_path):
    from app.core.config import settings
    from app.legal.identity import oidc
    secret = tmp_path / "oidc_secret"
    secret.write_text("s3cret-from-a-file\n")
    for k, v in {"LEGAL_OIDC_ISSUER": ISSUER, "LEGAL_OIDC_CLIENT_ID": CLIENT, "LEGAL_OIDC_CLIENT_SECRET_FILE": str(secret),
                 "LEGAL_OIDC_REDIRECT_URI": f"{LAW_TEST_ORIGIN}/api/auth/oidc/callback", "LEGAL_OIDC_LABEL": "Example Firm account",
                 "LEGAL_OIDC_USERNAME_CLAIM": "preferred_username", "LEGAL_OIDC_EMAIL_DOMAIN": "firm.example",
                 "LEGAL_OIDC_TRUST_MFA": False}.items():
        monkeypatch.setattr(settings, k, v)
    p = Provider()
    monkeypatch.setattr(oidc, "_get_json", p.get)
    monkeypatch.setattr(oidc, "_post_form", p.post)
    oidc.reset()
    world["idp"] = p
    return world


def browser(w):
    from fastapi.testclient import TestClient
    return TestClient(w["app"], base_url=LAW_TEST_ORIGIN, follow_redirects=False)


def start(c):
    r = c.get("/api/auth/oidc/start")
    assert r.status_code == 302, r.text
    return {k: v[0] for k, v in parse_qs(urlsplit(r.headers["location"]).query).items()}


def claims(q: dict, **over) -> dict:
    now = int(time.time())
    return {"iss": ISSUER, "aud": CLIENT, "sub": "idp-user-1", "preferred_username": "alice@firm.example", "nonce": q["nonce"],
            "iat": now, "exp": now + 300, **over}


def callback(c, q: dict, state: str | None = None):
    return c.get(f"/api/auth/oidc/callback?code=abc123&state={state or q['state']}")


def audit(db, action="auth.oidc"):
    with sqlite3.connect(db) as c:
        return c.execute(f"SELECT outcome, reason, actor_username, details_json FROM audit_events WHERE action='{action}' ORDER BY seq").fetchall()


def test_off_unless_the_firm_sets_its_provider(world):
    c = browser(world)
    assert "oidc" not in c.get("/api/auth/config").json()
    assert c.get("/api/auth/oidc/start").status_code == 404
    assert c.get("/api/auth/oidc/callback?code=x&state=y").status_code == 404


def test_the_sign_in_page_learns_the_provider_and_start_sends_the_browser_there_with_pkce(idp):
    c = browser(idp)
    assert c.get("/api/auth/config").json()["oidc"] == {"label": "Example Firm account"}
    q = start(c)
    assert q["response_type"] == "code" and q["client_id"] == CLIENT and q["scope"] == "openid email profile"
    assert q["redirect_uri"] == f"{LAW_TEST_ORIGIN}/api/auth/oidc/callback" and q["code_challenge_method"] == "S256"
    assert len(q["state"]) >= 32 and len(q["nonce"]) >= 32 and len(q["code_challenge"]) == 43
    assert any(ck.name == "__Host-echomind_oidc" for ck in c.cookies.jar)


def test_an_existing_account_signs_in_and_still_gives_this_server_s_code(idp, fresh_db):
    c = browser(idp)
    q = start(c)
    idp["idp"].claims = claims(q)
    r = callback(c, q)
    assert r.status_code == 302 and r.headers["location"] == "/signin?step=code"
    form = idp["idp"].posted[0]
    assert form["code"] == "abc123" and form["client_secret"] == "s3cret-from-a-file" and form["grant_type"] == "authorization_code"
    assert b64(hashlib.sha256(form["code_verifier"].encode()).digest()) == q["code_challenge"]       # PKCE
    assert c.get("/api/legal/matters").status_code == 401                                          # not yet: a partial session
    r = c.post("/api/auth/mfa/verify", json={"code": current_code(idp["users"]["alice"]["secret"])}, headers=csrf_headers(c))
    assert r.status_code == 200
    assert c.get("/api/legal/matters").status_code == 200
    rows = audit(fresh_db)
    assert rows[-1][:3] == ("allow", None, "alice") and "abc123" not in json.dumps(rows) and "idp-user-1" not in json.dumps(rows)


def test_with_the_provider_s_mfa_trusted_the_session_is_complete(idp, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_OIDC_TRUST_MFA", True)
    c = browser(idp)
    q = start(c)
    idp["idp"].claims = claims(q, amr=["pwd", "mfa"])
    assert callback(c, q).headers["location"] == "/home"
    assert c.get("/api/legal/matters").status_code == 200
    c2 = browser(idp)                                                                          # without "mfa" in amr: the code is still asked
    q2 = start(c2)
    idp["idp"].claims = claims(q2, amr=["pwd"])
    assert callback(c2, q2).headers["location"] == "/signin?step=code"


@pytest.mark.parametrize("case, reason", [
    ("wrong_state", "state"), ("other_browser", "state"), ("replayed", "state"), ("other_key", "id_token:signature"),
    ("alg_none", "id_token:algorithm"), ("wrong_issuer", "id_token:issuer"), ("wrong_audience", "id_token:audience"),
    ("expired", "id_token:expired"), ("wrong_nonce", "id_token:nonce"), ("unknown_user", "no_account"), ("disabled", "account_unavailable"),
])
def test_anything_wrong_is_refused_with_no_session_and_a_reason(idp, fresh_db, case, reason):
    from app.legal.identity import accounts
    c = browser(idp)
    q = start(c)
    p = idp["idp"]
    p.claims = claims(q)
    state = None
    if case == "wrong_state":
        state = "x" * 43
    elif case == "other_browser":
        c = browser(idp)                                                                        # no state cookie: login CSRF
    elif case == "replayed":
        assert callback(c, q).headers["location"] == "/signin?step=code"                        # used once, as meant …
        c.cookies.clear()                                                                       # (that sign-in set aside) …
    elif case == "other_key":
        p.sign_with = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    elif case == "alg_none":
        p.alg = "none"
    elif case == "wrong_issuer":
        p.claims = claims(q, iss="https://evil.example.test")
    elif case == "wrong_audience":
        p.claims = claims(q, aud="another-app")
    elif case == "expired":
        p.claims = claims(q, exp=int(time.time()) - 600)
    elif case == "wrong_nonce":
        p.claims = claims(q, nonce="n" * 43)
    elif case == "unknown_user":
        p.claims = claims(q, preferred_username="mallory@firm.example")
    elif case == "disabled":
        accounts.set_status("alice", "disabled")
    r = callback(c, q, state)
    assert r.status_code == 302 and r.headers["location"].startswith("/signin?oidc_error=")
    assert not any(ck.name == "__Host-echomind_s" and ck.value for ck in c.cookies.jar)
    assert c.get("/api/legal/matters").status_code == 401
    assert (audit(fresh_db) or [("", "")])[-1][:2] == ("deny", reason)
