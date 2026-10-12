"""Single sign-on with the firm's identity provider (OIDC; SEC-01, P-011; decision D-076). Off unless LEGAL_OIDC_ISSUER.

The authorisation-code flow with PKCE (S256). `start` makes a state, a nonce and a code verifier, keeps them here for
ten minutes under the state, and binds them to the browser with a cookie of their own (a callback from another browser
is refused: login CSRF). `finish` uses a state once, exchanges the code at the provider's token endpoint (with the
client secret from LEGAL_OIDC_CLIENT_SECRET_FILE when set), and checks the ID token:
- the signature, RS256 or ES256, with the provider's published keys (refetched once for an unknown key id);
- the issuer, the audience (and azp when there are several), expiry and issue time (60 s leeway), and the nonce.
It then finds the account already here whose username is the LEGAL_OIDC_USERNAME_CLAIM value (lower-cased; with
LEGAL_OIDC_EMAIL_DOMAIN, "@<domain>" is taken off). Nothing is created: roles and matter access stay this server's.
The provider's MFA counts only with LEGAL_OIDC_TRUST_MFA and an `amr` saying it was used; otherwise this server's own
two-step sign-in is asked for, as after a password. The backend reaches the provider over the network: that needs the
firm's approval and an exception in the egress lock (D-071).
"""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import urllib.parse
import urllib.request
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from ...core.config import settings

STATE_COOKIE = "__Host-echomind_oidc"
STATE_TTL = 600
LEEWAY = 60
MFA_AMR = frozenset({"mfa", "otp", "hwk", "swk", "fpt", "face", "sms"})
_MAX_PENDING = 5000
_lock = threading.Lock()
_pending: dict[str, dict] = {}
_cache: dict[str, tuple[float, dict]] = {}


class Refused(Exception):
    """A sign-in that cannot go on; `reason` is what the audit trail records (never a claim or a token)."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def enabled() -> bool:
    return bool((settings.LEGAL_OIDC_ISSUER or "").strip() and (settings.LEGAL_OIDC_CLIENT_ID or "").strip())


def reset() -> None:
    with _lock:
        _pending.clear()
        _cache.clear()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


# ── The provider, over the network (tests replace these two) ─────────────────────
def _get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=10) as r:
        return json.loads(r.read(1024 * 1024))


def _post_form(url: str, form: dict) -> dict:
    req = urllib.request.Request(url, data=urllib.parse.urlencode(form).encode(), method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read(1024 * 1024))


def _cached(url: str, fresh: bool = False) -> dict:
    now = time.monotonic()
    with _lock:
        hit = _cache.get(url)
    if hit and not fresh and now - hit[0] < 600:
        return hit[1]
    data = _get_json(url)
    with _lock:
        _cache[url] = (now, data)
    return data


def _discovery() -> dict:
    issuer = settings.LEGAL_OIDC_ISSUER.strip().rstrip("/")
    d = _cached(issuer + "/.well-known/openid-configuration")
    if d.get("issuer", "").rstrip("/") != issuer:
        raise Refused("discovery")
    return d


# ── Starting ─────────────────────────────────────────────────────────────────────
def start() -> tuple[str, str]:
    """(where to send the browser, the value of its state cookie)."""
    d = _discovery()
    state, nonce, verifier, browser = (secrets.token_urlsafe(32) for _ in range(4))
    challenge = _b64e(hashlib.sha256(verifier.encode()).digest())
    with _lock:
        now = time.monotonic()
        for k in [k for k, v in _pending.items() if now - v["at"] > STATE_TTL]:
            _pending.pop(k)
        while len(_pending) >= _MAX_PENDING:
            _pending.pop(next(iter(_pending)))
        _pending[state] = {"nonce": nonce, "verifier": verifier, "browser": hashlib.sha256(browser.encode()).hexdigest(), "at": now}
    query = {"response_type": "code", "client_id": settings.LEGAL_OIDC_CLIENT_ID, "redirect_uri": settings.LEGAL_OIDC_REDIRECT_URI,
             "scope": settings.LEGAL_OIDC_SCOPES, "state": state, "nonce": nonce, "code_challenge": challenge, "code_challenge_method": "S256"}
    return d["authorization_endpoint"] + ("&" if "?" in d["authorization_endpoint"] else "?") + urllib.parse.urlencode(query), browser


# ── Finishing ────────────────────────────────────────────────────────────────────
def _take(state: str, browser_cookie: Optional[str]) -> dict:
    with _lock:
        p = _pending.pop(state or "", None)                     # used once, whatever happens next
    if p is None or time.monotonic() - p["at"] > STATE_TTL or not browser_cookie or \
            not secrets.compare_digest(p["browser"], hashlib.sha256(browser_cookie.encode()).hexdigest()):
        raise Refused("state")
    return p


def _key(jwks_uri: str, kid: Optional[str], alg: str):
    for fresh in (False, True):
        for k in _cached(jwks_uri, fresh=fresh).get("keys", []):
            if kid is not None and k.get("kid") != kid:
                continue
            if alg == "RS256" and k.get("kty") == "RSA":
                return rsa.RSAPublicNumbers(int.from_bytes(_b64d(k["e"]), "big"), int.from_bytes(_b64d(k["n"]), "big")).public_key()
            if alg == "ES256" and k.get("kty") == "EC" and k.get("crv") == "P-256":
                return ec.EllipticCurvePublicNumbers(int.from_bytes(_b64d(k["x"]), "big"), int.from_bytes(_b64d(k["y"]), "big"),
                                                     ec.SECP256R1()).public_key()
    raise Refused("id_token:key")


def verify_id_token(token: str, nonce: str) -> dict:
    try:
        head_b, body_b, sig_b = token.split(".")
        head, claims, sig = json.loads(_b64d(head_b)), json.loads(_b64d(body_b)), _b64d(sig_b)
    except (ValueError, TypeError):
        raise Refused("id_token:format") from None
    alg = head.get("alg")
    if alg not in ("RS256", "ES256"):
        raise Refused("id_token:algorithm")                     # never "none", never a shared-secret HS*
    key = _key(_discovery()["jwks_uri"], head.get("kid"), alg)
    signed = f"{head_b}.{body_b}".encode()
    try:
        if alg == "RS256":
            key.verify(sig, signed, padding.PKCS1v15(), hashes.SHA256())
        else:
            if len(sig) != 64:
                raise InvalidSignature()
            key.verify(encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big")), signed, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        raise Refused("id_token:signature") from None
    now = time.time()
    aud = claims.get("aud")
    auds = aud if isinstance(aud, list) else [aud]
    if str(claims.get("iss", "")).rstrip("/") != settings.LEGAL_OIDC_ISSUER.strip().rstrip("/"):
        raise Refused("id_token:issuer")
    if settings.LEGAL_OIDC_CLIENT_ID not in auds or (len(auds) > 1 and claims.get("azp") != settings.LEGAL_OIDC_CLIENT_ID):
        raise Refused("id_token:audience")
    if not isinstance(claims.get("exp"), (int, float)) or claims["exp"] < now - LEEWAY:
        raise Refused("id_token:expired")
    if isinstance(claims.get("iat"), (int, float)) and claims["iat"] > now + LEEWAY:
        raise Refused("id_token:issued_in_future")
    if not isinstance(claims.get("nonce"), str) or not secrets.compare_digest(claims["nonce"], nonce):
        raise Refused("id_token:nonce")
    return claims


def username_of(claims: dict) -> Optional[str]:
    value = claims.get(settings.LEGAL_OIDC_USERNAME_CLAIM)
    if not isinstance(value, str) or not value.strip():
        return None
    name = value.strip().lower()
    domain = (settings.LEGAL_OIDC_EMAIL_DOMAIN or "").strip().lower()
    if domain and name.endswith("@" + domain):
        name = name[: -len(domain) - 1]
    return name


def mfa_by_provider(claims: dict) -> bool:
    amr = claims.get("amr")
    return bool(settings.LEGAL_OIDC_TRUST_MFA and isinstance(amr, list) and MFA_AMR & {str(a).lower() for a in amr})


def finish(code: str, state: str, browser_cookie: Optional[str]) -> dict:
    """The verified claims of a sign-in, or Refused."""
    p = _take(state, browser_cookie)
    if not code:
        raise Refused("code")
    form = {"grant_type": "authorization_code", "code": code, "redirect_uri": settings.LEGAL_OIDC_REDIRECT_URI,
            "client_id": settings.LEGAL_OIDC_CLIENT_ID, "code_verifier": p["verifier"]}
    if settings.LEGAL_OIDC_CLIENT_SECRET_FILE:
        with open(settings.LEGAL_OIDC_CLIENT_SECRET_FILE) as f:
            form["client_secret"] = f.read().strip()
    try:
        tokens = _post_form(_discovery()["token_endpoint"], form)
    except Refused:
        raise
    except Exception:
        raise Refused("token_exchange") from None
    if not isinstance(tokens.get("id_token"), str):
        raise Refused("token_exchange")
    return verify_id_token(tokens["id_token"], p["nonce"])
