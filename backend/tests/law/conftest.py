"""Shared fixtures for the EchoMind Law test suite (backend/tests/law).

Same convention as the legacy suite: ECHOMIND_DATA_DIR is pointed at a throwaway directory before
any `app` module is imported, so nothing here can touch a real /data volume.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_test_"))

import pytest  # noqa: E402


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Point the backend at a brand-new, empty SQLite file for the duration of one test."""
    from app.core.config import settings

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(settings, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(settings, "DB_PATH", str(data_dir / "echomind.sqlite"))
    return data_dir / "echomind.sqlite"


@pytest.fixture
def law_edition(monkeypatch):
    monkeypatch.setenv("ECHOMIND_EDITION", "law")


@pytest.fixture
def legacy_edition(monkeypatch):
    monkeypatch.delenv("ECHOMIND_EDITION", raising=False)


# ── In-process Law app (S1.1+) ───────────────────────────────────────────────────
LAW_TEST_SECRET = "t" * 48
LAW_TEST_ORIGIN = "https://testserver"


def write_master_key(path) -> str:
    import base64
    import secrets
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(base64.b64encode(secrets.token_bytes(32)).decode() + "\n")
    return str(path)


@pytest.fixture
def law_app(fresh_db, law_edition, monkeypatch):
    """The real FastAPI app in the Law edition against a fresh database.

    Settings are patched before `app.main` is (possibly first) imported, so the import-time
    posture check passes. Requests use https://testserver because session cookies are Secure.
    """
    from app.core.config import settings
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTH_SECRET", LAW_TEST_SECRET)
    monkeypatch.setattr(settings, "CORS_ALLOW_ORIGINS", LAW_TEST_ORIGIN)
    monkeypatch.setattr(settings, "AUTH_ADMIN_PASSWORD", "")
    monkeypatch.setattr(settings, "LEGAL_MFA_REQUIRED", True)
    # D-014: the master key lives outside the data folder, readable by the owner only.
    key = fresh_db.parent.parent / "master.key"
    write_master_key(key)
    monkeypatch.setattr(settings, "LEGAL_MASTER_KEY_FILE", str(key))
    from app.core.db import init_db
    init_db()
    import app.main as main
    return main.app


@pytest.fixture
def client(law_app):
    # Not used as a context manager on purpose: that would run the app lifespan (LLM/ASR warm-up and
    # the CUDA-fault watchdog, whose blocking thread keeps the event loop from shutting down).
    from fastapi.testclient import TestClient
    return TestClient(law_app, base_url=LAW_TEST_ORIGIN)


def make_user(username, password="correct horse battery staple", role="user", mfa=False, firm_roles=()):
    """Create a Law user; with mfa=True also enrol TOTP and return its secret.
    role="admin" also grants the firm_admin firm role (Law administration, S1.2)."""
    from app.core import auth
    from app.legal.identity import accounts
    from app.legal.identity import mfa as mfamod
    u = auth.create_user(username, password, role=role)
    for r in set(firm_roles) | ({"firm_admin"} if role == "admin" else set()):
        accounts.grant_role(u["id"], r)
    secret = None
    if mfa:
        secret = mfamod.begin_enrollment(u["id"])
        from app.core.db import get_conn
        with get_conn() as conn:
            conn.execute("UPDATE users SET mfa_enabled=1 WHERE id=?", (u["id"],))
            conn.commit()
    return {**u, "password": password, "secret": secret}


def csrf_headers(c) -> dict:
    return {"X-CSRF-Token": c.cookies.get("__Host-echomind_csrf") or ""}


def current_code(secret: str, offset_steps: int = 0) -> str:
    import time
    from app.legal.identity import mfa
    return mfa.totp(secret, time.time() + 30 * offset_steps)


@pytest.fixture(autouse=True)
def _fresh_signin_throttle():
    """D-075: the per-address sign-in counts live in the process; every test starts with none."""
    try:
        from app.legal import throttle
    except ImportError:
        yield
        return
    throttle.reset()
    yield
    throttle.reset()


def sign_in(c, user) -> None:
    """Full sign-in (password + TOTP) for an MFA-enrolled user."""
    r = c.post("/api/auth/login", json={"username": user["username"], "password": user["password"]})
    assert r.status_code == 200 and r.json()["status"] == "mfa_required", r.text
    r = c.post("/api/auth/mfa/verify", json={"code": current_code(user["secret"])}, headers=csrf_headers(c))
    assert r.status_code == 200, r.text


# ── The fictional T01 world (S1.2+) ──────────────────────────────────────────────
from datetime import datetime, timedelta, timezone  # noqa: E402


def _ts(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


@pytest.fixture
def world(law_app):
    """Client Alpha (M-101, M-114) and Beta (M-202 highly restricted, M-215). alice leads M-101/M-114;
    bob leads M-202/M-215; carol is read-only on M-101 (her M-215 membership was revoked); dave's
    M-202 membership expired; mp and mp2 are managing partners; admin (admin2) is firm_admin."""
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    org = default_org_id()
    u = {n: make_user(n, mfa=True, firm_roles=r) for n, r in {
        "alice": ["attorney"], "bob": ["attorney"], "carol": ["paralegal"], "dave": ["attorney"],
        "mp": ["managing_partner"], "mp2": ["managing_partner"]}.items()}
    u["admin"] = make_user("admin2", role="admin", mfa=True)
    alpha = repo.create_client(org, "Alpha Holdings", "ALPHA", u["alice"]["id"])
    beta = repo.create_client(org, "Beta Corp", "BETA", u["bob"]["id"])
    m = {
        "M-101": repo.create_matter(org, alpha["id"], "M-101", "MSA renewal", u["alice"]["id"]),
        "M-114": repo.create_matter(org, alpha["id"], "M-114", "Series B diligence", u["alice"]["id"], sensitivity="standard"),
        "M-202": repo.create_matter(org, beta["id"], "M-202", "Board investigation", u["bob"]["id"], sensitivity="highly_restricted"),
        "M-215": repo.create_matter(org, beta["id"], "M-215", "Lease dispute", u["bob"]["id"]),
    }
    repo.add_membership(m["M-101"]["id"], u["carol"]["id"], "read_only", granted_by=u["alice"]["id"])
    gone = repo.add_membership(m["M-215"]["id"], u["carol"]["id"], "paralegal", granted_by=u["bob"]["id"])
    repo.revoke_membership(m["M-215"]["id"], gone["id"], by=u["bob"]["id"], reason="left the team")
    repo.add_membership(m["M-202"]["id"], u["dave"]["id"], "attorney", granted_by=u["bob"]["id"],
                        valid_from=_ts(-30), valid_to=_ts(-1))
    return {"users": u, "matters": m, "clients": {"Alpha": alpha, "Beta": beta}, "app": law_app}


def as_user(world, name):
    """One signed-in client per person per test: an authenticator code is single-use (replay protection),
    so signing the same person in twice within 30 s would — correctly — be refused."""
    from fastapi.testclient import TestClient
    cache = world.setdefault("_clients", {})
    if name not in cache:
        c = TestClient(world["app"], base_url=LAW_TEST_ORIGIN)
        sign_in(c, world["users"][name])
        cache[name] = c
    return cache[name]


def numbers(resp) -> set:
    assert resp.status_code == 200, resp.text
    return {m["number"] for m in resp.json()["matters"]}




# ── Matter documents (S1.6) ──────────────────────────────────────────────────────
import hashlib  # noqa: E402
import re  # noqa: E402


class HashEmbedder:
    """Deterministic stand-in for the embedding server: a hashed bag of words, so passages that
    share words with the query are near it. Tests never call the real model server."""
    dim = 64

    async def embed(self, texts, kind="query"):
        import numpy as np
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]{2,}", (t or "").lower()):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return out


async def overlap_rerank(query, hits, top_k):
    """Stand-in for the cross-encoder: the share of query words found, minus a half, so a passage
    sharing fewer than half of the query's words scores below zero (not relevant)."""
    q = set(re.findall(r"[a-z0-9]{2,}", query.lower())) or {""}
    scored = [{**h, "score": len(q & set(re.findall(r"[a-z0-9]{2,}", (h.get("text") or "").lower()))) / len(q) - 0.5,
               "score_kind": "ce_logit"} for h in hits]
    return sorted(scored, key=lambda h: h["score"], reverse=True)[:top_k]


@pytest.fixture
def docs(world, monkeypatch):
    """The T01 world with the test embedder and reranker, plus helpers to import and index."""
    from app.core.config import settings
    from app.legal.retrieval import embedder, search
    monkeypatch.setattr(embedder, "_override", HashEmbedder())
    monkeypatch.setattr(search, "rerank", overlap_rerank)
    # The floor belongs to the reranker it was calibrated for: 0.0 for this stand-in (under half the query's
    # words = below zero); the real cross-encoder's default was calibrated in S2.2 (D-038).
    monkeypatch.setattr(settings, "LEGAL_RERANK_MIN_SCORE", 0.0)
    return world


def upload(c, matter_id, name, content, *, mime="text/plain", **form):
    if isinstance(content, str):
        content = content.encode()
    return c.post(f"/api/legal/matters/{matter_id}/documents/import", headers=csrf_headers(c),
                  files={"file": (name, content, mime)}, data={"kind": "contract", **form})


def index_all() -> int:
    from app.legal.jobs import worker
    return worker.run_pending()


def search_in(c, matter_id, query, **kw):
    return c.post(f"/api/legal/matters/{matter_id}/documents/search", headers=csrf_headers(c), json={"query": query, **kw})
