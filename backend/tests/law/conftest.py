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


def make_user(username, password="correct horse battery staple", role="user", mfa=False):
    """Create a Law user; with mfa=True also enrol TOTP and return its secret."""
    from app.core import auth
    from app.legal.identity import mfa as mfamod
    u = auth.create_user(username, password, role=role)
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


def sign_in(c, user) -> None:
    """Full sign-in (password + TOTP) for an MFA-enrolled user."""
    r = c.post("/api/auth/login", json={"username": user["username"], "password": user["password"]})
    assert r.status_code == 200 and r.json()["status"] == "mfa_required", r.text
    r = c.post("/api/auth/mfa/verify", json={"code": current_code(user["secret"])}, headers=csrf_headers(c))
    assert r.status_code == 200, r.text
