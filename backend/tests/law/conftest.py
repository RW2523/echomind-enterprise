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
