"""Limits and data hygiene (commercial-readiness review, 2026-10-28: F4, F8, R6, R11).

- Model work is limited per person (LEGAL_MODEL_REQUESTS_PER_MIN), so one person cannot starve the firm's shared model.
- A request body that is not an upload is small and declares its size, checked before anything parses it — also on the
  sign-in and intake-link routes, which need no session.
- An intake link's token, its only key, is never kept in the activity log.
- SQLite overwrites deleted rows (secure_delete), so purged text does not survive in free pages or in a backup.
- Rolling the schema back below the chat's migrations is refused while it holds lawyers' decisions or conversations.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from conftest import as_user, csrf_headers


def test_model_work_is_limited_per_person_and_says_when_to_retry(world, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_MODEL_REQUESTS_PER_MIN", 2)
    alice, bob = as_user(world, "alice"), as_user(world, "bob")
    url = "/api/legal/matters/not-a-matter/research/query"
    for _ in range(2):
        assert alice.post(url, headers=csrf_headers(alice), json={"question": "x"}).status_code != 429
    r = alice.post(url, headers=csrf_headers(alice), json={"question": "x"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert bob.post(url, headers=csrf_headers(bob), json={"question": "x"}).status_code != 429      # per person
    assert alice.get("/api/legal/matters").status_code == 200                                         # reading is not limited


def test_the_model_work_paths(monkeypatch):
    from app.legal import quota
    for p in ("/api/legal/matters/m/research/query", "/api/legal/counsel/conversations/c/turns", "/api/legal/matters/m/compare",
              "/api/legal/matters/m/compare/scan", "/api/legal/matters/m/compare/x/explain", "/api/legal/matters/m/panel",
              "/api/legal/matters/m/documents/d/summaries", "/api/legal/matters/m/documents/d/translate",
              "/api/legal/matters/m/documents/d/obligations/extract", "/api/legal/matters/m/meetings/x/minutes/draft",
              "/api/legal/matters/m/drafts", "/api/legal/intake/i/replies", "/api/legal/time/suggest", "/api/legal/matters/m/diligence/runs"):
        assert quota.MODEL_WORK.match(p), p
    for p in ("/api/legal/matters", "/api/legal/matters/m/documents/import", "/api/legal/counsel/actions",
              "/api/legal/matters/m/drafts/a/edit"):
        assert not quota.MODEL_WORK.match(p), p
    assert quota.take("x", per_minute=1, now=0.0) == 0 and quota.take("x", per_minute=1, now=1.0) > 0
    assert quota.take("x", per_minute=1, now=61.0) == 0                                               # refilled
    assert quota.take("y", per_minute=0) == 0                                                         # 0 = no limit


def test_large_or_undeclared_bodies_are_refused_before_anything_reads_them(world, monkeypatch):
    from fastapi.testclient import TestClient
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_MAX_BODY_BYTES", 1000)
    c = TestClient(world["app"], base_url="https://testserver")
    big = {"username": "alice", "password": "x" * 2000}
    assert c.post("/api/auth/login", json=big).status_code == 413
    assert c.post("/api/intake-link/" + "t" * 32 + "/submit", json={"answers": {"a": "x" * 2000}}).status_code == 413
    chunks = (b"{" + b" " * 10 + b"}" for _ in range(1))
    r = c.post("/api/auth/login", content=chunks, headers={"Content-Type": "application/json"})
    assert r.status_code == 411
    monkeypatch.setattr(settings, "LEGAL_MAX_BODY_BYTES", 2 * 1024 * 1024)
    assert c.post("/api/auth/login", json={"username": "alice", "password": "x" * 2000}).status_code == 422   # field limit


def test_intake_link_tokens_never_reach_the_activity_log(world, fresh_db):
    from fastapi.testclient import TestClient
    c = TestClient(world["app"], base_url="https://testserver")
    token = "Q" * 40
    c.post(f"/api/intake-link/{token}/submit", json={"answers": {}})
    with sqlite3.connect(fresh_db) as db:
        paths = [r[0] for r in db.execute("SELECT path FROM activity_log")]
    assert paths and not any(token in p for p in paths) and any("<secret>" in p for p in paths)


def test_deleted_rows_are_overwritten_in_the_database_file(fresh_db, law_edition):
    from app.core.db import get_conn
    canary = "PURGED-CANARY-" + "7" * 40
    with get_conn() as c:
        assert c.execute("PRAGMA secure_delete").fetchone()[0] == 1
        c.execute("CREATE TABLE IF NOT EXISTS t_secure (body TEXT)")
        c.execute("INSERT INTO t_secure VALUES (?)", (canary * 20,))
        c.commit()
        c.execute("DELETE FROM t_secure")
        c.commit()
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    with open(fresh_db, "rb") as f:
        assert canary.encode() not in f.read()


def test_rolling_back_below_the_chat_is_refused_while_it_holds_records(world, fresh_db):
    from app.legal.migrations import runner
    alice = as_user(world, "alice")
    mid = world["matters"]["M-101"]["id"]
    r = alice.post("/api/legal/counsel/conversations", headers=csrf_headers(alice), json={"matter_id": mid})
    assert r.status_code == 201, r.text
    with closing(sqlite3.connect(fresh_db)) as c:
        with pytest.raises(runner.MigrationError, match="chat conversation"):
            runner.migrate_down(c, target=35)
        assert all(r["state"] == "applied" for r in runner.status(c))


def test_ocr_renders_a_huge_page_at_a_lower_resolution(monkeypatch):
    from app.core.config import settings
    from app.legal.documents import extract

    class Rect:
        width, height = 72 * 100, 72 * 100                     # a 100 x 100 inch "page"

    class Page:
        rect = Rect()
        dpi = None

        def get_pixmap(self, dpi, colorspace):
            Page.dpi = dpi
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError):
        extract._ocr_page(Page())
    assert Page.dpi * 100 * Page.dpi * 100 <= settings.LEGAL_OCR_MAX_PIXELS and Page.dpi < settings.LEGAL_OCR_DPI
