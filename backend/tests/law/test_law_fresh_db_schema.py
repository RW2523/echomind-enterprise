"""S1.0 — a brand-new database must support every statement the code runs (defects DB-1, DB-2).

DB-1: ingestion inserts chunks.page_start/page_end, but init_db() never created them, so every
      upload on a fresh install failed.
DB-2: init_db() ALTERed boardroom_sessions before CREATE-ing it, so on first boot the table lacked
      scenario/namespace/speaker_map_json and Boardroom reads failed until a restart.
"""
from __future__ import annotations

import pathlib
import re
import sqlite3

APP_DIR = pathlib.Path(__file__).resolve().parents[2] / "app"
_INSERT_RE = re.compile(r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)\s*\(([^)]*)\)", re.IGNORECASE)


def _columns(db_path, table) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_fresh_chunks_table_has_page_columns(fresh_db, legacy_edition):
    from app.core.db import init_db
    init_db()
    assert {"page_start", "page_end", "contextualized_text"} <= _columns(fresh_db, "chunks")


def test_fresh_boardroom_table_has_meta_columns(fresh_db, legacy_edition):
    from app.core.db import init_db
    init_db()
    assert {"scenario", "namespace", "speaker_map_json"} <= _columns(fresh_db, "boardroom_sessions")


def test_every_insert_in_the_code_matches_the_fresh_schema(fresh_db, law_edition):
    """Static guard against schema drift: each INSERT's column list must exist on a fresh DB.
    Runs in the Law edition, whose schema is the legacy tables plus the Law migrations."""
    from app.core.db import init_db
    init_db()
    problems = []
    for py in sorted(APP_DIR.rglob("*.py")):
        src = py.read_text(encoding="utf-8")
        for m in _INSERT_RE.finditer(src):
            table = m.group(1)
            cols = {c.strip() for c in m.group(2).split(",") if c.strip()}
            existing = _columns(fresh_db, table)
            if not existing:
                problems.append(f"{py.relative_to(APP_DIR)}: table {table} missing")
                continue
            missing = cols - existing
            if missing:
                problems.append(f"{py.relative_to(APP_DIR)}: {table} missing {sorted(missing)}")
    assert not problems, "\n".join(problems)


def test_boardroom_session_roundtrip_on_first_boot(fresh_db, legacy_edition):
    from app.core.db import init_db
    from app.boardroom import service
    init_db()
    sid = service.create_session()
    assert service.set_session_meta(sid, scenario="legal", namespace="law", speaker_map={"Speaker 1": "lawyer"})
    s = service.get_session(sid)
    assert s["scenario"] == "legal" and s["namespace"] == "law" and s["speaker_map"] == {"Speaker 1": "lawyer"}


def test_existing_database_is_upgraded_in_place(fresh_db, legacy_edition):
    """A DB created by the old init_db (no page columns, boardroom without meta) gains them on boot."""
    with sqlite3.connect(fresh_db) as conn:
        conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER, text TEXT, source_json TEXT)")
        conn.execute("INSERT INTO chunks VALUES ('c1','d1',0,'kept','{}')")
        conn.execute("CREATE TABLE boardroom_sessions(id TEXT PRIMARY KEY, transcript_id TEXT, status TEXT, chunk_count INTEGER DEFAULT 0,"
                     " audio_format TEXT, diarized_json TEXT, report_json TEXT, created_at TEXT, updated_at TEXT)")
        conn.execute("INSERT INTO boardroom_sessions(id, status) VALUES ('brm_1','analysed')")
    from app.core.db import init_db
    init_db()
    assert {"page_start", "page_end"} <= _columns(fresh_db, "chunks")
    assert {"scenario", "namespace", "speaker_map_json"} <= _columns(fresh_db, "boardroom_sessions")
    with sqlite3.connect(fresh_db) as conn:
        assert conn.execute("SELECT text FROM chunks WHERE id='c1'").fetchone() == ("kept",)
        assert conn.execute("SELECT status FROM boardroom_sessions WHERE id='brm_1'").fetchone() == ("analysed",)


def test_init_db_is_idempotent(fresh_db, legacy_edition):
    from app.core.db import init_db
    init_db()
    init_db()
    assert {"page_start", "page_end"} <= _columns(fresh_db, "chunks")
