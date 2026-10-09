"""get_conn keeps the WAL in place between requests (S1.11 stall fix; TEST_RESULTS run 12).

Every request opens and closes its own connection. Without an idle connection held open, the one
that closes last checkpoints the WAL, deletes it and tears down its shared-memory index, and the
next one rebuilds them. That churn is where the E2E backend stalled (every thread in sqlite3
connect/close). These tests pin the fix: the WAL survives a close, and switching databases (as
the test suite does) releases the previous one cleanly.
"""
from __future__ import annotations

import os
import sqlite3
import threading


def _write(value: int) -> None:
    from app.core.db import get_conn
    with get_conn() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS t(x INTEGER)")
        conn.execute("INSERT INTO t(x) VALUES (?)", (value,))
        conn.commit()


def test_wal_and_index_survive_the_last_close(fresh_db):
    _write(1)
    assert os.path.exists(f"{fresh_db}-wal")
    assert os.path.exists(f"{fresh_db}-shm")


def test_switching_databases_releases_the_previous_one(fresh_db, tmp_path, monkeypatch):
    from app.core.config import settings
    _write(1)
    other = tmp_path / "other.sqlite"
    monkeypatch.setattr(settings, "DB_PATH", str(other))
    _write(2)
    assert not os.path.exists(f"{fresh_db}-wal")             # checkpointed into the file and removed
    with sqlite3.connect(fresh_db) as raw:
        assert raw.execute("SELECT x FROM t").fetchall() == [(1,)]
    assert os.path.exists(f"{other}-wal")


def test_many_threads_open_and_close_without_losing_writes(fresh_db):
    _write(0)
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(40):
                _write(n * 1000 + i)
        except BaseException as e:                           # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors and not any(t.is_alive() for t in threads)
    from app.core.db import get_conn
    with get_conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1 + 8 * 40
