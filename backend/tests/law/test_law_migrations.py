"""S1.0 / D-004 — versioned, checksummed, reversible migrations for the Law schema."""
from __future__ import annotations

import shutil
import sqlite3

import pytest

from app.legal.migrations import runner
from app.legal.migrations.runner import MigrationError


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "m.sqlite")
    yield c
    c.close()


@pytest.fixture
def mig_dir(tmp_path):
    """A private copy of the real migrations, safe to tamper with."""
    d = tmp_path / "migrations"
    d.mkdir()
    for f in runner.MIGRATIONS_DIR.glob("*.sql"):
        shutil.copy(f, d / f.name)
    return d


def _write(d, version, name, up, down="SELECT 1;"):
    (d / f"{version:04d}_{name}.up.sql").write_text(up)
    (d / f"{version:04d}_{name}.down.sql").write_text(down)


def test_real_migrations_are_well_formed():
    migs = runner.discover()
    assert [m.version for m in migs][:1] == [1]
    assert migs[0].name == "foundation"
    assert all(len(m.checksum) == 64 for m in migs)


def test_up_applies_and_records(conn):
    applied = runner.migrate_up(conn)
    assert 1 in applied
    assert {"organizations", "schema_migrations"} <= _tables(conn)
    v, name, checksum, applied_at = conn.execute(
        "SELECT version, name, checksum, applied_at FROM schema_migrations WHERE version=1").fetchone()
    assert (v, name) == (1, "foundation") and len(checksum) == 64 and applied_at.endswith("Z")


def test_up_is_idempotent(conn):
    runner.migrate_up(conn)
    assert runner.migrate_up(conn) == []


def test_down_reverts_to_zero(conn):
    runner.migrate_up(conn)
    reverted = runner.migrate_down(conn, target=0)
    assert 1 in reverted
    assert "organizations" not in _tables(conn)
    assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 0
    assert runner.migrate_up(conn) == [m.version for m in runner.discover()]  # re-applies cleanly


def test_edited_applied_migration_refuses(conn, mig_dir):
    runner.migrate_up(conn, directory=mig_dir)
    up = mig_dir / "0001_foundation.up.sql"
    up.write_text(up.read_text() + "\n-- edited after release\n")
    with pytest.raises(MigrationError, match="checksum"):
        runner.migrate_up(conn, directory=mig_dir)


def test_failed_migration_rolls_back_atomically(conn, mig_dir):
    nxt = max(m.version for m in runner.discover(mig_dir)) + 1
    _write(mig_dir, nxt, "broken", "CREATE TABLE half_done(x TEXT);\nCREATE TABLE half_done(x TEXT);\n")
    with pytest.raises(MigrationError, match="broken"):
        runner.migrate_up(conn, directory=mig_dir)
    assert "half_done" not in _tables(conn)
    assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == nxt - 1


def test_missing_down_script_is_rejected(mig_dir):
    (mig_dir / "0001_foundation.down.sql").unlink()
    with pytest.raises(MigrationError, match="down"):
        runner.discover(mig_dir)


def test_version_gap_is_rejected(mig_dir):
    _write(mig_dir, 9, "skips_ahead", "CREATE TABLE t9(x TEXT);")
    with pytest.raises(MigrationError, match="gap"):
        runner.discover(mig_dir)


def test_scripts_may_not_manage_transactions(mig_dir):
    nxt = max(m.version for m in runner.discover(mig_dir)) + 1
    _write(mig_dir, nxt, "sneaky", "BEGIN;\nCREATE TABLE t(x TEXT);\nCOMMIT;\n")
    with pytest.raises(MigrationError, match="transaction"):
        runner.discover(mig_dir)


def test_trigger_bodies_are_not_mistaken_for_transactions(conn, mig_dir):
    nxt = max(m.version for m in runner.discover(mig_dir)) + 1
    _write(mig_dir, nxt, "append_only", (
        "-- BEGIN; in a comment is ignored too\n"
        "CREATE TABLE ledger(x TEXT);\n"
        "CREATE TRIGGER ledger_no_update BEFORE UPDATE ON ledger\n"
        "BEGIN\n  SELECT RAISE(ABORT, 'append-only');\nEND;\n"),
        down="DROP TRIGGER ledger_no_update;\nDROP TABLE ledger;")
    runner.migrate_up(conn, directory=mig_dir)
    conn.execute("INSERT INTO ledger VALUES ('a')")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE ledger SET x='b'")


def test_database_newer_than_code_refuses(conn):
    runner.migrate_up(conn)
    conn.execute("INSERT INTO schema_migrations(version, name, checksum, applied_at) VALUES (999, 'future', ?, '2030-01-01T00:00:00Z')",
                 ("0" * 64,))
    conn.commit()
    with pytest.raises(MigrationError, match="999"):
        runner.migrate_up(conn)


def test_init_db_applies_law_migrations_only_in_law_edition(fresh_db, monkeypatch):
    from app.core.db import init_db
    monkeypatch.delenv("ECHOMIND_EDITION", raising=False)
    init_db()
    with sqlite3.connect(fresh_db) as c:
        assert "organizations" not in _tables(c)
    monkeypatch.setenv("ECHOMIND_EDITION", "law")
    init_db()
    with sqlite3.connect(fresh_db) as c:
        assert {"organizations", "schema_migrations"} <= _tables(c)


def test_cli_status_up_down(fresh_db, capsys):
    from app.legal.migrations.__main__ import main
    assert main(["status"]) == 0
    assert "pending" in capsys.readouterr().out
    assert main(["up"]) == 0
    assert main(["down", "--to", "0"]) == 0
    assert main(["status"]) == 0
    assert "pending" in capsys.readouterr().out
