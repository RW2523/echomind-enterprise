"""Versioned SQL migrations for the EchoMind Law schema (decision D-004).

Files live next to this module as ``NNNN_name.up.sql`` with a paired ``NNNN_name.down.sql``.
Each applied migration is recorded in ``schema_migrations`` with the SHA-256 of its up script.
At boot the runner refuses to continue if an applied script was edited, if the database
records a version the code does not know (database newer than code), or if the files have a
gap. Each migration runs in a single transaction together with its bookkeeping row, so a
failing script leaves no partial schema behind.

Legacy tables are still created by ``core.db.init_db``; this runner only owns Law tables and
additive changes made for the Law edition.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MIGRATIONS_DIR = Path(__file__).resolve().parent
_FILE_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.up\.sql$")
_TXN_RE = re.compile(r"^\s*(BEGIN|COMMIT|END|ROLLBACK|SAVEPOINT|RELEASE)\b", re.IGNORECASE | re.MULTILINE)
_COMMENT_RE = re.compile(r"--[^\n]*")
# Trigger bodies legitimately contain BEGIN ... END; (e.g. append-only audit triggers).
_TRIGGER_RE = re.compile(r"CREATE\s+(?:TEMP\s+|TEMPORARY\s+)?TRIGGER\b.*?\bEND\s*;", re.IGNORECASE | re.DOTALL)


def _manages_transactions(sql: str) -> bool:
    return bool(_TXN_RE.search(_TRIGGER_RE.sub("", _COMMENT_RE.sub("", sql))))


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    up_sql: str
    down_sql: str
    checksum: str


def discover(directory: Optional[Path] = None) -> list[Migration]:
    directory = Path(directory or MIGRATIONS_DIR)
    found: list[Migration] = []
    for up in sorted(directory.glob("*.up.sql")):
        m = _FILE_RE.match(up.name)
        if not m:
            raise MigrationError(f"bad migration file name: {up.name}")
        version, name = int(m.group(1)), m.group(2)
        down = up.with_name(f"{m.group(1)}_{name}.down.sql")
        if not down.exists():
            raise MigrationError(f"migration {up.name} has no down script ({down.name})")
        up_sql, down_sql = up.read_text(encoding="utf-8"), down.read_text(encoding="utf-8")
        for label, sql in (("up", up_sql), ("down", down_sql)):
            if _manages_transactions(sql):
                raise MigrationError(f"{up.name} ({label}): scripts must not manage transactions; the runner wraps each one")
        found.append(Migration(version, name, up_sql, down_sql, hashlib.sha256(up_sql.encode("utf-8")).hexdigest()))
    for expected, mig in enumerate(found, start=1):
        if mig.version != expected:
            raise MigrationError(f"migration version gap: expected {expected:04d}, found {mig.version:04d}_{mig.name}")
    return found


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    conn.commit()


def applied_versions(conn: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    _ensure_table(conn)
    return {v: (n, c) for v, n, c in conn.execute("SELECT version, name, checksum FROM schema_migrations")}


def _verify(applied: dict[int, tuple[str, str]], migrations: list[Migration]) -> None:
    known = {m.version: m for m in migrations}
    for version, (name, checksum) in sorted(applied.items()):
        mig = known.get(version)
        if mig is None:
            raise MigrationError(f"database has migration {version:04d}_{name} that this code does not know (database newer than code)")
        if mig.checksum != checksum:
            raise MigrationError(f"checksum mismatch for applied migration {version:04d}_{mig.name}: the script was edited after it was applied")


def _run_atomic(conn: sqlite3.Connection, sql: str, bookkeeping: str, label: str) -> None:
    # executescript() commits any open transaction first, then runs the text as-is; the explicit
    # BEGIN/COMMIT makes the script and its schema_migrations row one atomic unit. Values in
    # `bookkeeping` come only from validated file names, hex digests and timestamps.
    try:
        conn.executescript(f"BEGIN;\n{sql}\n;\n{bookkeeping}\nCOMMIT;")
    except sqlite3.Error as e:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise MigrationError(f"migration {label} failed and was rolled back: {e}") from e


def migrate_up(conn: sqlite3.Connection, directory: Optional[Path] = None, target: Optional[int] = None) -> list[int]:
    migrations = discover(directory)
    applied = applied_versions(conn)
    _verify(applied, migrations)
    done: list[int] = []
    for mig in migrations:
        if mig.version in applied or (target is not None and mig.version > target):
            continue
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        row = (f"INSERT INTO schema_migrations(version, name, checksum, applied_at) "
               f"VALUES ({mig.version}, '{mig.name}', '{mig.checksum}', '{ts}');")
        _run_atomic(conn, mig.up_sql, row, f"{mig.version:04d}_{mig.name}")
        done.append(mig.version)
    return done


# Reverting some migrations would destroy records the firm must keep — the audit trail (0006) and
# whatever is under a legal hold (0010; spec T04: no system path deletes held records) — or widen
# access: below 0012 the code no longer knows the permissions a document system set on synced
# documents. The runner refuses; such an upgrade is rolled back by restoring the backup taken
# before it (app.legal.backup).
_REFUSE_BELOW = (
    (6, "SELECT COUNT(*) FROM audit_events", "delete the audit trail ({n} events)"),
    (10, "SELECT COUNT(*) FROM legal_holds WHERE status IN ('active','release_requested')",
     "delete records under {n} active legal hold(s)"),
    (12, "SELECT COUNT(*) FROM legal_documents WHERE acl_restricted = 1 AND deleted_at IS NULL",
     "make {n} document(s) restricted by their source's permissions visible to the whole matter"),
    (18, "SELECT COUNT(*) FROM legal_transcript_revisions",
     "delete {n} transcript correction(s) with who made them and why"),
    (20, "SELECT COUNT(*) FROM legal_review_approvals",
     "delete {n} review decision(s) on AI drafts, with who made them"),
    (21, "SELECT COUNT(*) FROM legal_ai_artifacts WHERE kind = 'research_note'",
     "delete {n} research note(s) sent for review"),
    (23, "SELECT (SELECT COUNT(*) FROM legal_ai_artifacts WHERE kind = 'legal_draft') + (SELECT COUNT(*) FROM legal_template_versions)",
     "delete {n} legal draft(s) and firm template version(s)"),
    (24, "SELECT (SELECT COUNT(*) FROM legal_playbook_positions) + (SELECT COUNT(*) FROM legal_playbook_decisions)",
     "delete {n} playbook position(s) and lawyers' decisions on playbook checks"),
    (25, "SELECT COUNT(*) FROM legal_ai_artifacts WHERE kind NOT IN ('minutes','research_note','legal_draft')",
     "delete {n} redline version(s)"),
    (26, "SELECT COUNT(*) FROM legal_citation_notes",
     "delete {n} lawyers' note(s) on citation checks"),
    (27, "SELECT COUNT(*) FROM legal_definition_notes",
     "delete {n} lawyers' note(s) on definition checks"),
    (28, "SELECT COUNT(*) FROM legal_obligation_events WHERE action <> 'proposed'",
     "delete {n} decision(s) on obligations (confirmations, edits, dismissals) with who made them and why"),
    (29, "SELECT (SELECT COUNT(*) FROM legal_dd_checklists) + (SELECT COUNT(*) FROM legal_dd_issue_events WHERE action <> 'found')",
     "delete {n} due-diligence checklist version(s) and reviewers' decisions on issues"),
    (30, "SELECT COUNT(*) FROM legal_dd_issues WHERE related_json <> '[]'",
     "drop the second document quoted by {n} governance finding(s)"),
    (31, "SELECT (SELECT COUNT(*) FROM legal_intakes) + (SELECT COUNT(*) FROM legal_matter_parties) "
         "+ (SELECT COUNT(*) FROM user_roles WHERE role IN ('intake','conflicts'))",
     "delete {n} intake record(s), matter party record(s) and intake or conflicts role grant(s)"),
    (32, "SELECT COUNT(*) FROM legal_time_entries WHERE status = 'approved' OR edited = 1",
     "delete {n} time entr(y/ies) people edited or approved"),
    (33, "SELECT COUNT(*) FROM legal_panel_reviews", "delete {n} panel review(s) and the questions asked"),
    (34, "SELECT COUNT(*) FROM legal_regulatory_matches WHERE status <> 'new'",
     "delete {n} lawyers' decision(s) on regulatory updates, with who made them and why"),
)


def _refuse_unsafe_rollback(conn: sqlite3.Connection, applied: dict, target: int) -> None:
    for version, sql, what in _REFUSE_BELOW:
        if version in applied and target < version:
            n = conn.execute(sql).fetchone()[0]
            if n:
                raise MigrationError(f"reverting below {version:04d} would {what.format(n=n)}. Nothing was changed; "
                                     f"restore the backup taken before the upgrade instead")


def migrate_down(conn: sqlite3.Connection, target: int, directory: Optional[Path] = None) -> list[int]:
    migrations = {m.version: m for m in discover(directory)}
    applied = applied_versions(conn)
    _verify(applied, list(migrations.values()))
    _refuse_unsafe_rollback(conn, applied, target)
    done: list[int] = []
    for version in sorted((v for v in applied if v > target), reverse=True):
        mig = migrations[version]
        _run_atomic(conn, mig.down_sql, f"DELETE FROM schema_migrations WHERE version = {version};", f"{version:04d}_{mig.name} (down)")
        done.append(version)
    return done


def status(conn: sqlite3.Connection, directory: Optional[Path] = None) -> list[dict]:
    applied = applied_versions(conn)
    return [{"version": m.version, "name": m.name, "state": "applied" if m.version in applied else "pending"}
            for m in discover(directory)]


def apply_legal_migrations(db_path: str) -> list[int]:
    """Bring the Law schema up to date. Called from init_db() when the edition is `law`."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        return migrate_up(conn)
