"""The audit chain: storage, hashing, daily anchors and verification (S1.5, SEC-04; decision D-008).

Each event stores the previous event's hash and its own:

    hash = SHA-256( prev_hash + canonical_json(event fields) )

canonical_json is `json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
over the columns in FIELDS (seq and the two hashes excluded); the first event's prev_hash is 64
zeros. Anyone holding a JSONL export can recompute the chain with this rule alone.

Triggers refuse UPDATE and DELETE. Someone with direct file access could drop them, edit a row and
re-hash every later row; the anchor files (the head of the chain, written to DATA_DIR/legal/
audit_anchor/ at least daily and kept in backups) are what catch that, and a removed tail.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Iterator, Optional

from ...core.config import settings
from ...core.db import get_conn

GENESIS = "0" * 64
FIELDS = ("id", "ts", "org_id", "actor_user_id", "actor_username", "actor_roles", "auth", "session_ref", "ip",
          "action", "object_type", "object_id", "matter_id", "outcome", "reason", "details_json")
COLUMNS = ("seq",) + FIELDS + ("prev_hash", "hash")

# The head is re-anchored at most this often per process (and always on a day's first event).
ANCHOR_INTERVAL_S = 60

_lock = threading.Lock()
_last_anchor: dict[str, float] = {}


def canonical(fields: dict) -> str:
    return json.dumps({k: fields.get(k) for k in FIELDS}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(prev_hash: str, fields: dict) -> str:
    return hashlib.sha256((prev_hash + canonical(fields)).encode("utf-8")).hexdigest()


def append(fields: dict) -> tuple[int, str]:
    """Add one event at the head of the chain; returns (seq, hash)."""
    with _lock, get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")          # no other writer between reading the head and inserting
        try:
            row = conn.execute("SELECT hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
            h = digest(row[0] if row else GENESIS, fields)
            cur = conn.execute(
                "INSERT INTO audit_events (id, ts, org_id, actor_user_id, actor_username, actor_roles, auth, session_ref, "
                "ip, action, object_type, object_id, matter_id, outcome, reason, details_json, prev_hash, hash) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                tuple(fields.get(k) for k in FIELDS) + (row[0] if row else GENESIS, h))
            conn.commit()
            seq = cur.lastrowid
        except BaseException:
            conn.rollback()
            raise
    _maybe_anchor(seq, h)
    return seq, h


# ── Anchors ──────────────────────────────────────────────────────────────────────
def anchor_dir() -> str:
    return os.path.join(settings.DATA_DIR, "legal", "audit_anchor")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _read_anchor(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_anchor(seq: int, h: str) -> str:
    """Record (seq, hash) as the day's anchor; never moves an anchor backwards."""
    d = anchor_dir()
    os.makedirs(d, mode=0o700, exist_ok=True)
    path = os.path.join(d, f"{_today()}.json")
    current = _read_anchor(path)
    if current and int(current.get("seq", 0)) >= seq:
        return path
    tmp = f"{path}.{os.getpid()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"date": _today(), "seq": seq, "hash": h,
                   "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, f)
    os.replace(tmp, path)
    _last_anchor[path] = time.monotonic()
    return path


def _maybe_anchor(seq: int, h: str) -> None:
    path = os.path.join(anchor_dir(), f"{_today()}.json")
    last = _last_anchor.get(path)
    if last is None or not os.path.exists(path) or time.monotonic() - last >= ANCHOR_INTERVAL_S:
        write_anchor(seq, h)


def anchor_now() -> Optional[str]:
    """Anchor the current head (CLI / scheduled job)."""
    with get_conn() as conn:
        row = conn.execute("SELECT seq, hash FROM audit_events ORDER BY seq DESC LIMIT 1").fetchone()
    return write_anchor(row[0], row[1]) if row else None


def _anchors() -> list[dict]:
    d = anchor_dir()
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            a = _read_anchor(os.path.join(d, name))
            out.append(a if isinstance(a, dict) else {"file": name, "unreadable": True})
    return out


# ── Reading and verifying ────────────────────────────────────────────────────────
def rows(after_seq: int = 0, batch: int = 1000) -> Iterator[dict]:
    """Every event in chain order, as dicts of COLUMNS."""
    while True:
        with get_conn() as conn:
            chunk = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM audit_events WHERE seq > ? ORDER BY seq LIMIT ?",
                                 (after_seq, batch)).fetchall()
        for r in chunk:
            yield dict(zip(COLUMNS, r))
        if len(chunk) < batch:
            return
        after_seq = chunk[-1][0]


def verify() -> dict:
    """Recompute the whole chain, then check every anchor against it."""
    prev, count, broken_at, problem = GENESIS, 0, None, None
    hashes: dict[int, str] = {}
    head_seq = 0
    for r in rows():
        count += 1
        head_seq = r["seq"]
        hashes[r["seq"]] = r["hash"]
        if broken_at is None:
            if r["prev_hash"] != prev:
                broken_at, problem = r["seq"], "link_broken"        # an event was removed or reordered
            elif digest(prev, r) != r["hash"]:
                broken_at, problem = r["seq"], "event_altered"
        prev = r["hash"]
    anchors_checked = 0
    for a in _anchors():
        anchors_checked += 1
        if problem:
            break
        if a.get("unreadable"):
            problem = "anchor_unreadable"
        elif int(a.get("seq", 0)) > head_seq:
            problem = "events_missing_after_anchor"
        elif hashes.get(int(a.get("seq", 0))) != a.get("hash"):
            broken_at, problem = int(a.get("seq", 0)), "anchor_mismatch"
    return {"ok": problem is None, "events": count, "head": prev if count else None, "head_seq": head_seq or None,
            "broken_at": broken_at, "problem": problem, "anchors_checked": anchors_checked,
            "verified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
