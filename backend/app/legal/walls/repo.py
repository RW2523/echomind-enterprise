"""Storage for ethical walls, exceptions and break-glass grants. Authorisation is done by routes."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from ...core.db import get_conn
from ..policy import now_iso


def _id() -> str:
    return str(uuid.uuid4())


def iso_in(minutes: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _username(conn, user_id: Optional[str]) -> Optional[str]:
    if not user_id:
        return None
    row = conn.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
    return row[0] if row else None


# ── Walls ────────────────────────────────────────────────────────────────────────
def create_wall(org_id: str, name: str, rationale: str, requested_by: str, screened: list[str],
                matters: list[str], clients: list[str]) -> str:
    wid, now = _id(), now_iso()
    with get_conn() as conn:
        conn.execute("INSERT INTO ethical_walls (id, org_id, name, rationale, status, requested_by, requested_at) "
                     "VALUES (?,?,?,?,'requested',?,?)", (wid, org_id, name.strip(), rationale.strip(), requested_by, now))
        conn.executemany("INSERT INTO wall_screened (wall_id, user_id) VALUES (?,?)", [(wid, u) for u in screened])
        conn.executemany("INSERT INTO wall_protected (wall_id, object_type, object_id) VALUES (?,?,?)",
                         [(wid, "matter", m) for m in matters] + [(wid, "client", c) for c in clients])
        conn.commit()
    return wid


def get_wall_row(org_id: str, wall_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT id, status, requested_by, release_requested_by FROM ethical_walls WHERE id=? AND org_id=?",
                           (wall_id, org_id)).fetchone()
    return dict(zip(("id", "status", "requested_by", "release_requested_by"), row)) if row else None


def protected_of(wall_id: str) -> tuple[list[str], list[str]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT object_type, object_id FROM wall_protected WHERE wall_id=?", (wall_id,)).fetchall()
    return [r[1] for r in rows if r[0] == "matter"], [r[1] for r in rows if r[0] == "client"]


def set_wall_status(wall_id: str, status: str, **cols) -> None:
    sets = ", ".join(["status = :status"] + [f"{k} = :{k}" for k in cols])
    with get_conn() as conn:
        conn.execute(f"UPDATE ethical_walls SET {sets} WHERE id = :id", {"status": status, "id": wall_id, **cols})
        conn.commit()


def wall_covers(wall_id: str, user_id: str, matter_id: str) -> bool:
    """The wall screens this user and protects this matter (directly or through its client)."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT 1 FROM wall_screened ws, matters m, wall_protected wp WHERE ws.wall_id = :w AND ws.user_id = :u "
            "AND m.id = :m AND wp.wall_id = :w AND ((wp.object_type='matter' AND wp.object_id = m.id) "
            "OR (wp.object_type='client' AND wp.object_id = m.client_id))",
            {"w": wall_id, "u": user_id, "m": matter_id}).fetchone() is not None


def list_walls(org_id: str) -> list[dict]:
    with get_conn() as conn:
        walls = conn.execute(
            "SELECT id, name, rationale, status, requested_by, requested_at, approved_by, approved_at, "
            "release_requested_by, released_by, released_at FROM ethical_walls WHERE org_id=? ORDER BY requested_at DESC",
            (org_id,)).fetchall()
        out = []
        for w in walls:
            wid = w[0]
            screened = [r[0] for r in conn.execute(
                "SELECT u.username FROM wall_screened ws JOIN users u ON u.id = ws.user_id WHERE ws.wall_id=? ORDER BY u.username", (wid,))]
            protected = []
            for t, oid in conn.execute("SELECT object_type, object_id FROM wall_protected WHERE wall_id=?", (wid,)):
                if t == "matter":
                    r = conn.execute("SELECT m.number, m.title, c.name FROM matters m JOIN clients c ON c.id=m.client_id WHERE m.id=?", (oid,)).fetchone()
                    label = f"{r[0]} · {r[1]} ({r[2]})" if r else "Unknown matter"
                else:
                    r = conn.execute("SELECT name FROM clients WHERE id=?", (oid,)).fetchone()
                    label = f"Client: {r[0]}" if r else "Unknown client"
                protected.append({"type": t, "id": oid, "label": label})
            exceptions = [dict(zip(("id", "username", "matter_id", "reason", "expires_at", "approved", "revoked", "requested_by"),
                                   (e[0], e[1], e[2], e[3], e[4], e[5] is not None, e[6] is not None, _username(conn, e[7]))))
                          for e in conn.execute(
                              "SELECT we.id, u.username, we.matter_id, we.reason, we.expires_at, we.approved_at, we.revoked_at, we.requested_by "
                              "FROM wall_exceptions we JOIN users u ON u.id = we.user_id WHERE we.wall_id=? ORDER BY we.requested_at", (wid,))]
            out.append({"id": wid, "name": w[1], "rationale": w[2], "status": w[3],
                        "requested_by": _username(conn, w[4]), "requested_at": w[5],
                        "approved_by": _username(conn, w[6]), "approved_at": w[7],
                        "release_requested_by": _username(conn, w[8]), "released_by": _username(conn, w[9]),
                        "released_at": w[10], "screened": screened, "protected": protected, "exceptions": exceptions})
    return out


def get_wall(org_id: str, wall_id: str) -> Optional[dict]:
    return next((w for w in list_walls(org_id) if w["id"] == wall_id), None)


# ── Exceptions ───────────────────────────────────────────────────────────────────
def create_exception(wall_id: str, user_id: str, matter_id: str, reason: str, requested_by: str, expires_at: str) -> dict:
    eid = _id()
    with get_conn() as conn:
        conn.execute("INSERT INTO wall_exceptions (id, wall_id, user_id, matter_id, reason, requested_by, requested_at, expires_at) "
                     "VALUES (?,?,?,?,?,?,?,?)", (eid, wall_id, user_id, matter_id, reason.strip(), requested_by, now_iso(), expires_at))
        conn.commit()
    return get_exception(wall_id, eid)


def get_exception(wall_id: str, exc_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT id, user_id, matter_id, requested_by, approved_at, revoked_at, expires_at FROM wall_exceptions "
                           "WHERE id=? AND wall_id=?", (exc_id, wall_id)).fetchone()
    return dict(zip(("id", "user_id", "matter_id", "requested_by", "approved_at", "revoked_at", "expires_at"), row)) if row else None


def approve_exception(exc_id: str, by: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE wall_exceptions SET approved_by=?, approved_at=? WHERE id=?", (by, now_iso(), exc_id))
        conn.commit()


def revoke_exception(exc_id: str, by: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE wall_exceptions SET revoked_by=?, revoked_at=? WHERE id=? AND revoked_at IS NULL", (by, now_iso(), exc_id))
        conn.commit()


# ── Break-glass ──────────────────────────────────────────────────────────────────
_BG_COLS = ("id", "user_id", "requested_number", "matter_id", "reason", "minutes", "requested_at", "status",
            "decided_by", "decided_at", "decision_note", "expires_at", "ended_at")


def create_break_glass(org_id: str, user_id: str, number: str, reason: str, minutes: int) -> dict:
    gid = _id()
    with get_conn() as conn:
        conn.execute("INSERT INTO break_glass_grants (id, org_id, user_id, requested_number, reason, minutes, requested_at, status) "
                     "VALUES (?,?,?,?,?,?,?,'requested')", (gid, org_id, user_id, number.strip(), reason.strip(), minutes, now_iso()))
        conn.commit()
    return get_break_glass(org_id, gid)


def get_break_glass(org_id: str, gid: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(f"SELECT {', '.join(_BG_COLS)} FROM break_glass_grants WHERE id=? AND org_id=?", (gid, org_id)).fetchone()
    return dict(zip(_BG_COLS, row)) if row else None


def list_break_glass(org_id: str, user_id: Optional[str] = None) -> list[dict]:
    q = f"SELECT {', '.join(_BG_COLS)} FROM break_glass_grants WHERE org_id=?"
    args: list = [org_id]
    if user_id:
        q += " AND user_id=?"
        args.append(user_id)
    with get_conn() as conn:
        rows = conn.execute(q + " ORDER BY requested_at DESC LIMIT 200", args).fetchall()
        out = []
        for r in rows:
            d = dict(zip(_BG_COLS, r))
            d["requester"] = _username(conn, d["user_id"])
            d["decided_by_name"] = _username(conn, d["decided_by"])
            out.append(d)
    return out


def matter_by_number(org_id: str, number: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT m.id, m.number, m.title, m.sensitivity, c.name FROM matters m JOIN clients c ON c.id = m.client_id "
                           "WHERE m.org_id=? AND m.number=?", (org_id, number.strip())).fetchone()
    return dict(zip(("id", "number", "title", "sensitivity", "client"), row)) if row else None


def decide_break_glass(gid: str, status: str, by: str, note: str = "", matter_id: Optional[str] = None,
                       expires_at: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE break_glass_grants SET status=?, decided_by=?, decided_at=?, decision_note=?, "
                     "matter_id=COALESCE(?, matter_id), expires_at=? WHERE id=?",
                     (status, by, now_iso(), note, matter_id, expires_at, gid))
        conn.commit()


def end_break_glass(gid: str, by: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE break_glass_grants SET status='ended', ended_by=?, ended_at=? WHERE id=?", (by, now_iso(), gid))
        conn.commit()


# ── Lookups for routes ───────────────────────────────────────────────────────────
def org_user(org_id: str, username: str) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM users WHERE username=? AND org_id=? AND COALESCE(status,'active')='active'",
                           ((username or "").strip(), org_id)).fetchone()
    return row[0] if row else None


def org_has(org_id: str, table: str, object_id: str) -> bool:
    assert table in ("matters", "clients")
    with get_conn() as conn:
        return conn.execute(f"SELECT 1 FROM {table} WHERE id=? AND org_id=?", (object_id, org_id)).fetchone() is not None


def matter_directory(org_id: str, q: str = "") -> list[dict]:
    """Every matter of the organisation (number, title, client) — for the walls authority only (D-022)."""
    like = f"%{q.strip()}%"
    with get_conn() as conn:
        rows = conn.execute("SELECT m.id, m.number, m.title, c.id, c.name FROM matters m JOIN clients c ON c.id = m.client_id "
                            "WHERE m.org_id=? AND (m.number LIKE ? OR m.title LIKE ? OR c.name LIKE ?) ORDER BY m.number LIMIT 100",
                            (org_id, like, like, like)).fetchall()
    return [{"id": r[0], "number": r[1], "title": r[2], "client": {"id": r[3], "name": r[4]}} for r in rows]
