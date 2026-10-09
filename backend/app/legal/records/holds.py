"""Legal holds (S1.9; policy P-003 safe default).

A records administrator requests a hold over whole matters; it is in force once a different
records administrator or a managing partner approves. Releasing it also needs two people.
While a hold is in force (or awaiting release), nothing in its matters can be deleted by any
path: a person's request, an approval, the retention runner or a direct service call.
"""
from __future__ import annotations

import uuid
from typing import Optional

from ...core.db import get_conn
from ..policy import now_iso

IN_FORCE = ("active", "release_requested")


class Held(Exception):
    """The material is under a legal hold."""


def _on_hold_sql(param: str = "?") -> str:
    return ("EXISTS (SELECT 1 FROM legal_hold_scopes s JOIN legal_holds h ON h.id = s.hold_id "
            f"WHERE h.status IN ('active','release_requested') AND s.object_type = 'matter' AND s.object_id = {param})")


def matter_on_hold(conn, matter_id: str) -> bool:
    return bool(conn.execute(f"SELECT {_on_hold_sql()}", (matter_id,)).fetchone()[0])


def assert_not_held(conn, matter_id: str) -> None:
    if matter_on_hold(conn, matter_id):
        raise Held(matter_id)


def matters_on_hold(matter_ids: list[str]) -> set[str]:
    if not matter_ids:
        return set()
    with get_conn() as conn:
        rows = conn.execute(f"SELECT DISTINCT s.object_id FROM legal_hold_scopes s JOIN legal_holds h ON h.id = s.hold_id "
                            f"WHERE h.status IN ('active','release_requested') AND s.object_type = 'matter' "
                            f"AND s.object_id IN ({','.join('?' * len(matter_ids))})", tuple(matter_ids)).fetchall()
    return {r[0] for r in rows}


_COLS = ("id", "org_id", "name", "reason", "status", "requested_by", "requested_at", "approved_by", "approved_at",
         "release_requested_by", "release_requested_at", "released_by", "released_at")


def create(org_id: str, name: str, reason: str, by: str, matter_ids: list[str]) -> str:
    hid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_holds (id, org_id, name, reason, status, requested_by, requested_at) VALUES (?,?,?,?,?,?,?)",
                     (hid, org_id, name, reason, "requested", by, now_iso()))
        conn.executemany("INSERT INTO legal_hold_scopes (hold_id, object_type, object_id) VALUES (?,?,?)",
                         [(hid, "matter", m) for m in dict.fromkeys(matter_ids)])
        conn.commit()
    return hid


def row(org_id: str, hold_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_holds WHERE id=? AND org_id=?", (hold_id, org_id)).fetchone()
    return dict(zip(_COLS, r)) if r else None


def update(hold_id: str, **fields) -> None:
    with get_conn() as conn:
        conn.execute(f"UPDATE legal_holds SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", (*fields.values(), hold_id))
        conn.commit()


def listing(org_id: str, hold_id: Optional[str] = None) -> list[dict]:
    q = f"SELECT {', '.join(_COLS)} FROM legal_holds WHERE org_id=?"
    args: list = [org_id]
    if hold_id:
        q += " AND id=?"
        args.append(hold_id)
    with get_conn() as conn:
        out = []
        for r in conn.execute(q + " ORDER BY requested_at DESC", args).fetchall():
            h = dict(zip(_COLS, r))
            names = {}
            for key in ("requested_by", "approved_by", "release_requested_by", "released_by"):
                if h[key]:
                    u = conn.execute("SELECT username FROM users WHERE id=?", (h[key],)).fetchone()
                    names[key] = u[0] if u else None
            matters = conn.execute("SELECT m.id, m.number, m.title, c.name FROM legal_hold_scopes s JOIN matters m ON m.id = s.object_id "
                                   "JOIN clients c ON c.id = m.client_id WHERE s.hold_id=? AND s.object_type='matter' ORDER BY m.number",
                                   (h["id"],)).fetchall()
            out.append({"id": h["id"], "name": h["name"], "reason": h["reason"], "status": h["status"],
                        "requested_by": names.get("requested_by"), "requested_at": h["requested_at"],
                        "approved_by": names.get("approved_by"), "approved_at": h["approved_at"],
                        "release_requested_by": names.get("release_requested_by"), "released_by": names.get("released_by"),
                        "released_at": h["released_at"],
                        "matters": [{"id": m[0], "number": m[1], "title": m[2], "client": m[3]} for m in matters]})
    return out
