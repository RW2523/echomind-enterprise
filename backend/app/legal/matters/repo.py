"""Storage for clients, matters and matter memberships (S1.2, SEC-02).

Authorisation is NOT done here: routes call legal.policy first. Listing functions take the
principal and return only what its live memberships make visible.
"""
from __future__ import annotations

import sqlite3
import uuid
from typing import Optional

from ...core.db import get_conn
from ..policy import _CLIENT_WALLED, _LIVE_MEMBERSHIP, my_role_sql, now_iso, visible_matters_sql


class Conflict(Exception):
    """A uniqueness, version or invariant conflict (→ 409)."""


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Clients ──────────────────────────────────────────────────────────────────────
_CLIENT_COLS = "c.id, c.org_id, c.name, c.code, c.status, c.created_at, c.created_by, c.updated_at, c.version"


def _client(row) -> Optional[dict]:
    if not row:
        return None
    return dict(zip(("id", "org_id", "name", "code", "status", "created_at", "created_by", "updated_at", "version"), row))


def create_client(org_id: str, name: str, code: Optional[str], created_by: str) -> dict:
    cid, now = _new_id(), now_iso()
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO clients (id, org_id, name, code, status, created_at, created_by, updated_at, version) "
                         "VALUES (?,?,?,?,?,?,?,?,1)", (cid, org_id, name.strip(), (code or "").strip() or None,
                                                       "active", now, created_by, now))
            conn.commit()
    except sqlite3.IntegrityError:
        raise Conflict("A client with that code already exists.")
    return get_client(cid)


def get_client(client_id: str) -> Optional[dict]:
    with get_conn() as conn:
        return _client(conn.execute(f"SELECT {_CLIENT_COLS} FROM clients c WHERE c.id=?", (client_id,)).fetchone())


def _visible_clients_where() -> str:
    # A client is visible to its creator (unless walled from that client), or through any matter the
    # principal may see (which already accounts for walls and break-glass).
    return (f"c.org_id = :org AND ((c.created_by = :uid AND NOT {_CLIENT_WALLED}) OR EXISTS (SELECT 1 FROM matters m "
            f"WHERE m.client_id = c.id AND {visible_matters_sql()}))")


def list_visible_clients(principal: dict) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {_CLIENT_COLS} FROM clients c WHERE {_visible_clients_where()} ORDER BY c.name",
                            {"org": principal["org_id"], "uid": principal["sub"], "now": now_iso()}).fetchall()
    return [_client(r) for r in rows]


def get_visible_client(principal: dict, client_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(f"SELECT {_CLIENT_COLS} FROM clients c WHERE c.id = :cid AND {_visible_clients_where()}",
                           {"cid": client_id, "org": principal["org_id"], "uid": principal["sub"], "now": now_iso()}).fetchone()
    return _client(row)


def update_client(client_id: str, fields: dict, expected_version: int) -> dict:
    sets = {k: v for k, v in fields.items() if k in ("name", "code", "status") and v is not None}
    if not sets:
        return get_client(client_id)
    assigns = ", ".join(f"{k} = :{k}" for k in sets)
    try:
        with get_conn() as conn:
            cur = conn.execute(f"UPDATE clients SET {assigns}, version = version + 1, updated_at = :now "
                               f"WHERE id = :id AND version = :v", {**sets, "now": now_iso(), "id": client_id, "v": expected_version})
            conn.commit()
    except sqlite3.IntegrityError:
        raise Conflict("A client with that code already exists.")
    if cur.rowcount != 1:
        raise Conflict("This client was changed by someone else. Reload and try again.")
    return get_client(client_id)


# ── Matters ──────────────────────────────────────────────────────────────────────
_MATTER_COLS = ("m.id, m.org_id, m.client_id, m.number, m.title, m.practice_area, m.jurisdiction, m.status, "
                "m.sensitivity, m.created_at, m.created_by, m.updated_at, m.version, c.name, c.code")
_MATTER_KEYS = ("id", "org_id", "client_id", "number", "title", "practice_area", "jurisdiction", "status",
                "sensitivity", "created_at", "created_by", "updated_at", "version")


def _matter(row, my_role: Optional[str] = None) -> Optional[dict]:
    if not row:
        return None
    d = dict(zip(_MATTER_KEYS, row[:13]))
    d["client"] = {"id": d["client_id"], "name": row[13], "code": row[14]}
    if my_role is not None:
        d["my_role"] = my_role
    return d


def create_matter(org_id: str, client_id: str, number: str, title: str, created_by: str, *,
                  practice_area: Optional[str] = None, jurisdiction: Optional[str] = None,
                  sensitivity: str = "restricted") -> dict:
    """Open a matter; its creator becomes the lead, in the same transaction."""
    mid, now = _new_id(), now_iso()
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO matters (id, org_id, client_id, number, title, practice_area, jurisdiction, status, sensitivity, "
                "acl_epoch, created_at, created_by, updated_at, version) VALUES (?,?,?,?,?,?,?,'open',?,1,?,?,?,1)",
                (mid, org_id, client_id, number.strip(), title.strip(), practice_area, jurisdiction, sensitivity, now, created_by, now))
            conn.execute(
                "INSERT INTO matter_memberships (id, matter_id, user_id, role, can_export, valid_from, granted_by, granted_at) "
                "VALUES (?,?,?,?,?,?,?,?)", (_new_id(), mid, created_by, "lead", 1, now, created_by, now))
            conn.commit()
    except sqlite3.IntegrityError:
        raise Conflict("A matter with that number already exists.")
    return get_matter(mid, my_role="lead")


def get_matter(matter_id: str, my_role: Optional[str] = None) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(f"SELECT {_MATTER_COLS} FROM matters m JOIN clients c ON c.id = m.client_id WHERE m.id = ?",
                           (matter_id,)).fetchone()
    return _matter(row, my_role)


def list_visible_matters(principal: dict) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT {_MATTER_COLS}, {my_role_sql()} FROM matters m JOIN clients c ON c.id = m.client_id "
            f"WHERE {visible_matters_sql()} ORDER BY m.number",
            {"org": principal["org_id"], "uid": principal["sub"], "now": now_iso()}).fetchall()
    return [_matter(r, r[15]) for r in rows]


def update_matter(matter_id: str, fields: dict, expected_version: int) -> dict:
    allowed = ("title", "practice_area", "jurisdiction", "status", "sensitivity")
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    assigns = ", ".join([f"{k} = :{k}" for k in sets] + ["version = version + 1", "updated_at = :now"])
    with get_conn() as conn:
        cur = conn.execute(f"UPDATE matters SET {assigns} WHERE id = :id AND version = :v",
                           {**sets, "now": now_iso(), "id": matter_id, "v": expected_version})
        conn.commit()
    if cur.rowcount != 1:
        raise Conflict("This matter was changed by someone else. Reload and try again.")
    return get_matter(matter_id)


# ── Memberships ──────────────────────────────────────────────────────────────────
_MM_COLS = ("mm.id, mm.matter_id, mm.user_id, u.username, COALESCE(u.display_name, u.username), mm.role, mm.can_export, "
            "mm.valid_from, mm.valid_to, mm.granted_by, mm.granted_at")
_MM_KEYS = ("id", "matter_id", "user_id", "username", "display_name", "role", "can_export", "valid_from", "valid_to",
            "granted_by", "granted_at")


def _membership(row) -> Optional[dict]:
    if not row:
        return None
    d = dict(zip(_MM_KEYS, row))
    d["can_export"] = bool(d["can_export"])
    return d


def _bump_epoch(conn, matter_id: str) -> None:
    conn.execute("UPDATE matters SET acl_epoch = acl_epoch + 1 WHERE id = ?", (matter_id,))


def list_live_memberships(matter_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {_MM_COLS} FROM matter_memberships mm JOIN users u ON u.id = mm.user_id "
                            f"WHERE mm.matter_id = :mid AND {_LIVE_MEMBERSHIP} ORDER BY mm.granted_at",
                            {"mid": matter_id, "now": now_iso()}).fetchall()
    return [_membership(r) for r in rows]


def get_live_membership(matter_id: str, membership_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(f"SELECT {_MM_COLS} FROM matter_memberships mm JOIN users u ON u.id = mm.user_id "
                           f"WHERE mm.id = :id AND mm.matter_id = :mid AND {_LIVE_MEMBERSHIP}",
                           {"id": membership_id, "mid": matter_id, "now": now_iso()}).fetchone()
    return _membership(row)


def add_membership(matter_id: str, user_id: str, role: str, granted_by: str, *, can_export: bool = False,
                   valid_from: Optional[str] = None, valid_to: Optional[str] = None) -> dict:
    mmid, now = _new_id(), now_iso()
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO matter_memberships (id, matter_id, user_id, role, can_export, valid_from, valid_to, "
                         "granted_by, granted_at) VALUES (?,?,?,?,?,?,?,?,?)",
                         (mmid, matter_id, user_id, role, 1 if can_export else 0, valid_from or now, valid_to, granted_by, now))
            _bump_epoch(conn, matter_id)
            conn.commit()
    except sqlite3.IntegrityError:
        raise Conflict("That person is already a member of this matter.")
    with get_conn() as conn:
        row = conn.execute(f"SELECT {_MM_COLS} FROM matter_memberships mm JOIN users u ON u.id = mm.user_id WHERE mm.id = ?",
                           (mmid,)).fetchone()
    return _membership(row)


def _live_leads(conn, matter_id: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM matter_memberships mm WHERE mm.matter_id = :mid AND mm.role = 'lead' "
                        f"AND {_LIVE_MEMBERSHIP}", {"mid": matter_id, "now": now_iso()}).fetchone()[0]


def revoke_membership(matter_id: str, membership_id: str, by: str, reason: str = "removed") -> None:
    with get_conn() as conn:
        row = conn.execute("SELECT role FROM matter_memberships WHERE id=? AND matter_id=? AND revoked_at IS NULL",
                           (membership_id, matter_id)).fetchone()
        if not row:
            raise LookupError("membership not found")
        if row[0] == "lead" and _live_leads(conn, matter_id) <= 1:
            raise Conflict("A matter must keep at least one lead.")
        conn.execute("UPDATE matter_memberships SET revoked_at=?, revoked_by=?, reason=? WHERE id=?",
                     (now_iso(), by, reason, membership_id))
        _bump_epoch(conn, matter_id)
        conn.commit()


def change_membership(matter_id: str, membership_id: str, by: str, *, role: Optional[str] = None,
                      can_export: Optional[bool] = None, valid_to: Optional[str] = None) -> dict:
    """Changes never overwrite: the old membership is closed (reason 'changed') and a new one opened."""
    current = get_live_membership(matter_id, membership_id)
    if not current:
        raise LookupError("membership not found")
    new_role = role or current["role"]
    with get_conn() as conn:
        if current["role"] == "lead" and new_role != "lead" and _live_leads(conn, matter_id) <= 1:
            raise Conflict("A matter must keep at least one lead.")
        now = now_iso()
        conn.execute("UPDATE matter_memberships SET revoked_at=?, revoked_by=?, reason='changed' WHERE id=?",
                     (now, by, membership_id))
        mmid = _new_id()
        conn.execute("INSERT INTO matter_memberships (id, matter_id, user_id, role, can_export, valid_from, valid_to, "
                     "granted_by, granted_at) VALUES (?,?,?,?,?,?,?,?,?)",
                     (mmid, matter_id, current["user_id"], new_role,
                      1 if (current["can_export"] if can_export is None else can_export) else 0,
                      now, valid_to if valid_to is not None else current["valid_to"], by, now))
        _bump_epoch(conn, matter_id)
        conn.commit()
    return get_live_membership(matter_id, mmid)


def active_colleague(org_id: str, username: str) -> Optional[dict]:
    """An active account in the same organisation (for adding to a matter)."""
    with get_conn() as conn:
        row = conn.execute("SELECT id, username FROM users WHERE username=? AND org_id=? AND COALESCE(status,'active')='active'",
                           ((username or "").strip(), org_id)).fetchone()
    return {"id": row[0], "username": row[1]} if row else None


def bump_epochs(matter_ids: list[str] = (), client_ids: list[str] = ()) -> None:
    """Signal that access to these matters (or every matter of these clients) changed."""
    with get_conn() as conn:
        for mid in matter_ids:
            conn.execute("UPDATE matters SET acl_epoch = acl_epoch + 1 WHERE id = ?", (mid,))
        for cid in client_ids:
            conn.execute("UPDATE matters SET acl_epoch = acl_epoch + 1 WHERE client_id = ?", (cid,))
        conn.commit()
