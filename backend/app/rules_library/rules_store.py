"""SQLite persistence for Rules Library (rule sets, rules, session activations)."""
from __future__ import annotations

import sqlite3
from typing import Any, List, Optional

from ..core.db import get_conn
from ..utils.ids import new_id, now_iso
from .matching import first_matching_rule


def _as_rule_set_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["is_active_default"] = bool(d.get("is_active_default"))
    return d


def _as_rule_dict(row: sqlite3.Row) -> dict:
    return dict(row)


def create_rule_set(
    name: str,
    description: str = "",
    version: str = "1.0.0",
    priority: int = 0,
    is_active_default: bool = False,
    source_policy_text: Optional[str] = None,
) -> dict:
    rid = new_id("rset")
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO rule_sets (id, name, description, version, priority, is_active_default, "
            "source_policy_text, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                rid,
                name.strip(),
                description or "",
                version or "1.0.0",
                int(priority),
                1 if is_active_default else 0,
                source_policy_text,
                ts,
                ts,
            ),
        )
        conn.commit()
    row = get_rule_set(rid)
    assert row is not None
    return row


def get_rule_set(rule_set_id: str) -> Optional[dict]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM rule_sets WHERE id = ?", (rule_set_id,)).fetchone()
        if row is None:
            return None
        return _as_rule_set_dict(row)


def list_rule_sets() -> List[dict]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM rule_sets ORDER BY priority DESC, name ASC").fetchall()
    return [_as_rule_set_dict(r) for r in rows]


def update_rule_set(
    rule_set_id: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
    priority: Optional[int] = None,
    is_active_default: Optional[bool] = None,
    source_policy_text: Optional[str] = None,
) -> Optional[dict]:
    cur = get_rule_set(rule_set_id)
    if not cur:
        return None
    fields: List[str] = []
    vals: List[Any] = []
    if name is not None:
        fields.append("name = ?")
        vals.append(name.strip())
    if description is not None:
        fields.append("description = ?")
        vals.append(description)
    if version is not None:
        fields.append("version = ?")
        vals.append(version)
    if priority is not None:
        fields.append("priority = ?")
        vals.append(int(priority))
    if is_active_default is not None:
        fields.append("is_active_default = ?")
        vals.append(1 if is_active_default else 0)
    if source_policy_text is not None:
        fields.append("source_policy_text = ?")
        vals.append(source_policy_text)
    if not fields:
        return cur
    now = now_iso()
    fields.append("updated_at = ?")
    vals.append(now)
    vals.append(rule_set_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE rule_sets SET {', '.join(fields)} WHERE id = ?", vals)
        conn.commit()
    return get_rule_set(rule_set_id)


def delete_rule_set(rule_set_id: str) -> bool:
    with get_conn() as conn:
        conn.execute("DELETE FROM rules WHERE rule_set_id = ?", (rule_set_id,))
        conn.execute("DELETE FROM session_rule_activations WHERE rule_set_id = ?", (rule_set_id,))
        cur = conn.execute("DELETE FROM rule_sets WHERE id = ?", (rule_set_id,))
        conn.commit()
        return cur.rowcount > 0


def create_rule(
    rule_set_id: str,
    title: str,
    text: str,
    severity: str = "medium",
    category: str = "general",
) -> Optional[dict]:
    if not get_rule_set(rule_set_id):
        return None
    rid = new_id("rule")
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO rules (id, rule_set_id, title, text, severity, category, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (rid, rule_set_id, title.strip(), text, severity or "medium", category or "general", ts, ts),
        )
        conn.commit()
    return get_rule(rid)


def get_rule(rule_id: str) -> Optional[dict]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)).fetchone()
        if row is None:
            return None
        return _as_rule_dict(row)


def list_rules(rule_set_id: str) -> List[dict]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM rules WHERE rule_set_id = ? ORDER BY title ASC",
            (rule_set_id,),
        ).fetchall()
    return [_as_rule_dict(r) for r in rows]


def update_rule(
    rule_id: str,
    *,
    title: Optional[str] = None,
    text: Optional[str] = None,
    severity: Optional[str] = None,
    category: Optional[str] = None,
) -> Optional[dict]:
    if not get_rule(rule_id):
        return None
    fields: List[str] = []
    vals: List[Any] = []
    if title is not None:
        fields.append("title = ?")
        vals.append(title.strip())
    if text is not None:
        fields.append("text = ?")
        vals.append(text)
    if severity is not None:
        fields.append("severity = ?")
        vals.append(severity)
    if category is not None:
        fields.append("category = ?")
        vals.append(category)
    if not fields:
        return get_rule(rule_id)
    now = now_iso()
    fields.append("updated_at = ?")
    vals.append(now)
    vals.append(rule_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE rules SET {', '.join(fields)} WHERE id = ?", vals)
        conn.commit()
    return get_rule(rule_id)


def delete_rule(rule_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
        conn.commit()
        return cur.rowcount > 0


def set_session_rule_activation(
    session_id: str,
    rule_set_id: str,
    *,
    enabled: bool,
    priority_override: Optional[int] = None,
) -> dict:
    """Upsert activation for (session_id, rule_set_id)."""
    if not get_rule_set(rule_set_id):
        raise ValueError("unknown_rule_set")
    ts = now_iso()
    aid = new_id("sra")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM session_rule_activations WHERE session_id = ? AND rule_set_id = ?",
            (session_id, rule_set_id),
        ).fetchone()
        if row:
            eid = row[0]
            conn.execute(
                "UPDATE session_rule_activations SET enabled = ?, priority_override = ?, updated_at = ? WHERE id = ?",
                (1 if enabled else 0, priority_override, ts, eid),
            )
        else:
            conn.execute(
                "INSERT INTO session_rule_activations (id, session_id, rule_set_id, enabled, priority_override, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (aid, session_id, rule_set_id, 1 if enabled else 0, priority_override, ts, ts),
            )
        conn.commit()
    return list_session_activations(session_id)


def list_session_activations(session_id: str) -> dict:
    """Return { activations: [...] } with rule set metadata."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT sra.id, sra.session_id, sra.rule_set_id, sra.enabled, sra.priority_override, "
            "sra.created_at, sra.updated_at, rs.name as rule_set_name, rs.version, rs.priority as rule_set_priority "
            "FROM session_rule_activations sra "
            "JOIN rule_sets rs ON rs.id = sra.rule_set_id "
            "WHERE sra.session_id = ? "
            "ORDER BY COALESCE(sra.priority_override, rs.priority) DESC, rs.name ASC",
            (session_id,),
        ).fetchall()
    acts = []
    for r in rows:
        acts.append(
            {
                "id": r[0],
                "session_id": r[1],
                "rule_set_id": r[2],
                "enabled": bool(r[3]),
                "priority_override": r[4],
                "created_at": r[5],
                "updated_at": r[6],
                "rule_set_name": r[7],
                "rule_set_version": r[8],
                "rule_set_priority": int(r[9] or 0),
            }
        )
    return {"activations": acts, "session_id": session_id}


def list_enabled_rules_for_session(session_id: str) -> List[dict]:
    """
    Ordered rules from enabled activations. Each dict: rule fields + rule_set_id, rule_set_name,
    effective_priority (int).
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT r.id, r.rule_set_id, r.title, r.text, r.severity, r.category, rs.name as rule_set_name, "
            "COALESCE(sra.priority_override, rs.priority) as eff_pri "
            "FROM session_rule_activations sra "
            "JOIN rule_sets rs ON rs.id = sra.rule_set_id "
            "JOIN rules r ON r.rule_set_id = rs.id "
            "WHERE sra.session_id = ? AND sra.enabled = 1 "
            "ORDER BY eff_pri DESC, rs.name ASC, r.title ASC",
            (session_id,),
        ).fetchall()
    out: List[dict] = []
    for row in rows:
        out.append(
            {
                "id": row[0],
                "rule_set_id": row[1],
                "title": row[2],
                "text": row[3],
                "severity": row[4],
                "category": row[5],
                "rule_set_name": row[6],
                "effective_priority": int(row[7] or 0),
            }
        )
    return out


def match_first_enabled_rule(session_id: str, transcript: str) -> Optional[dict]:
    rows = list_enabled_rules_for_session(session_id)
    return first_matching_rule(transcript, rows)
