"""SQLite persistence for Document Studio generation jobs."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from ..core.db import get_conn
from ..utils.ids import new_id, now_iso

logger = logging.getLogger(__name__)

# status: queued -> planning -> writing -> assembling -> done | error
ACTIVE_STATUSES = ("queued", "planning", "writing", "assembling")


def create_job(*, title: str, template_id: str, persona: str, mode: str) -> str:
    jid = new_id("doc")
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO docgen_jobs
               (id, title, template_id, persona, mode, status, stage, doc_json, error, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (jid, title, template_id, persona, mode, "queued", "queued", None, None, ts, ts),
        )
        conn.commit()
    return jid


def update_status(job_id: str, status: str, *, stage: Optional[str] = None,
                  error: Optional[str] = None, doc: Optional[Dict[str, Any]] = None) -> None:
    sets = ["status = ?", "updated_at = ?"]
    vals: list = [status, now_iso()]
    if stage is not None:
        sets.append("stage = ?"); vals.append(stage)
    if error is not None:
        sets.append("error = ?"); vals.append(error)
    if doc is not None:
        sets.append("doc_json = ?"); vals.append(json.dumps(doc))
    vals.append(job_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE docgen_jobs SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT id, title, template_id, persona, mode, status, stage, doc_json, error, created_at, updated_at
               FROM docgen_jobs WHERE id = ?""",
            (job_id,),
        ).fetchone()
    if not row:
        return None
    doc = None
    if row[7]:
        try:
            doc = json.loads(row[7])
        except Exception:
            doc = None
    return {
        "id": row[0], "title": row[1], "template_id": row[2], "persona": row[3], "mode": row[4],
        "status": row[5], "stage": row[6], "document": doc, "error": row[8],
        "created_at": row[9], "updated_at": row[10],
    }


def list_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, title, template_id, persona, status, created_at, updated_at
               FROM docgen_jobs ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [
        {"id": r[0], "title": r[1], "template_id": r[2], "persona": r[3],
         "status": r[4], "created_at": r[5], "updated_at": r[6]}
        for r in rows
    ]


def delete_job(job_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM docgen_jobs WHERE id = ?", (job_id,))
        conn.commit()
        return cur.rowcount > 0


# ── Custom (uploaded) templates ──────────────────────────────────────────────

def save_custom_template(blueprint: Dict[str, Any], source_path: Optional[str] = None) -> str:
    """Persist a parsed custom template blueprint. Returns its (namespaced) id."""
    tid = "ctpl_" + new_id("t")
    blueprint = dict(blueprint or {})
    blueprint["id"] = tid
    blueprint["custom"] = True
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO docgen_templates (id, name, blueprint_json, source_path, created_at) VALUES (?,?,?,?,?)",
            (tid, blueprint.get("name", "Custom Template"), json.dumps(blueprint), source_path, now_iso()),
        )
        conn.commit()
    return tid


def get_custom_template(template_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT blueprint_json, source_path FROM docgen_templates WHERE id = ?", (template_id,)
        ).fetchone()
    if not row:
        return None
    try:
        bp = json.loads(row[0])
    except Exception:
        return None
    bp["_source_path"] = row[1]
    return bp


def list_custom_templates() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT blueprint_json FROM docgen_templates ORDER BY created_at DESC"
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for (bj,) in rows:
        try:
            bp = json.loads(bj)
        except Exception:
            continue
        out.append({
            "id": bp.get("id"), "name": bp.get("name", "Custom Template"), "icon": "File",
            "persona": bp.get("persona", "General Assistant"),
            "description": bp.get("description", "Uploaded custom template."),
            "sections": [s.get("title", "") for s in bp.get("section_blueprint", [])],
            "default_doc_type": bp.get("default_doc_type", "Document"),
            "theme": bp.get("theme", "midnight"), "supports_images": False, "custom": True,
            "source_format": bp.get("source_format", ""),
        })
    return out


def delete_custom_template(template_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM docgen_templates WHERE id = ?", (template_id,))
        conn.commit()
        return cur.rowcount > 0
