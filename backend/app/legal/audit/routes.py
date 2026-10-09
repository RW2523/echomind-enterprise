"""/api/legal/audit and /api/legal/matters/{id}/activity (S1.5, SEC-04).

The firm-wide trail is for the governance roles (audit.read.org). Matter numbers and titles are
shown to the roles that already hold the firm's matter directory (managing partners, records
administrators); a firm administrator sees only an opaque matter reference, so running the system
never reveals which matters exist. A matter's own activity is for its lead. Reading, verifying and
exporting the trail are themselves recorded.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Iterator, Literal, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from ...core.db import get_conn
from .. import policy
from ..matters.routes import principal_of, require, require_law_edition
from . import chain
from .record import audited, note

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])

_DIRECTORY_ROLES = frozenset({"managing_partner", "records_admin"})
_LIST_COLUMNS = ("seq", "ts", "actor_username", "actor_roles", "auth", "session_ref", "ip", "action",
                 "object_type", "object_id", "matter_id", "outcome", "reason", "details_json")


def _matter_labels(ids: set) -> dict:
    if not ids:
        return {}
    with get_conn() as conn:
        rows = conn.execute(f"SELECT id, number, title FROM matters WHERE id IN ({','.join('?' * len(ids))})",
                            tuple(ids)).fetchall()
    return {r[0]: {"id": r[0], "number": r[1], "title": r[2]} for r in rows}


def _events(where: str, params: list, limit: int, with_numbers: bool) -> dict:
    with get_conn() as conn:
        rows = [dict(zip(_LIST_COLUMNS, r)) for r in conn.execute(
            f"SELECT {', '.join(_LIST_COLUMNS)} FROM audit_events WHERE {where} ORDER BY seq DESC LIMIT ?",
            (*params, limit + 1)).fetchall()]
    more = len(rows) > limit
    rows = rows[:limit]
    labels = _matter_labels({r["matter_id"] for r in rows if r["matter_id"]}) if with_numbers else {}
    out = []
    for r in rows:
        mid = r["matter_id"]
        out.append({
            "seq": r["seq"], "ts": r["ts"], "action": r["action"], "outcome": r["outcome"], "reason": r["reason"],
            "actor": {"username": r["actor_username"], "roles": [x for x in (r["actor_roles"] or "").split(",") if x]}
            if r["actor_username"] else None,
            "auth": r["auth"], "session_ref": r["session_ref"], "ip": r["ip"],
            "object_type": r["object_type"], "object_id": r["object_id"],
            "matter": (labels.get(mid) or {"id": mid}) if mid else None,
            "details": json.loads(r["details_json"]) if r["details_json"] else None,
        })
    return {"events": out, "next_before": out[-1]["seq"] if more and out else None}


def _action_filter(action: Optional[str], where: list, params: list) -> None:
    if action:
        # The action itself or anything under it ("matter" matches matter.read, matter.update, ...).
        where.append("(action = ? OR substr(action, 1, ?) = ?)")
        params += [action, len(action) + 1, action + "."]


@router.get("/audit")
@audited("audit.read")
def org_audit(request: Request, action: Optional[str] = Query(None, max_length=80),
              actor: Optional[str] = Query(None, max_length=120), outcome: Optional[Literal["allow", "deny", "error"]] = None,
              matter: Optional[str] = Query(None, max_length=64), since: Optional[str] = Query(None, max_length=30),
              until: Optional[str] = Query(None, max_length=30), before: Optional[int] = None,
              limit: int = Query(50, ge=1, le=200)):
    p = principal_of(request)
    require(p, "audit.read.org")
    with_numbers = bool(policy.firm_roles_of(p["sub"]) & _DIRECTORY_ROLES)
    where, params = ["org_id = ?"], [p["org_id"]]
    _action_filter(action, where, params)
    if actor:
        where.append("actor_username = ?"); params.append(actor)
    if outcome:
        where.append("outcome = ?"); params.append(outcome)
    if matter and with_numbers:
        where.append("matter_id IN (SELECT id FROM matters WHERE org_id = ? AND number = ?)"); params += [p["org_id"], matter]
    if since:
        where.append("ts >= ?"); params.append(since)
    if until:
        where.append("ts < ?"); params.append(until)
    if before:
        where.append("seq < ?"); params.append(before)
    note(details={"action": action, "outcome": outcome, "limit": limit})
    return {**_events(" AND ".join(where), params, limit, with_numbers), "shows_matter_numbers": with_numbers}


@router.get("/audit/verify")
@audited("audit.verify")
def verify(request: Request):
    require(principal_of(request), "audit.read.org")
    result = chain.verify()
    note(details={"status": "intact" if result["ok"] else "broken", "events": result["events"]})
    return result


def _export_rows(org_id: str, since: Optional[str], until: Optional[str]) -> Iterator[dict]:
    for r in chain.rows():
        if r["org_id"] == org_id and (not since or r["ts"] >= since) and (not until or r["ts"] < until):
            yield r


# Reading order for people; JSONL keeps the stored columns.
_CSV_COLUMNS = ("seq", "ts", "actor_username", "actor_roles", "auth", "action", "object_type", "object_id", "matter_id",
                "outcome", "reason", "details_json", "ip", "session_ref", "actor_user_id", "org_id", "id", "prev_hash", "hash")


def _csv_cell(v):
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "@") else s       # never a spreadsheet formula


@router.get("/audit/export")
@audited("audit.export")
def export(request: Request, format: Literal["jsonl", "csv"] = "jsonl",
           since: Optional[str] = Query(None, max_length=30), until: Optional[str] = Query(None, max_length=30)):
    """JSONL is the verifiable form (exact rows and hashes; see chain.py for the rule). CSV is for reading."""
    p = principal_of(request)
    require(p, "audit.read.org")
    note(details={"format": format})
    rows = _export_rows(p["org_id"], since, until)

    if format == "jsonl":
        def gen():
            for r in rows:
                yield json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
        media, ext = "application/x-ndjson", "jsonl"
    else:
        def gen():
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(_CSV_COLUMNS)
            for r in rows:
                w.writerow([_csv_cell(r[c]) for c in _CSV_COLUMNS])
                yield buf.getvalue()
                buf.seek(0); buf.truncate()
            yield buf.getvalue()
        media, ext = "text/csv; charset=utf-8", "csv"
    return StreamingResponse(gen(), media_type=media,
                             headers={"Content-Disposition": f'attachment; filename="echomind-audit.{ext}"',
                                      "Cache-Control": "no-store"})


@router.get("/matters/{matter_id}/activity")
@audited("matter.activity.read", object_type="matter", object_param="matter_id")
def matter_activity(matter_id: str, request: Request, before: Optional[int] = None, limit: int = Query(50, ge=1, le=200)):
    require(principal_of(request), "audit.read.matter", matter_id)
    where, params = ["matter_id = ?"], [matter_id]
    if before:
        where.append("seq < ?"); params.append(before)
    return _events(" AND ".join(where), params, limit, with_numbers=True)     # the lead's own matter
