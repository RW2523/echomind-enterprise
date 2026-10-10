"""Obligations and deadlines (S3.6, DOC-05; decision D-050).

- `POST /api/legal/matters/{id}/documents/{did}/obligations/extract`: read a document's obligations (`ai.draft`).
  Reading again adds only what is new, and keeps every decision.
- `GET  …/obligations`, `GET …/obligations/{oid}` (with its history): for those who can see the document.
- `POST …/obligations`: a lawyer adds one from the document's exact words (`ai.draft`).
- `PATCH …/obligations/{oid}`: confirm or dismiss (a lawyer: lead, attorney, reviewer), edit, mark done or reopen
  (`ai.draft`). A reason is required to edit or dismiss; each change is a version.
- `GET …/obligations.ics`: confirmed, dated obligations as a calendar file, only when the firm turns it on.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from typing import Literal, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...core.config import settings
from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note
from ..documents import numbering
from ..documents import repo as documents
from ..matters import repo as matters
from ..matters.routes import principal_of, require, require_law_edition
from . import extract as ex
from . import model

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])
_COLS = ("id", "matter_id", "document_id", "version_id", "clause", "kind", "party", "action_text", "when_text", "rule_json", "due_date",
         "due_note", "timezone", "status", "owner_id", "evidence_json", "source", "confirmed_by", "confirmed_at", "version", "created_by",
         "created_at", "updated_at")


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


class AddIn(BaseModel):
    document_id: str = Field(min_length=1, max_length=64)
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    kind: Literal[ex.KINDS] = "notice"  # type: ignore[valid-type]
    action: str = Field(min_length=1, max_length=300)
    party: Optional[str] = Field(default=None, max_length=80)
    when: Optional[str] = Field(default=None, max_length=300)
    due_date: Optional[date] = None
    timezone: Optional[str] = Field(default=None, max_length=64)


class PatchIn(BaseModel):
    expected_version: int = Field(ge=1)
    action: Literal["confirm", "edit", "done", "dismiss", "reopen"]
    reason: Optional[str] = Field(default=None, max_length=500)
    owner: Optional[str] = Field(default=None, max_length=80)
    due_date: Optional[date] = None
    timezone: Optional[str] = Field(default=None, max_length=64)
    party: Optional[str] = Field(default=None, max_length=80)
    action_text: Optional[str] = Field(default=None, max_length=300)
    when: Optional[str] = Field(default=None, max_length=300)


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise HTTPException(status_code=422, detail=f"{name} is not a timezone (use a name such as Europe/London).")


def _username(conn, uid: Optional[str]) -> Optional[str]:
    if not uid:
        return None
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
    return r[0] if r else None


def _out(conn, row) -> dict:
    d = dict(zip(_COLS, row))
    due_at, overdue = None, False
    if d["due_date"]:
        z = ZoneInfo(d["timezone"])
        end = datetime.combine(date.fromisoformat(d["due_date"]), datetime.max.time().replace(microsecond=0), tzinfo=z)
        due_at = end.isoformat()
        overdue = d["status"] in ("proposed", "confirmed") and end < datetime.now(timezone.utc)
    title = conn.execute("SELECT title FROM legal_documents WHERE id=?", (d["document_id"],)).fetchone()
    return {"id": d["id"], "document_id": d["document_id"], "document_title": title[0] if title else None, "version_id": d["version_id"],
            "clause": d["clause"], "kind": d["kind"], "party": d["party"], "action": d["action_text"], "when": d["when_text"],
            "rule": json.loads(d["rule_json"]), "due_date": d["due_date"], "due_note": d["due_note"], "timezone": d["timezone"],
            "due_at": due_at, "overdue": overdue, "status": d["status"], "owner": _username(conn, d["owner_id"]),
            "evidence": json.loads(d["evidence_json"]), "source": d["source"], "confirmed_by": _username(conn, d["confirmed_by"]),
            "confirmed_at": d["confirmed_at"], "version": d["version"], "created_by": _username(conn, d["created_by"]),
            "created_at": d["created_at"], "updated_at": d["updated_at"]}


def _event(conn, oid: str, action: str, by: str, changes: Optional[dict] = None, reason: Optional[str] = None) -> None:
    conn.execute("INSERT INTO legal_obligation_events (id, obligation_id, action, changes_json, reason, by, at) VALUES (?,?,?,?,?,?,?)",
                 (str(uuid.uuid4()), oid, action, json.dumps(changes or {}, ensure_ascii=False), reason, by, policy.now_iso()))


def _visible_rows(p: dict, matter_id: str, where: str = "", args: tuple = ()) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_obligations WHERE matter_id=? {where} "
                            "ORDER BY due_date IS NULL, due_date, clause, created_at", (matter_id, *args)).fetchall()
        seen: dict[str, bool] = {}
        out = []
        for r in rows:
            did = r[2]
            if did not in seen:
                seen[did] = documents.visible(matter_id, did, p["sub"])
            if seen[did]:
                out.append(_out(conn, r))
        return out


def _one(p: dict, matter_id: str, oid: str) -> dict:
    found = _visible_rows(p, matter_id, "AND id=?", (oid,))
    if not found:
        raise _not_found()
    return found[0]


def _text(matter_id: str, document_id: str) -> tuple[str, str, list[dict]]:
    with get_conn() as conn:
        r = conn.execute("SELECT v.id, t.text FROM legal_documents d JOIN legal_document_versions v ON v.id = d.current_version_id "
                         "JOIN legal_version_text t ON t.version_id = v.id WHERE d.id=? AND d.matter_id=?", (document_id, matter_id)).fetchone()
        if not r:
            raise LookupError(document_id)
        rows = [{"number": n, "heading": h, "level": lv, "parent": pa, "start": s, "end": e} for n, h, lv, pa, s, e in conn.execute(
            "SELECT number, heading, level, parent, char_start, char_end FROM legal_clauses WHERE version_id=? ORDER BY char_start, level", (r[0],))]
    return r[0], r[1], rows


@router.post("/matters/{matter_id}/documents/{document_id}/obligations/extract")
@audited("obligation.extract", object_type="document", object_param="document_id")
async def extract_obligations(matter_id: str, document_id: str, request: Request):
    p = principal_of(request)
    await run_in_threadpool(require, p, "ai.draft", matter_id)
    if not await run_in_threadpool(documents.visible, matter_id, document_id, p["sub"]):
        raise _not_found()
    try:
        vid, text, rows = await run_in_threadpool(_text, matter_id, document_id)
    except LookupError:
        raise HTTPException(status_code=409, detail="This document has not been read yet.")
    found = ex.read(text, rows)
    model_state, stats = "off", {}
    if settings.LEGAL_OBLIGATIONS_MODEL:
        try:
            more, stats = await model.propose(text, rows, found)
            found, model_state = found + more, "used"
        except Exception:                                   # noqa: BLE001 — the rules' findings stand on their own
            model_state = "unavailable"
    out, added = await run_in_threadpool(_store, p, matter_id, document_id, vid, found)
    note(details={"found": len(found), "added": added, "model_used": model_state == "used", "model_unavailable": model_state == "unavailable",
                  "model_kept": stats.get("kept", 0), "model_dropped": stats.get("dropped", 0)})   # the audit keeps no strings
    return {**out, "model": model_state}


def _store(p: dict, matter_id: str, document_id: str, vid: str, found: list[dict]) -> tuple[dict, int]:
    added = 0
    now = policy.now_iso()
    with get_conn() as conn:
        have = {(k, s) for k, s in conn.execute("SELECT kind, json_extract(evidence_json, '$.start') FROM legal_obligations "
                                                "WHERE document_id=? AND version_id=?", (document_id, vid))}
        for o in found:
            if (o["kind"], o["evidence"]["start"]) in have:
                continue                                    # already read: its decisions stay as they are
            oid = str(uuid.uuid4())
            conn.execute("INSERT INTO legal_obligations (id, matter_id, document_id, version_id, clause, kind, party, action_text, when_text, "
                         "rule_json, due_date, due_note, timezone, status, owner_id, evidence_json, source, confirmed_by, confirmed_at, version, "
                         "created_by, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (oid, matter_id, document_id, vid, o["clause"], o["kind"], o["party"], o["action"], o["when"],
                          json.dumps(o["rule"]), o["due_date"], o["due_note"], settings.LEGAL_TIMEZONE, "proposed", None,
                          json.dumps(o["evidence"], ensure_ascii=False), o.get("source", "rules"), None, None, 1, p["sub"], now, now))
            _event(conn, oid, "proposed", p["sub"])
            added += 1
        conn.commit()
    return {"obligations": _visible_rows(p, matter_id, "AND document_id=?", (document_id,)), "added": added}, added


@router.get("/matters/{matter_id}/obligations")
def list_obligations(matter_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    return {"obligations": _visible_rows(p, matter_id), "timezone": settings.LEGAL_TIMEZONE, "calendar": settings.LEGAL_OBLIGATIONS_ICS}


@router.get("/matters/{matter_id}/obligations.ics")
@audited("obligation.ics", object_type="matter", object_param="matter_id")
def calendar_file(matter_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    if not settings.LEGAL_OBLIGATIONS_ICS:
        raise _not_found()                                  # off unless the firm turns it on
    m = matters.get_matter(matter_id) or {}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//EchoMind Law//Obligations//EN", "CALSCALE:GREGORIAN"]
    events = [o for o in _visible_rows(p, matter_id) if o["status"] == "confirmed" and o["due_date"]]
    for o in events:
        esc = lambda v: (v or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")  # noqa: E731
        lines += ["BEGIN:VEVENT", f"UID:{o['id']}@echomind-law", f"DTSTAMP:{stamp}", f"DTSTART;VALUE=DATE:{o['due_date'].replace('-', '')}",
                  f"SUMMARY:{esc(m.get('number', ''))}: {esc(o['action'])}",
                  f"DESCRIPTION:{esc((o['party'] or '') + ' — ' + (o['when'] or '') + ' (clause ' + (o['clause'] or '—') + ')')}",
                  "END:VEVENT"]
    lines.append("END:VCALENDAR")
    note(details={"events": len(events)})
    return Response("\r\n".join(lines) + "\r\n", media_type="text/calendar; charset=utf-8",
                    headers={"Cache-Control": "no-store", "Content-Disposition": 'attachment; filename="obligations.ics"'})


@router.get("/matters/{matter_id}/obligations/{obligation_id}")
def read_obligation(matter_id: str, obligation_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    o = _one(p, matter_id, obligation_id)
    with get_conn() as conn:
        hist = [{"action": a, "changes": json.loads(c), "reason": r, "by": _username(conn, by), "at": at} for a, c, r, by, at in conn.execute(
            "SELECT action, changes_json, reason, by, at FROM legal_obligation_events WHERE obligation_id=? ORDER BY at, rowid", (obligation_id,))]
    return {"obligation": o, "history": hist}


@router.post("/matters/{matter_id}/obligations")
@audited("obligation.create", object_type="document")
def add_obligation(matter_id: str, inp: AddIn, request: Request):
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    if not documents.visible(matter_id, inp.document_id, p["sub"]):
        raise _not_found()
    try:
        vid, text, rows = _text(matter_id, inp.document_id)
    except LookupError:
        raise HTTPException(status_code=409, detail="This document has not been read yet.")
    if not (inp.start < inp.end <= len(text)) or not text[inp.start:inp.end].strip():
        raise HTTPException(status_code=422, detail="Choose the words of the document the obligation comes from.")
    tz = inp.timezone or settings.LEGAL_TIMEZONE
    _zone(tz)
    all_rows = numbering.complete(text, rows)
    inside = [c for c in all_rows if c["start"] <= inp.start < max(c["end"], c["start"] + 1) and numbering.split(c["number"])]
    clause = max(inside, key=lambda c: (c["level"], c["start"]))["number"] if inside else None
    oid, now = str(uuid.uuid4()), policy.now_iso()
    evidence = {"start": inp.start, "end": inp.end, "quote": text[inp.start:inp.end]}
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_obligations (id, matter_id, document_id, version_id, clause, kind, party, action_text, when_text, "
                     "rule_json, due_date, due_note, timezone, status, owner_id, evidence_json, source, confirmed_by, confirmed_at, version, "
                     "created_by, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (oid, matter_id, inp.document_id, vid, clause, inp.kind, inp.party, inp.action.strip(), (inp.when or "").strip() or None,
                      "{}", inp.due_date.isoformat() if inp.due_date else None, "Set by a lawyer when added." if inp.due_date else None,
                      tz, "proposed", None, json.dumps(evidence, ensure_ascii=False), "person", None, None, 1, p["sub"], now, now))
        _event(conn, oid, "proposed", p["sub"])
        conn.commit()
    note(object_id=inp.document_id, details={"obligation": oid, "kind": inp.kind})
    return {"obligation": _one(p, matter_id, oid)}


@router.patch("/matters/{matter_id}/obligations/{obligation_id}")
@audited("obligation.update", object_type="obligation", object_param="obligation_id")
def update_obligation(matter_id: str, obligation_id: str, inp: PatchIn, request: Request):
    p = principal_of(request)
    require(p, "artifacts.approve" if inp.action in ("confirm", "dismiss") else "ai.draft", matter_id)
    cur = _one(p, matter_id, obligation_id)
    if cur["version"] != inp.expected_version:
        raise HTTPException(status_code=409, detail="This obligation changed since you opened it. Reload and try again.")
    reason = (inp.reason or "").strip() or None
    if inp.action in ("edit", "dismiss") and (not reason or len(reason) < 3):
        raise HTTPException(status_code=422, detail="Say why.")
    sets: dict = {}
    changes: dict = {}

    def change(col: str, key: str, new):
        if new is not None and new != cur[key]:
            sets[col] = new
            changes[key] = [cur[key], new]
    if inp.owner is not None:
        with get_conn() as conn:
            u = conn.execute("SELECT id FROM users WHERE username=?", (inp.owner,)).fetchone()
        if not u or not any(m["user_id"] == u[0] for m in matters.list_live_memberships(matter_id)):
            raise HTTPException(status_code=422, detail=f"{inp.owner} is not a member of this matter.")
        if inp.owner != cur["owner"]:
            sets["owner_id"], changes["owner"] = u[0], [cur["owner"], inp.owner]
    if inp.action == "edit":
        if inp.timezone is not None:
            _zone(inp.timezone)
        change("due_date", "due_date", inp.due_date.isoformat() if inp.due_date else None)
        change("timezone", "timezone", inp.timezone)
        change("party", "party", inp.party)
        change("action_text", "action", inp.action_text)
        change("when_text", "when", inp.when)
        if "due_date" in changes:
            sets["due_note"] = f"Set by {_username_of(p['sub'])}: {reason}"
        if not changes:
            raise HTTPException(status_code=422, detail="Nothing was changed.")
    status = {"confirm": "confirmed", "done": "done", "dismiss": "dismissed", "reopen": "proposed"}.get(inp.action)
    if status:
        if status == cur["status"]:
            raise HTTPException(status_code=409, detail=f"This obligation is already {status}.")
        sets["status"], changes["status"] = status, [cur["status"], status]
        if status == "confirmed":
            sets["confirmed_by"], sets["confirmed_at"] = p["sub"], policy.now_iso()
    sets["version"], sets["updated_at"] = cur["version"] + 1, policy.now_iso()
    cols = ", ".join(f"{k}=?" for k in sets)
    with get_conn() as conn:
        n = conn.execute(f"UPDATE legal_obligations SET {cols} WHERE id=? AND version=?", (*sets.values(), obligation_id, cur["version"])).rowcount
        if n != 1:
            raise HTTPException(status_code=409, detail="This obligation changed since you opened it. Reload and try again.")
        _event(conn, obligation_id, inp.action, p["sub"], changes, reason)
        conn.commit()
    note(details={"action": inp.action, "fields": sorted(k for k in changes if k != "status"), "version": cur["version"] + 1})
    return {"obligation": _one(p, matter_id, obligation_id)}


def _username_of(uid: str) -> str:
    with get_conn() as conn:
        return _username(conn, uid) or "a lawyer"
