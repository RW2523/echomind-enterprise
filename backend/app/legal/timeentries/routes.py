"""Time-entry drafts (S4.3, OPT-03; decision D-056). Off unless LEGAL_TIME_ENTRIES is on (404 otherwise). Each person
sees and changes only their own entries.

- `POST /api/legal/time/suggest` {from, to}: suggest (or refresh) drafts from the person's own audited work.
- `GET  /api/legal/time?from&to`: their entries.
- `PATCH /api/legal/time/{id}` {minutes?, narrative?, version}: edit a draft.
- `POST /api/legal/time/{id}/approve|discard|reopen` {version}.
- `GET  /api/legal/time/export.csv?from&to`: approved entries as CSV. Nothing is sent to a billing system.
"""
from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ...core.config import settings
from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note
from ..matters.routes import principal_of, require_law_edition
from . import suggest as sg


def time_on() -> None:
    if not settings.LEGAL_TIME_ENTRIES:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition), Depends(time_on)])
_COLS = ("id", "matter_id", "work_date", "source", "minutes_suggested", "narrative_suggested", "sessions_json", "counts_json", "minutes",
         "narrative", "edited", "status", "created_at", "updated_at", "approved_at", "version")


def _bounds(frm: Optional[str], to: Optional[str]):
    today = date.today().isoformat()
    try:
        return sg.bounds(frm or today, to or frm or today)
    except sg.Invalid as e:
        raise HTTPException(status_code=422, detail=str(e))


def _out(conn, r) -> dict:
    d = dict(zip(_COLS, r))
    m = conn.execute("SELECT m.number, m.title, c.name FROM matters m JOIN clients c ON c.id = m.client_id WHERE m.id=?", (d["matter_id"],)).fetchone()
    return {"id": d["id"], "work_date": d["work_date"], "matter": {"id": d["matter_id"], "number": m[0], "title": m[1], "client": m[2]},
            "source": d["source"], "minutes": d["minutes"], "hours": sg.hours(d["minutes"]), "minutes_suggested": d["minutes_suggested"],
            "narrative": d["narrative"], "narrative_suggested": d["narrative_suggested"], "sessions": json.loads(d["sessions_json"]),
            "counts": json.loads(d["counts_json"]), "edited": bool(d["edited"]), "status": d["status"], "approved_at": d["approved_at"],
            "updated_at": d["updated_at"], "version": d["version"]}


def _entries(p: dict, a: date, b: date) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_time_entries WHERE user_id=? AND org_id=? AND work_date >= ? AND work_date <= ? "
                            "ORDER BY work_date, created_at", (p["sub"], p["org_id"], a.isoformat(), b.isoformat())).fetchall()
        return [_out(conn, r) for r in rows if policy.authorize(p, "matter.read", r[1]).allowed]


def _mine(p: dict, entry_id: str):
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_time_entries WHERE id=? AND user_id=? AND org_id=?",
                         (entry_id, p["sub"], p["org_id"])).fetchone()
    if not r or not policy.authorize(p, "matter.read", r[1]).allowed:
        raise HTTPException(status_code=404, detail="Not Found")
    return dict(zip(_COLS, r))


class RangeIn(BaseModel):
    from_: Optional[str] = Field(default=None, alias="from", max_length=10)
    to: Optional[str] = Field(default=None, max_length=10)


class EditIn(BaseModel):
    minutes: Optional[int] = None
    narrative: Optional[str] = Field(default=None, max_length=2000)
    version: int = Field(ge=1)


class VersionIn(BaseModel):
    version: int = Field(ge=1)


@router.post("/time/suggest")
@audited("time.suggest", object_type="time_entries")
def suggest(inp: RangeIn, request: Request):
    p = principal_of(request)
    a, b, start, end = _bounds(inp.from_, inp.to)
    found = sg.work(p["sub"], start, end, lambda mid: policy.authorize(p, "matter.read", mid).allowed)
    now, made, refreshed = policy.now_iso(), 0, 0
    with get_conn() as conn:
        for (mid, day), w in found.items():
            r = conn.execute("SELECT id, status, edited FROM legal_time_entries WHERE user_id=? AND matter_id=? AND work_date=? AND source='suggested'",
                             (p["sub"], mid, day)).fetchone()
            common = (w["minutes"], w["narrative"], json.dumps(w["sessions"]), json.dumps(w["counts"]))
            if r is None:
                conn.execute("INSERT INTO legal_time_entries (id, org_id, user_id, matter_id, work_date, source, minutes_suggested, narrative_suggested, "
                             "sessions_json, counts_json, minutes, narrative, edited, status, created_at, updated_at, version) "
                             "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (str(uuid.uuid4()), p["org_id"], p["sub"], mid, day, "suggested", *common, w["minutes"], w["narrative"], 0, "draft", now, now, 1))
                made += 1
            elif r[1] == "draft" and not r[2]:                    # untouched: follows the activity
                conn.execute("UPDATE legal_time_entries SET minutes_suggested=?, narrative_suggested=?, sessions_json=?, counts_json=?, minutes=?, "
                             "narrative=?, updated_at=?, version=version+1 WHERE id=?", (*common, w["minutes"], w["narrative"], now, r[0]))
                refreshed += 1
            elif r[1] == "draft":                                  # edited: the person's figures stand; the suggestion is kept beside them
                conn.execute("UPDATE legal_time_entries SET minutes_suggested=?, narrative_suggested=?, sessions_json=?, counts_json=?, updated_at=?, "
                             "version=version+1 WHERE id=?", (*common, now, r[0]))
                refreshed += 1
        conn.commit()
    note(details={"entries": made + refreshed, "new": made, "days": (b - a).days + 1})
    return {"entries": _entries(p, a, b), "timezone": settings.LEGAL_TIMEZONE}


@router.get("/time")
def list_entries(request: Request, from_: Optional[str] = None, to: Optional[str] = None):
    p = principal_of(request)
    q = request.query_params
    a, b, _, _ = _bounds(q.get("from") or from_, q.get("to") or to)
    return {"entries": _entries(p, a, b), "timezone": settings.LEGAL_TIMEZONE, "unit_minutes": sg.UNIT_MINUTES}


def _change(p: dict, entry_id: str, version: int, allowed: tuple[str, ...], sets: dict) -> dict:
    d = _mine(p, entry_id)
    if d["version"] != version:
        raise HTTPException(status_code=409, detail="This entry changed since you opened it. Reload it.")
    if d["status"] not in allowed:
        raise HTTPException(status_code=409, detail={"approved": "An approved entry is locked. Reopen it to change it.",
                                                     "discarded": "This entry was discarded.", "draft": "This entry is still a draft."}[d["status"]])
    cols = ", ".join(f"{k}=?" for k in sets)
    with get_conn() as conn:
        conn.execute(f"UPDATE legal_time_entries SET {cols}, updated_at=?, version=version+1 WHERE id=?", (*sets.values(), policy.now_iso(), entry_id))
        conn.commit()
        r = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_time_entries WHERE id=?", (entry_id,)).fetchone()
        return _out(conn, r)


@router.patch("/time/{entry_id}")
@audited("time.edit", object_type="time_entry", object_param="entry_id")
def edit(entry_id: str, inp: EditIn, request: Request):
    p = principal_of(request)
    sets: dict = {"edited": 1}
    if inp.minutes is not None:
        if inp.minutes < sg.UNIT_MINUTES or inp.minutes > 24 * 60 or inp.minutes % sg.UNIT_MINUTES:
            raise HTTPException(status_code=422, detail=f"Record time in {sg.UNIT_MINUTES}-minute units (0.1 hour), up to 24 hours.")
        sets["minutes"] = inp.minutes
    if inp.narrative is not None:
        text = " ".join(inp.narrative.split())
        if not text or len(text) > 1000:
            raise HTTPException(status_code=422, detail="Write a narrative of up to 1,000 characters.")
        sets["narrative"] = text
    out = _change(p, entry_id, inp.version, ("draft",), sets)
    note(details={"minutes": "minutes" in sets, "narrative": "narrative" in sets})
    return {"entry": out}


@router.post("/time/{entry_id}/approve")
@audited("time.approve", object_type="time_entry", object_param="entry_id")
def approve(entry_id: str, inp: VersionIn, request: Request):
    p = principal_of(request)
    return {"entry": _change(p, entry_id, inp.version, ("draft",), {"status": "approved", "approved_at": policy.now_iso()})}


@router.post("/time/{entry_id}/discard")
@audited("time.discard", object_type="time_entry", object_param="entry_id")
def discard(entry_id: str, inp: VersionIn, request: Request):
    p = principal_of(request)
    return {"entry": _change(p, entry_id, inp.version, ("draft",), {"status": "discarded"})}


@router.post("/time/{entry_id}/reopen")
@audited("time.reopen", object_type="time_entry", object_param="entry_id")
def reopen(entry_id: str, inp: VersionIn, request: Request):
    p = principal_of(request)
    return {"entry": _change(p, entry_id, inp.version, ("approved", "discarded"), {"status": "draft", "approved_at": None})}


def _cell(v) -> str:
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s       # a spreadsheet never runs a cell as a formula


@router.get("/time/export.csv")
@audited("time.export", object_type="time_entries")
def export(request: Request):
    p = principal_of(request)
    q = request.query_params
    a, b, _, _ = _bounds(q.get("from"), q.get("to"))
    rows = [e for e in _entries(p, a, b) if e["status"] == "approved"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Date", "Matter", "Client", "Hours", "Minutes", "Narrative", "Approved"])
    for e in rows:
        w.writerow([_cell(x) for x in (e["work_date"], e["matter"]["number"], e["matter"]["client"], e["hours"], e["minutes"], e["narrative"],
                                       (e["approved_at"] or "")[:10])])
    note(details={"entries": len(rows)})
    return Response(content=buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="time-entries-{a.isoformat()}-{b.isoformat()}.csv"', "Cache-Control": "no-store"})
