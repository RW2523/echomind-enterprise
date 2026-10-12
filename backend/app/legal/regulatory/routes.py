"""Regulatory updates (S4.4, OPT-04; decision D-068). Off unless LEGAL_REGULATORY is on (404 otherwise).

- `GET  /api/legal/regulatory?jurisdiction=UK|US&mine=1&days=`: recent updates, for the firm's lawyers, each with the
  open matters it may concern that the person can see, and why. The updates are public law; the matters are not.
- `GET  /api/legal/matters/{m}/regulatory`: the updates matched to a matter, newest first, with their decisions.
- `POST /api/legal/matters/{m}/regulatory/{update_id}` {status, note?, version}: a lawyer marks a match reviewed, not
  relevant (with a reason) or new again. The audit trail records the decision, never the reason's words.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.config import settings
from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note
from ..matters.routes import principal_of, require, require_law_edition


def regulatory_on() -> None:
    if not settings.LEGAL_REGULATORY:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition), Depends(regulatory_on)])
_UPDATE = ("id", "source", "external_id", "jurisdiction", "kind", "title", "summary", "url", "published", "subjects_json", "licence")
_SOURCE_NAMES = {"legislation.gov.uk": "legislation.gov.uk", "federalregister.gov": "Federal Register"}


def _update(r) -> dict:
    d = dict(zip(_UPDATE, r))
    d["subjects"] = json.loads(d.pop("subjects_json"))
    d["source_name"] = _SOURCE_NAMES[d["source"]]
    return d


def _match(r, users: dict) -> dict:
    terms, status, decided_by, decided_at, note_, version = r
    return {"terms": json.loads(terms), "status": status, "decided_by": users.get(decided_by), "decided_at": decided_at,
            "note": note_, "version": version}


def _usernames(conn, ids) -> dict:
    ids = [i for i in set(ids) if i]
    if not ids:
        return {}
    return dict(conn.execute(f"SELECT id, username FROM users WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall())


@router.get("/regulatory")
@audited("regulatory.read", object_type="regulatory_updates")
def list_updates(request: Request, jurisdiction: Optional[Literal["UK", "US"]] = None, mine: bool = False, days: int = 30):
    p = principal_of(request)
    require(p, "regulatory.read")
    since = (date.today() - timedelta(days=max(1, min(days, 365)))).isoformat()
    args: dict = {"org": p["org_id"], "uid": p["sub"], "now": policy.now_iso(), "since": since}
    where = "u.org_id = :org AND u.published >= :since"
    if jurisdiction:
        where += " AND u.jurisdiction = :j"
        args["j"] = jurisdiction
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join('u.' + c for c in _UPDATE)} FROM legal_regulatory_updates u WHERE {where} "
                            "ORDER BY u.published DESC, u.title LIMIT 300", args).fetchall()
        seen: dict[str, list] = {}
        for uid, mid, number, title, status in conn.execute(
                "SELECT r.update_id, m.id, m.number, m.title, r.status FROM legal_regulatory_matches r JOIN matters m ON m.id = r.matter_id "
                f"JOIN legal_regulatory_updates u ON u.id = r.update_id WHERE {where} AND m.status = 'open' AND "
                f"{policy.visible_matters_sql()} ORDER BY m.number", args):
            seen.setdefault(uid, []).append({"id": mid, "number": number, "title": title, "status": status})
    out = [{**_update(r), "matters": seen.get(r[0], [])} for r in rows]
    if mine:
        out = [u for u in out if u["matters"]]
    note(details={"updates": len(out), "matched": sum(1 for u in out if u["matters"])})
    return {"updates": out, "since": since}


@router.get("/matters/{matter_id}/regulatory")
@audited("regulatory.read.matter", object_type="matter", object_param="matter_id")
def for_matter(matter_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join('u.' + c for c in _UPDATE)}, r.terms_json, r.status, r.decided_by, r.decided_at, r.note, r.version "
                            "FROM legal_regulatory_matches r JOIN legal_regulatory_updates u ON u.id = r.update_id "
                            "WHERE r.matter_id=? AND u.org_id=? ORDER BY u.published DESC, u.title", (matter_id, p["org_id"])).fetchall()
        users = _usernames(conn, [r[len(_UPDATE) + 2] for r in rows])
    n = len(_UPDATE)
    note(matter_id=matter_id, details={"matches": len(rows)})
    return {"matches": [{"update": _update(r[:n]), **_match(r[n:], users)} for r in rows]}


class DecisionIn(BaseModel):
    status: Literal["reviewed", "not_relevant", "new"]
    note: Optional[str] = Field(default=None, max_length=1000)
    version: int = Field(ge=1)


@router.post("/matters/{matter_id}/regulatory/{update_id}")
@audited("regulatory.decide", object_type="regulatory_update", object_param="update_id")
def decide(matter_id: str, update_id: str, inp: DecisionIn, request: Request):
    p = principal_of(request)
    require(p, "regulatory.decide", matter_id)
    text = " ".join((inp.note or "").split()) or None
    if inp.status == "not_relevant" and not text:
        raise HTTPException(status_code=422, detail="Give a reason why this update does not concern the matter.")
    with get_conn() as conn:
        r = conn.execute("SELECT r.version FROM legal_regulatory_matches r JOIN legal_regulatory_updates u ON u.id = r.update_id "
                         "WHERE r.update_id=? AND r.matter_id=? AND u.org_id=?", (update_id, matter_id, p["org_id"])).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Not Found")
        if r[0] != inp.version:
            raise HTTPException(status_code=409, detail="This update's decision changed since you opened it. Reload it.")
        decided = inp.status != "new"
        conn.execute("UPDATE legal_regulatory_matches SET status=?, decided_by=?, decided_at=?, note=?, version=version+1 "
                     "WHERE update_id=? AND matter_id=?", (inp.status, p["sub"] if decided else None, policy.now_iso() if decided else None,
                                                           text if decided else None, update_id, matter_id))
        conn.commit()
        row = conn.execute("SELECT terms_json, status, decided_by, decided_at, note, version FROM legal_regulatory_matches "
                           "WHERE update_id=? AND matter_id=?", (update_id, matter_id)).fetchone()
        users = _usernames(conn, [row[2]])
    note(matter_id=matter_id, details={"status": inp.status, "reason_given": bool(text)})
    return {"match": _match(row, users)}
