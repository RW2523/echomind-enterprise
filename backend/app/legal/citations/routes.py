"""Citation and quote checks (S3.4, DOC-03; decision D-048).

- `POST /api/legal/matters/{id}/documents/{did}/citation-check`: check a document (`research.query`: any member who
  can see it, a reviewer checking a memo before approving it included; IMPLEMENTATION_PLAN §5). No model is used and
  nothing is looked up outside the firm (P-009).
- `GET  …/documents/{did}/citation-checks`, `GET …/citation-checks/{cid}`: for those who can see the document;
  each item's target is shown only to a person who can open the document it points to.
- `POST …/citation-checks/{cid}/notes`: a lawyer notes an item (lead, attorney, reviewer). Notes are append-only.
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note
from ..documents import repo as documents
from ..matters.routes import principal_of, require, require_law_edition
from . import check as checkmod

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


class CheckIn(BaseModel):
    version_id: Optional[str] = Field(default=None, max_length=64)


class NoteIn(BaseModel):
    item: str = Field(min_length=1, max_length=16)
    note: str = Field(min_length=1, max_length=1000)


_COLS = "id, matter_id, document_id, version_id, results_json, created_by, created_at"


def _out(conn, r, uid: str) -> dict:
    cid, mid, did, vid, rj, by, at = r
    stored = json.loads(rj)
    notes: dict[str, list] = {}
    for item, text, who, when in conn.execute(
            "SELECT item_id, note, (SELECT username FROM users u WHERE u.id = n.by), at FROM legal_citation_notes n WHERE check_id=? "
            "ORDER BY at, rowid", (cid,)):
        notes.setdefault(item, []).append({"note": text, "by": who, "at": when})
    items = checkmod.present(mid, stored["items"], uid)
    for it in items:
        it["notes"] = notes.get(it["id"], [])
    version = conn.execute("SELECT number FROM legal_document_versions WHERE id=?", (vid,)).fetchone()
    return {"id": cid, "document_id": did, "version_id": vid, "version_number": version[0] if version else None, "items": items,
            "summary": checkmod.summary(items), "engine": stored.get("engine"), "external_lookup": False, "notice": checkmod.NOTICE,
            "created_by": (conn.execute("SELECT username FROM users WHERE id=?", (by,)).fetchone() or [None])[0], "created_at": at}


def _check(p: dict, matter_id: str, check_id: str) -> dict:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {_COLS} FROM legal_citation_checks WHERE id=? AND matter_id=?", (check_id, matter_id)).fetchone()
        if not r or not documents.visible(matter_id, r[2], p["sub"]):
            raise _not_found()
        return _out(conn, r, p["sub"])


@router.post("/matters/{matter_id}/documents/{document_id}/citation-check")
@audited("check.citations", object_type="document", object_param="document_id")
async def run_check(matter_id: str, document_id: str, request: Request, inp: Optional[CheckIn] = None):
    p = principal_of(request)
    await run_in_threadpool(require, p, "research.query", matter_id)
    if not await run_in_threadpool(documents.visible, matter_id, document_id, p["sub"]):
        raise _not_found()
    try:
        items, inputs = await run_in_threadpool(checkmod.run, matter_id, document_id, p["sub"], inp.version_id if inp else None)
    except LookupError:
        raise _not_found()
    cid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_citation_checks (id, matter_id, document_id, version_id, results_json, created_by, created_at) "
                     "VALUES (?,?,?,?,?,?,?)", (cid, matter_id, document_id, inputs["version_id"],
                                                json.dumps({"items": items, "engine": inputs["engine"]}, ensure_ascii=False),
                                                p["sub"], policy.now_iso()))
        conn.commit()
    out = await run_in_threadpool(_check, p, matter_id, cid)
    note(details={"check": cid, **out["summary"]})
    return {"check": out}


@router.get("/matters/{matter_id}/documents/{document_id}/citation-checks")
def list_checks(matter_id: str, document_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    if not documents.visible(matter_id, document_id, p["sub"]):
        raise _not_found()
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {_COLS} FROM legal_citation_checks WHERE matter_id=? AND document_id=? ORDER BY created_at DESC",
                            (matter_id, document_id)).fetchall()
        return {"checks": [_out(conn, r, p["sub"]) for r in rows]}


@router.get("/matters/{matter_id}/citation-checks/{check_id}")
def read_check(matter_id: str, check_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    return {"check": _check(p, matter_id, check_id)}


@router.post("/matters/{matter_id}/citation-checks/{check_id}/notes")
@audited("check.citations.note", object_type="citation_check", object_param="check_id")
def add_note(matter_id: str, check_id: str, inp: NoteIn, request: Request):
    p = principal_of(request)
    require(p, "artifacts.approve", matter_id)                 # lead, attorney or reviewer
    current = _check(p, matter_id, check_id)
    if not any(i["id"] == inp.item for i in current["items"]):
        raise HTTPException(status_code=422, detail=f"This check has no item {inp.item}.")
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_citation_notes (id, check_id, item_id, note, by, at) VALUES (?,?,?,?,?,?)",
                     (str(uuid.uuid4()), check_id, inp.item, inp.note.strip(), p["sub"], policy.now_iso()))
        conn.commit()
    note(details={"item": inp.item})
    return {"check": _check(p, matter_id, check_id)}
