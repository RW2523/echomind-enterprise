"""Playbooks and playbook checks (S3.2, DOC-02a; decision D-046).

Firm:
- `GET  /api/legal/playbooks`: approved positions for every lawyer; drafts too for those who manage playbooks.
- `POST /api/legal/playbooks`, `POST …/{pid}/positions` (a new version of a topic): `playbooks.manage`.
- `POST …/{pid}/positions/{posid}/approve`: a managing partner other than the author, for that hash.
- `POST …/{pid}/import`: a precedent read into proposed positions (nothing stored).

Matter:
- `POST /api/legal/matters/{id}/documents/{did}/playbook-check`: check a document (`ai.draft`).
- `GET  …/documents/{did}/playbook-checks`, `GET …/playbook-checks/{cid}`: for those who can see the document.
- `POST …/playbook-checks/{cid}/decisions`: a lawyer agrees or disagrees with a line (lead, attorney, reviewer).
"""
from __future__ import annotations

import json
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note
from ..matters.routes import principal_of, require, require_law_edition
from ..retrieval import scope as scopemod
from . import check as checkmod
from . import library

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


class PlaybookIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    jurisdiction: Optional[str] = Field(default=None, max_length=100)


class FallbackIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    note: Optional[str] = Field(default=None, max_length=500)


class PositionIn(BaseModel):
    topic: str = Field(min_length=1, max_length=40)
    title: Optional[str] = Field(default=None, max_length=200)
    standard: str = Field(min_length=1, max_length=4000)
    fallbacks: list[FallbackIn] = Field(default_factory=list, max_length=5)
    guidance: Optional[str] = Field(default=None, max_length=2000)


class ApproveIn(BaseModel):
    content_sha256: str = Field(min_length=64, max_length=64)


def _manager(p: dict) -> bool:
    return policy.authorize(p, "playbooks.manage").allowed or policy.authorize(p, "playbooks.approve").allowed


@router.get("/playbooks")
def list_playbooks(request: Request):
    p = principal_of(request)
    return {"playbooks": library.listing(p["org_id"], drafts=_manager(p)), "topics": {k: v[0] for k, v in library.TOPICS.items()},
            "can": {"manage": policy.authorize(p, "playbooks.manage").allowed, "approve": policy.authorize(p, "playbooks.approve").allowed}}


@router.post("/playbooks")
@audited("playbook.create", object_type="playbook")
def create_playbook(inp: PlaybookIn, request: Request):
    p = principal_of(request)
    require(p, "playbooks.manage")
    pb = library.create(p["org_id"], inp.name, inp.description, inp.jurisdiction, p["sub"])
    note(object_id=pb["id"])
    return {"playbook": pb}


@router.post("/playbooks/{playbook_id}/positions")
@audited("playbook.position", object_type="playbook", object_param="playbook_id")
def add_position(playbook_id: str, inp: PositionIn, request: Request):
    p = principal_of(request)
    require(p, "playbooks.manage")
    try:
        pb = library.add_position(p["org_id"], playbook_id, {"topic": inp.topic, "title": inp.title, "standard": inp.standard,
                                                              "fallbacks": [f.model_dump() for f in inp.fallbacks], "guidance": inp.guidance}, p["sub"])
    except library.Invalid as e:
        raise HTTPException(status_code=422, detail=str(e))
    except LookupError:
        raise _not_found()
    note(details={"topic": inp.topic, "fallbacks": len(inp.fallbacks)})
    return {"playbook": pb}


@router.post("/playbooks/{playbook_id}/positions/{position_id}/approve")
@audited("playbook.approve", object_type="playbook", object_param="playbook_id")
def approve_position(playbook_id: str, position_id: str, inp: ApproveIn, request: Request):
    p = principal_of(request)
    require(p, "playbooks.approve")
    try:
        pb = library.approve(p["org_id"], playbook_id, position_id, p["sub"], inp.content_sha256)
    except library.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except LookupError:
        raise _not_found()
    note(details={"position": position_id})
    return {"playbook": pb}


@router.post("/playbooks/{playbook_id}/import")
@audited("playbook.import", object_type="playbook", object_param="playbook_id")
async def import_precedent(playbook_id: str, request: Request, file: UploadFile = File(...)):
    """A precedent read into proposed positions; nothing is stored until a lawyer adds a position."""
    p = principal_of(request)
    await run_in_threadpool(require, p, "playbooks.manage")
    if not await run_in_threadpool(library.get, p["org_id"], playbook_id):
        raise _not_found()
    data = await file.read(20_000_001)
    if len(data) > 20_000_000:
        raise HTTPException(status_code=413, detail="The file is larger than 20 MB.")
    try:
        found = await run_in_threadpool(library.proposals, file.filename or "precedent.txt", data)
    except Exception:                                   # noqa: BLE001 — a file the reader cannot read
        raise HTTPException(status_code=422, detail="This file could not be read. Use Word (.docx), PDF or text.")
    note(details={"proposals": len(found)})
    return {"proposals": found}


# ── Checks ───────────────────────────────────────────────────────────────────────
class CheckIn(BaseModel):
    playbook_id: str = Field(min_length=1, max_length=64)
    version_id: Optional[str] = Field(default=None, max_length=64)


class DecisionIn(BaseModel):
    topic: str = Field(min_length=1, max_length=40)
    decision: Literal["agree", "disagree"]
    reason: Optional[str] = Field(default=None, max_length=500)


def _visible(matter_id: str, document_id: str, uid: str) -> bool:
    from ..documents import repo as documents
    return documents.visible(matter_id, document_id, uid)


def _check_out(conn, r) -> dict:
    cid, mid, did, vid, pbid, rj, ij, by, at = r
    results = json.loads(rj)
    decisions = {}
    for topic, decision, reason, who, when in conn.execute(
            "SELECT topic, decision, reason, (SELECT username FROM users u WHERE u.id = d.by), at FROM legal_playbook_decisions d "
            "WHERE check_id=? ORDER BY at, rowid", (cid,)):
        decisions[topic] = {"decision": decision, "reason": reason, "by": who, "at": when}
    for row in results:
        row["decision"] = decisions.get(row["topic"])
    pb = conn.execute("SELECT name FROM legal_playbooks WHERE id=?", (pbid,)).fetchone()
    return {"id": cid, "document_id": did, "version_id": vid, "playbook": {"id": pbid, "name": pb[0] if pb else None}, "results": results,
            "summary": checkmod.summary(results), "inputs": json.loads(ij), "created_by": (conn.execute(
                "SELECT username FROM users WHERE id=?", (by,)).fetchone() or [None])[0], "created_at": at, "advisory": True}


_CCOLS = "id, matter_id, document_id, version_id, playbook_id, results_json, inputs_json, created_by, created_at"


def _check(p: dict, matter_id: str, check_id: str) -> dict:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {_CCOLS} FROM legal_playbook_checks WHERE id=? AND matter_id=?", (check_id, matter_id)).fetchone()
        if not r or not _visible(matter_id, r[2], p["sub"]):
            raise _not_found()
        return _check_out(conn, r)


@router.post("/matters/{matter_id}/documents/{document_id}/playbook-check")
@audited("playbook.check", object_type="document", object_param="document_id")
async def run_check(matter_id: str, document_id: str, inp: CheckIn, request: Request):
    p = principal_of(request)
    await run_in_threadpool(require, p, "ai.draft", matter_id)
    if not await run_in_threadpool(_visible, matter_id, document_id, p["sub"]):
        raise _not_found()
    positions = await run_in_threadpool(library.approved_positions, p["org_id"], inp.playbook_id)
    if not positions:
        raise HTTPException(status_code=409, detail="This playbook has no approved positions yet; a managing partner approves them first.")
    scope = scopemod.authorize(p, matter_id, "ai.draft")
    try:
        results, inputs = await checkmod.run(scope, document_id, positions, inp.version_id)
    except LookupError:
        raise _not_found()
    cid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_playbook_checks (id, matter_id, document_id, version_id, playbook_id, results_json, inputs_json, "
                     "created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                     (cid, matter_id, document_id, inputs["version_id"], inp.playbook_id, json.dumps(results, ensure_ascii=False),
                      json.dumps(inputs), p["sub"], policy.now_iso()))
        conn.commit()
    out = await run_in_threadpool(_check, p, matter_id, cid)
    note(details={"check": cid, **out["summary"]})
    return {"check": out}


@router.get("/matters/{matter_id}/documents/{document_id}/playbook-checks")
def list_checks(matter_id: str, document_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    if not _visible(matter_id, document_id, p["sub"]):
        raise _not_found()
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {_CCOLS} FROM legal_playbook_checks WHERE matter_id=? AND document_id=? ORDER BY created_at DESC",
                            (matter_id, document_id)).fetchall()
        return {"checks": [_check_out(conn, r) for r in rows]}


@router.get("/matters/{matter_id}/playbook-checks/{check_id}")
def read_check(matter_id: str, check_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    return {"check": _check(p, matter_id, check_id)}


@router.post("/matters/{matter_id}/playbook-checks/{check_id}/decisions")
@audited("playbook.decision", object_type="playbook_check", object_param="check_id")
def decide(matter_id: str, check_id: str, inp: DecisionIn, request: Request):
    p = principal_of(request)
    require(p, "artifacts.approve", matter_id)                 # lead, attorney or reviewer
    current = _check(p, matter_id, check_id)
    if not any(r["topic"] == inp.topic for r in current["results"]):
        raise HTTPException(status_code=422, detail=f"This check has no line for {inp.topic}.")
    if inp.decision == "disagree" and len((inp.reason or "").strip()) < 3:
        raise HTTPException(status_code=422, detail="Say why you disagree.")
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_playbook_decisions (id, check_id, topic, decision, reason, by, at) VALUES (?,?,?,?,?,?,?)",
                     (str(uuid.uuid4()), check_id, inp.topic, inp.decision, (inp.reason or "").strip() or None, p["sub"], policy.now_iso()))
        conn.commit()
    note(details={"topic": inp.topic, "decision": inp.decision})
    return {"check": _check(p, matter_id, check_id)}
