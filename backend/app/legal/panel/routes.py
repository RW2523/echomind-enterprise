"""The four-perspective review panel (S4.5, OPT-05; decision D-057). Off unless LEGAL_PANEL is on (404 otherwise).

- `POST /api/legal/matters/{id}/panel` (`ai.draft`) {question, specialist}: ask the panel (a job).
- `GET  /api/legal/matters/{id}/panel`: the matter's panel reviews, for its members.
- `GET  /api/legal/matters/{id}/panel/{rid}`: one review, its newest version and the earlier ones.
- `POST /api/legal/matters/{id}/panel/{rid}/facts` (`ai.draft`) {facts}: the lawyer's answers to the missing facts; the
  panel runs again with them, marked as the lawyer's.
Approval is the review route's (`/artifacts/{aid}/review`), open to a partner only: the matter's lead or a managing partner.
"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from ...core.config import settings
from ...core.db import get_conn
from .. import artifacts, policy
from ..audit.record import audited, note
from ..jobs import worker
from ..matters.routes import principal_of, require, require_law_edition
from .run import ROLES, SPECIALISMS, is_partner


def panel_on() -> None:
    if not settings.LEGAL_PANEL:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition), Depends(panel_on)])
_COLS = ("id", "matter_id", "question", "specialist", "facts_json", "status", "artifact_id", "created_by", "created_at", "finished_at")


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    specialist: str = Field(min_length=1, max_length=40)

    @field_validator("question")
    @classmethod
    def _asked(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 10:
            raise ValueError("Ask a full question.")
        return v

    @field_validator("specialist")
    @classmethod
    def _area(cls, v: str) -> str:
        if v not in SPECIALISMS:
            raise ValueError(f"Choose one of: {', '.join(SPECIALISMS)}.")
        return v


class Fact(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    answer: str = Field(min_length=1, max_length=1000)


class FactsIn(BaseModel):
    facts: list[Fact] = Field(min_length=1, max_length=8)


def _enqueue(conn, org_id: str, matter_id: str, rid: str, by: str) -> None:
    conn.execute("INSERT INTO legal_jobs (id, kind, org_id, matter_id, object_id, requested_by, created_at) VALUES (?,?,?,?,?,?,?)",
                 (str(uuid.uuid4()), "panel.run", org_id, matter_id, rid, by, policy.now_iso()))


def _username(conn, uid):
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return r[0] if r else None


def _review(conn, r, p: dict) -> dict:
    d = dict(zip(_COLS, r))
    return {"id": d["id"], "question": d["question"], "specialist": d["specialist"], "area": SPECIALISMS.get(d["specialist"]),
            "status": d["status"], "facts": json.loads(d["facts_json"]), "created_by": _username(conn, d["created_by"]),
            "created_at": d["created_at"], "finished_at": d["finished_at"], "mine": d["created_by"] == p["sub"]}


def _visible(a: dict, uid: str) -> bool:
    return artifacts.visible_to(a, uid)


@router.post("/matters/{matter_id}/panel")
@audited("panel.ask", object_type="matter", object_param="matter_id")
def ask(matter_id: str, inp: AskIn, request: Request):
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    rid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_panel_reviews (id, org_id, matter_id, question, specialist, facts_json, status, created_by, created_at) "
                     "VALUES (?,?,?,?,?,?,?,?,?)", (rid, p["org_id"], matter_id, inp.question, inp.specialist, "[]", "queued", p["sub"], policy.now_iso()))
        _enqueue(conn, p["org_id"], matter_id, rid, p["sub"])
        conn.commit()
    worker.wake()
    note(matter_id=matter_id, details={"kind": inp.specialist})
    return {"review": {"id": rid, "status": "queued"}}


@router.get("/matters/{matter_id}/panel")
def listing(matter_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    with get_conn() as conn:
        rows = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_panel_reviews WHERE matter_id=? ORDER BY created_at DESC", (matter_id,)).fetchall()
        out = []
        for r in rows:
            a = artifacts.latest(matter_id, "panel_review", r[0], "panel")
            if a and not _visible(a, p["sub"]):
                continue
            out.append({**_review(conn, r, p), "version": a["version"] if a else None, "approval": a["status"] if a else None})
    return {"reviews": out, "specialisms": SPECIALISMS, "roles": [{"role": r, "label": label} for r, label, _ in ROLES],
            "can": {"ask": policy.authorize(p, "ai.draft", matter_id).allowed, "approve": is_partner(p, matter_id)}}


@router.get("/matters/{matter_id}/panel/{review_id}")
@audited("panel.read", object_type="panel_review", object_param="review_id")
def read(matter_id: str, review_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_panel_reviews WHERE id=? AND matter_id=?", (review_id, matter_id)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Not Found")
        review = _review(conn, r, p)
    versions = artifacts.versions(matter_id, "panel_review", review_id, "panel")
    if versions and not _visible(versions[0], p["sub"]):
        raise HTTPException(status_code=404, detail="Not Found")
    a = versions[0] if versions else None
    return {"review": review, "artifact": a,
            "versions": [{"id": v["id"], "version": v["version"], "status": v["status"], "created_at": v["created_at"]} for v in versions],
            "can": {"approve": bool(a) and is_partner(p, matter_id) and a["created_by"] != p["sub"],
                    "facts": policy.authorize(p, "ai.draft", matter_id).allowed, "author": bool(a) and a["created_by"] == p["sub"]}}


@router.post("/matters/{matter_id}/panel/{review_id}/facts")
@audited("panel.facts", object_type="panel_review", object_param="review_id")
def facts(matter_id: str, review_id: str, inp: FactsIn, request: Request):
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    with get_conn() as conn:
        r = conn.execute("SELECT status, facts_json FROM legal_panel_reviews WHERE id=? AND matter_id=?", (review_id, matter_id)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Not Found")
        if r[0] in ("queued", "running"):
            raise HTTPException(status_code=409, detail="The panel is still working on this question.")
        given = {f["question"]: f for f in json.loads(r[1])}
        for f in inp.facts:
            given[" ".join(f.question.split())] = {"question": " ".join(f.question.split()), "answer": " ".join(f.answer.split())}
        conn.execute("UPDATE legal_panel_reviews SET facts_json=?, status='queued', finished_at=NULL WHERE id=?",
                     (json.dumps(list(given.values()), ensure_ascii=False), review_id))
        _enqueue(conn, p["org_id"], matter_id, review_id, p["sub"])
        conn.commit()
    worker.wake()
    note(matter_id=matter_id, details={"facts": len(inp.facts)})
    return {"review": {"id": review_id, "status": "queued"}}
