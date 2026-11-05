"""The four-perspective review panel (S4.5, OPT-05; decision D-057). Off unless LEGAL_PANEL is on (404 otherwise).

- `POST /api/legal/matters/{id}/panel` (`ai.draft`) {question, specialist, depth}: ask the panel (a job); the perspectives are
  chosen by rule from the question when it is asked (run.assemble) and kept with it.
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
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ...core.config import settings
from ...core.db import get_conn
from .. import artifacts, policy
from ..audit.record import audited, note
from ..documents import repo as documents
from ..jobs import worker
from ..matters.routes import principal_of, require, require_law_edition
from .run import LIBRARY, SPECIALISMS, assemble, is_partner, label_of


def panel_on() -> None:
    if not settings.LEGAL_PANEL:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition), Depends(panel_on)])
_COLS = ("id", "matter_id", "question", "specialist", "facts_json", "status", "artifact_id", "created_by", "created_at", "finished_at",
         "documents_json", "roles_json", "depth")


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    specialist: str = Field(min_length=1, max_length=40)
    documents: list[str] = Field(default_factory=list, max_length=20)     # consult only these (all the matter's when empty)
    depth: Literal["focused", "deep"] = "focused"                         # deep: the first perspective also answers the objections

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


class ConveneIn(BaseModel):
    specialist: str = Field(default="commercial", min_length=1, max_length=40)
    depth: Literal["focused", "deep"] = "focused"

    @field_validator("specialist")
    @classmethod
    def _area(cls, v: str) -> str:
        if v not in SPECIALISMS:
            raise ValueError(f"Choose one of: {', '.join(SPECIALISMS)}.")
        return v


class DismissIn(BaseModel):
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def _said(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("Say why this question is not worth asking.")
        return v


class Fact(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    answer: str = Field(min_length=1, max_length=1000)


class FactsIn(BaseModel):
    facts: list[Fact] = Field(min_length=1, max_length=8)


class ProceedIn(BaseModel):
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def _said(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("Say why the panel should continue without these facts.")
        return v


def _enqueue(conn, org_id: str, matter_id: str, rid: str, by: str) -> None:
    conn.execute("INSERT INTO legal_jobs (id, kind, org_id, matter_id, object_id, requested_by, created_at) VALUES (?,?,?,?,?,?,?)",
                 (str(uuid.uuid4()), "panel.run", org_id, matter_id, rid, by, policy.now_iso()))


def _username(conn, uid):
    r = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    return r[0] if r else None


def _review(conn, r, p: dict) -> dict:
    d = dict(zip(_COLS, r))
    docs = []
    for did in json.loads(d["documents_json"] or "[]"):
        doc = documents.get_document(d["matter_id"], did, p["sub"])
        docs.append({"id": did, "title": doc["title"] if doc else None})
    area = SPECIALISMS.get(d["specialist"], d["specialist"])
    roles = [r for r in json.loads(d["roles_json"] or "[]") if r in LIBRARY]
    return {"id": d["id"], "question": d["question"], "specialist": d["specialist"], "area": SPECIALISMS.get(d["specialist"]), "documents": docs,
            "roles": [{"role": r, "label": label_of(r, area)} for r in roles], "depth": d["depth"], "status": d["status"], "facts": json.loads(d["facts_json"]), "created_by": _username(conn, d["created_by"]),
            "created_at": d["created_at"], "finished_at": d["finished_at"], "mine": d["created_by"] == p["sub"]}


def _visible(a: dict, uid: str) -> bool:
    return artifacts.visible_to(a, uid)


@router.post("/matters/{matter_id}/panel")
@audited("panel.ask", object_type="matter", object_param="matter_id")
def ask(matter_id: str, inp: AskIn, request: Request):
    return start(principal_of(request), matter_id, inp)


def start(p: dict, matter_id: str, inp: AskIn) -> dict:
    """Queue a panel review with every check of the route (the chat calls this too; the panel must be on)."""
    panel_on()
    require(p, "ai.draft", matter_id)
    chosen = list(dict.fromkeys(inp.documents))
    if any(not documents.visible(matter_id, did, p["sub"]) for did in chosen):
        raise HTTPException(status_code=422, detail="Choose documents of this matter that you can see.")
    rid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_panel_reviews (id, org_id, matter_id, question, specialist, facts_json, documents_json, status, created_by, "
                     "created_at, roles_json, depth) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (rid, p["org_id"], matter_id, inp.question, inp.specialist, "[]", json.dumps(chosen), "queued", p["sub"], policy.now_iso(),
                      json.dumps(assemble(inp.question)), inp.depth))
        _enqueue(conn, p["org_id"], matter_id, rid, p["sub"])
        conn.commit()
    worker.wake()
    note(matter_id=matter_id, details={"kind": inp.specialist, "documents": len(chosen), "depth": inp.depth})
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
    from ..council import discovery
    questions, more = discovery.listed(p, matter_id) if discovery.enabled() else ([], 0)
    number = _number(matter_id)
    return {"reviews": out, "specialisms": SPECIALISMS, "roles": [{"role": r, "label": label} for r, (label, _) in LIBRARY.items()],
            "discovery": discovery.enabled(), "questions": [discovery.shown(q, number) for q in questions], "more_questions": more,
            "can": {"ask": policy.authorize(p, "ai.draft", matter_id).allowed, "approve": is_partner(p, matter_id)}}


def _number(matter_id: str) -> str:
    with get_conn() as conn:
        r = conn.execute("SELECT number FROM matters WHERE id=?", (matter_id,)).fetchone()
    return r[0] if r else "?"


def _discovery_on() -> None:
    from ..council import discovery
    if not discovery.enabled():
        raise HTTPException(status_code=404, detail="Not Found")


@router.post("/matters/{matter_id}/council/questions/{question_id}/convene")
@audited("council.convene", object_type="council_question", object_param="question_id")
def convene(matter_id: str, question_id: str, inp: ConveneIn, request: Request):
    """Put a suggested question to the Council: a review asked with the question's words, linked to it."""
    from ..council import discovery
    _discovery_on()
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    q = discovery.get(p, matter_id, question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Not Found")
    if q["status"] != "suggested":
        raise HTTPException(status_code=409, detail="This question was already convened, dismissed or replaced.")
    out = start(p, matter_id, AskIn(question=q["title"], specialist=inp.specialist, depth=inp.depth))
    rid = out["review"]["id"]
    with get_conn() as conn:
        conn.execute("UPDATE legal_panel_reviews SET question_id=? WHERE id=?", (question_id, rid))
        conn.commit()
    if not discovery.decide(question_id, "convened", p["sub"], review_id=rid):
        raise HTTPException(status_code=409, detail="This question was already convened, dismissed or replaced.")
    note(matter_id=matter_id, details={"detector": q["detector"], "depth": inp.depth})
    return {"review": out["review"], "question": {"id": question_id, "status": "convened"}}


@router.post("/matters/{matter_id}/council/questions/{question_id}/dismiss")
@audited("council.dismiss", object_type="council_question", object_param="question_id")
def dismiss(matter_id: str, question_id: str, inp: DismissIn, request: Request):
    """Not worth asking: kept with the person's reason, and not suggested again until its sources change."""
    from ..council import discovery
    _discovery_on()
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    q = discovery.get(p, matter_id, question_id)
    if not q:
        raise HTTPException(status_code=404, detail="Not Found")
    if not discovery.decide(question_id, "dismissed", p["sub"], reason=inp.reason):
        raise HTTPException(status_code=409, detail="This question was already convened, dismissed or replaced.")
    note(matter_id=matter_id, details={"detector": q["detector"]})
    return {"question": {"id": question_id, "status": "dismissed"}}


@router.post("/matters/{matter_id}/council/discover")
@audited("council.discover", object_type="matter", object_param="matter_id")
def discover(matter_id: str, request: Request):
    """Look for questions now (rules only; it also runs after each document is read, and daily)."""
    from ..council import discovery
    _discovery_on()
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    with get_conn() as conn:
        queued = discovery.enqueue(conn, p["org_id"], matter_id, p["sub"])
        conn.commit()
    worker.wake()
    return {"queued": queued}


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


@router.post("/matters/{matter_id}/panel/{review_id}/proceed")
@audited("panel.proceed", object_type="panel_review", object_param="review_id")
def proceed(matter_id: str, review_id: str, inp: ProceedIn, request: Request):
    """A stopped panel continues without the facts it asked for, on the record: a new version with the same answers,
    noting what was left open, by whom, when and why. A partner can then decide on it."""
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    a = artifacts.latest(matter_id, "panel_review", review_id, "panel")
    if not a or not artifacts.visible_to(a, p["sub"]):
        raise HTTPException(status_code=404, detail="Not Found")
    if not a["content"].get("stopped"):
        raise HTTPException(status_code=409, detail="This panel has not stopped for missing facts.")
    content = {**a["content"], "stopped": False,
               "proceeded_without": {"facts": a["content"]["missing_facts"], "by": p["username"], "reason": inp.reason, "at": policy.now_iso()}}
    artifacts.create(matter_id, "panel", "panel_review", review_id, content, {**a["inputs"], "proceeded_from": a["version"]}, p["sub"], "person")
    with get_conn() as conn:
        conn.execute("UPDATE legal_panel_reviews SET status='done' WHERE id=?", (review_id,))
        conn.commit()
    note(matter_id=matter_id, details={"facts": len(a["content"]["missing_facts"])})
    return {"review": {"id": review_id, "status": "done"}}

