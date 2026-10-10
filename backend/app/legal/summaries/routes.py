"""Summaries (S4.2, OPT-02; decision D-055). Off unless LEGAL_SUMMARIES is on (404 otherwise).

- `POST /api/legal/matters/{id}/documents/{did}/summaries` (`ai.draft`): summarise the document's current version (a job).
- `GET  …/documents/{did}/summaries`: its summary versions, newest first, and whether a job is waiting.
- `GET  /api/legal/matters/{id}/summaries`: the newest summary of each document the person can see.
- `POST /api/legal/matters/{id}/summaries/{aid}/edit` (`ai.draft`): a reviewer's edit of the latest version as a new
  version: points reworded (kept with their quote and citation, marked edited), reordered or removed.
Reading, approval and filing use the review routes (`/artifacts/{aid}`, `…/review`, `…/file`).
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.config import settings
from ...core.db import get_conn
from .. import artifacts, policy
from ..audit.record import audited, note
from ..documents import repo as documents
from ..jobs import worker
from ..matters.routes import principal_of, require, require_law_edition


def summaries_on() -> None:
    if not settings.LEGAL_SUMMARIES:
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition), Depends(summaries_on)])


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


def _job(matter_id: str, document_id: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute("SELECT status, error, created_at FROM legal_jobs WHERE kind='summary.run' AND matter_id=? AND object_id=? "
                         "ORDER BY created_at DESC LIMIT 1", (matter_id, document_id)).fetchone()
    return {"status": r[0], "error": r[1], "created_at": r[2]} if r else None


@router.post("/matters/{matter_id}/documents/{document_id}/summaries")
@audited("summary.create", object_type="document", object_param="document_id")
def create(matter_id: str, document_id: str, request: Request):
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    d = documents.get_document(matter_id, document_id, p["sub"])
    if not d:
        raise _not_found()
    if not d.get("current_version") or d["current_version"].get("index_status") != "indexed":
        raise HTTPException(status_code=409, detail="This document is still being read. Try again in a moment.")
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_jobs (id, kind, org_id, matter_id, object_id, requested_by, created_at) VALUES (?,?,?,?,?,?,?)",
                     (str(uuid.uuid4()), "summary.run", p["org_id"], matter_id, document_id, p["sub"], policy.now_iso()))
        conn.commit()
    worker.wake()
    note(matter_id=matter_id)
    return {"job": {"status": "queued"}}


@router.get("/matters/{matter_id}/documents/{document_id}/summaries")
@audited("summary.read", object_type="document", object_param="document_id")
def for_document(matter_id: str, document_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    if not documents.visible(matter_id, document_id, p["sub"]):
        raise _not_found()
    versions = artifacts.versions(matter_id, "document", document_id, "summary")
    return {"summaries": versions, "job": _job(matter_id, document_id),
            "can": {"summarise": policy.authorize(p, "ai.draft", matter_id).allowed}}


@router.get("/matters/{matter_id}/summaries")
@audited("summary.list", object_type="matter", object_param="matter_id")
def for_matter(matter_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    with get_conn() as conn:
        ids = [r[0] for r in conn.execute("SELECT DISTINCT subject_id FROM legal_ai_artifacts WHERE matter_id=? AND kind='summary'", (matter_id,))]
    out = []
    for did in ids:
        newest = artifacts.latest(matter_id, "document", did, "summary")
        if newest and artifacts.visible_to(newest, p["sub"]):
            out.append(newest)
    out.sort(key=lambda a: a["created_at"], reverse=True)
    note(details={"summaries": len(out)})
    return {"summaries": out, "can": {"summarise": policy.authorize(p, "ai.draft", matter_id).allowed}}


class PointEdit(BaseModel):
    id: str = Field(min_length=1, max_length=20)
    text: Optional[str] = Field(default=None, min_length=1, max_length=600)


class EditIn(BaseModel):
    points: list[PointEdit] = Field(max_length=400)


@router.post("/matters/{matter_id}/summaries/{artifact_id}/edit")
@audited("summary.edit", object_type="ai_artifact", object_param="artifact_id")
def edit(matter_id: str, artifact_id: str, inp: EditIn, request: Request):
    p = principal_of(request)
    require(p, "ai.draft", matter_id)
    a = artifacts.get(matter_id, artifact_id)
    if not a or a["kind"] != "summary" or not artifacts.visible_to(a, p["sub"]):
        raise _not_found()
    newest = artifacts.latest(matter_id, "document", a["subject_id"], "summary")
    if newest["id"] != a["id"]:
        raise HTTPException(status_code=409, detail="A newer version of this summary exists. Edit that one.")
    if not inp.points:
        raise HTTPException(status_code=422, detail="Keep at least one point.")
    by_id = {pt["id"]: pt for pt in a["content"]["points"]}
    if any(e.id not in by_id for e in inp.points) or len({e.id for e in inp.points}) != len(inp.points):
        raise HTTPException(status_code=409, detail="The summary changed since you opened it. Reload it and edit again.")
    points, changed = [], 0
    for e in inp.points:
        pt = dict(by_id[e.id])
        text = " ".join((e.text or "").split())
        if text and text != pt["text"]:
            pt.update(text=text, origin="person")
            changed += 1
        points.append(pt)
    content = {**a["content"], "points": points, "unverified": sum(1 for x in points if x["verification"] == "failed"),
               "figures_unsupported": sum(1 for x in points if x.get("figures_unsupported"))}
    new = artifacts.create(matter_id, "summary", "document", a["subject_id"], content,
                           {**a["inputs"], "edited_from": a["version"]}, p["sub"], "person")
    note(details={"edited": changed, "removed": len(by_id) - len(points), "version": new["version"]})
    return {"summary": new}
