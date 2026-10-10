"""Reviews (S2.7, decision D-043): any AI draft read, reviewed and filed by kind-independent routes,
and the queue of everything waiting for a person's decision.

- `GET  /api/legal/reviews`: what waits for you, and what of yours waits for others.
- `GET  /api/legal/matters/{id}/artifacts/{aid}`: one version, its decisions and the subject's versions.
- `POST /api/legal/matters/{id}/artifacts/{aid}/review`: approve the version you read (its content
  hash), or ask for changes with a reason. Not by its author.
- `POST /api/legal/matters/{id}/artifacts/{aid}/file`: an approved version becomes a matter document.

A version is shown only to members who can see every document it quotes, and minutes only while
their transcript exists; otherwise the answer is the same 404 as for anything that does not exist.
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import artifacts, policy
from ..audit.record import audited, note
from ..matters.routes import principal_of, require, require_law_edition
from . import queue

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])
_BASE = "/matters/{matter_id}/artifacts/{artifact_id}"


class ReviewIn(BaseModel):
    decision: Literal["approved", "changes_requested"]
    content_sha256: str = Field(min_length=64, max_length=64)
    reason: Optional[str] = Field(default=None, max_length=500)


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


def _live(a: dict) -> bool:
    if a["kind"] != "minutes":
        return True
    from ...core.db import get_conn
    with get_conn() as conn:
        return bool(conn.execute("SELECT 1 FROM legal_meetings WHERE id=? AND matter_id=? AND transcript_deleted_at IS NULL",
                                 (a["subject_id"], a["matter_id"])).fetchone())


def _artifact(p: dict, matter_id: str, artifact_id: str) -> dict:
    a = artifacts.get(matter_id, artifact_id)
    if not a or not _live(a) or not artifacts.visible_to(a, p["sub"]):
        raise _not_found()
    return a


def _as_text(a: dict, approval: dict) -> tuple[bytes, str]:
    if a["kind"] == "minutes":
        from ..meetings import minutes
        return minutes.as_text(a["content"], approval).encode("utf-8"), f"minutes-{a['subject_id'][:8]}-v{a['version']}.md"
    from ..research import notes
    return notes.as_text(a["content"], approval).encode("utf-8"), f"research-note-{a['subject_id'][:8]}-v{a['version']}.md"


@router.get("/reviews")
def reviews(request: Request):
    """Polled by the navigation for its count, so not written to the audit trail on every read (like the
    research history); each decision taken from it is."""
    return queue.for_principal(principal_of(request))


@router.get(_BASE)
@audited("artifact.read", object_type="ai_artifact", object_param="artifact_id")
def read_artifact(matter_id: str, artifact_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    a = _artifact(p, matter_id, artifact_id)
    others = [{"id": v["id"], "version": v["version"], "status": v["status"], "created_by_name": v["created_by_name"],
               "created_at": v["created_at"], "origin": v["origin"]}
              for v in artifacts.versions(matter_id, a["subject_type"], a["subject_id"], a["kind"])]
    can = {"approve": policy.authorize(p, "artifacts.approve", matter_id).allowed and a["created_by"] != p["sub"],
           "file": policy.authorize(p, "documents.import", matter_id).allowed, "author": a["created_by"] == p["sub"]}
    note(details={"kind": a["kind"], "version": a["version"]})
    return {"artifact": a, "versions": others, "can": can}


@router.post(_BASE + "/review")
@audited("artifact.review", object_type="ai_artifact", object_param="artifact_id")
def review_artifact(matter_id: str, artifact_id: str, inp: ReviewIn, request: Request):
    """Approve the version you read (by its content hash), or ask for changes with a reason."""
    p = principal_of(request)
    require(p, "artifacts.approve", matter_id)
    _artifact(p, matter_id, artifact_id)
    if inp.decision == "changes_requested" and not (inp.reason or "").strip():
        raise HTTPException(status_code=422, detail="Say what should change.")
    try:
        out = artifacts.decide(matter_id, artifact_id, inp.decision, p["sub"], inp.content_sha256, (inp.reason or "").strip() or None)
    except artifacts.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    note(details={"decision": inp.decision, "kind": out["kind"], "version": out["version"]})
    return {"artifact": out}


@router.post(_BASE + "/file")
@audited("artifact.file", object_type="ai_artifact", object_param="artifact_id")
def file_artifact(matter_id: str, artifact_id: str, request: Request):
    """An approved version becomes a matter document carrying its approval and the hash it bound."""
    from ..documents import repo as docs
    p = principal_of(request)
    require(p, "documents.import", matter_id)
    a = _artifact(p, matter_id, artifact_id)
    approval = artifacts.approval_of(a)
    if not approval:
        raise HTTPException(status_code=409, detail="Only an approved version can be filed.")
    if a.get("document_id") and docs.get_document(matter_id, a["document_id"]):
        return {"document_id": a["document_id"], "changed": False}
    if a["kind"] == "legal_draft":                       # S3.1: a Word document in the firm's styles
        from ..drafting.routes import filed
        body, filename, mime, kind = filed(p, a, approval)
        doc_id = artifacts.file(p, a, body, filename, mime, kind)
    else:
        body, filename = _as_text(a, approval)
        doc_id = artifacts.file(p, a, body, filename)
    note(details={"kind": a["kind"], "version": a["version"], "content_sha256": approval["content_sha256"]})
    return {"document_id": doc_id, "changed": True}
