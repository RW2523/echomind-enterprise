"""/api/legal: privilege labels, legal holds, deletion requests and retention (S1.9).

Matter members request deletions and set labels inside their matters. Records administrators and
managing partners work firm-wide: holds, deletion approvals and retention, with the firm's matter
directory (numbers, titles, clients) to find matters (D-027) — never their content.
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note
from ..matters.routes import principal_of, require, require_law_edition
from ..walls import repo as walls_repo
from . import deletion, holds, labels, retention

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])

Label = Literal["privileged", "work_product", "confidential", "not_privileged", "unreviewed"]
ObjectType = Literal["document", "meeting"]
_HELD = "This material is under legal hold and cannot be deleted. A records administrator can tell you more."
_SECOND = "A different person must approve this."


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


class ReasonIn(BaseModel):
    reason: str = Field(min_length=1, max_length=1000)


class NoteIn(BaseModel):
    note: Optional[str] = Field(default=None, max_length=1000)


class HoldIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2000)
    matter_ids: list[str] = Field(min_length=1, max_length=500)


class LabelIn(BaseModel):
    label: Label


class PolicyIn(BaseModel):
    object_type: Literal["document", "meeting_transcript"]
    days_after_close: int = Field(ge=0, le=36500)
    note: Optional[str] = Field(default=None, max_length=1000)


class RunIn(BaseModel):
    mode: Literal["dry_run", "apply"] = "dry_run"


# ── Labels ───────────────────────────────────────────────────────────────────────
def _label_target(matter_id: str, object_type: str, object_id: str) -> None:
    table, extra = ("legal_documents", " AND deleted_at IS NULL") if object_type == "document" else ("legal_meetings", "")
    with get_conn() as conn:
        if not conn.execute(f"SELECT 1 FROM {table} WHERE id=? AND matter_id=?{extra}", (object_id, matter_id)).fetchone():
            raise _not_found()


@router.get("/matters/{matter_id}/labels/{object_type}/{object_id}")
def get_labels(matter_id: str, object_type: ObjectType, object_id: str, request: Request):
    require(principal_of(request), "matter.read", matter_id)
    _label_target(matter_id, object_type, object_id)
    return labels.state(object_type, object_id)


@router.post("/matters/{matter_id}/labels/{object_type}/{object_id}")
@audited("label.set", object_type="label")
def set_label(matter_id: str, object_type: ObjectType, object_id: str, inp: LabelIn, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    _label_target(matter_id, object_type, object_id)
    if policy.authorize(p, "labels.set", matter_id).allowed:
        kind = "set"
    else:
        require(p, "labels.suggest", matter_id)
        kind = "suggested"
    note(object_id=object_id, details={"label": inp.label, "kind": kind, "object_type": object_type})
    labels.add(p["org_id"], matter_id, object_type, object_id, inp.label, kind, "person", p["sub"])
    return labels.state(object_type, object_id)


# ── Deletion requests inside a matter ────────────────────────────────────────────
def _request(p: dict, matter_id: str, object_type: str, object_id: str, reason: str) -> dict:
    require(p, "deletion.request", matter_id)
    try:
        d = deletion.request(p["org_id"], matter_id, object_type, object_id, reason.strip(), p["sub"])
    except holds.Held:
        note(reason="legal_hold")
        raise HTTPException(status_code=409, detail=_HELD)
    except deletion.NotFound:
        raise _not_found()
    except deletion.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    note(object_id=d["id"], details={"object_type": object_type})
    return {"deletion": d}


@router.get("/matters/{matter_id}/documents/{document_id}/deletion-preview")
def preview_document_deletion(matter_id: str, document_id: str, request: Request):
    require(principal_of(request), "deletion.request", matter_id)
    try:
        return {"report": deletion.preview(matter_id, "document", document_id)}
    except deletion.NotFound:
        raise _not_found()


@router.post("/matters/{matter_id}/documents/{document_id}/delete")
@audited("deletion.request", object_type="deletion")
def request_document_deletion(matter_id: str, document_id: str, inp: ReasonIn, request: Request):
    return _request(principal_of(request), matter_id, "document", document_id, inp.reason)


@router.get("/matters/{matter_id}/meetings/{meeting_id}/transcript/deletion-preview")
def preview_transcript_deletion(matter_id: str, meeting_id: str, request: Request):
    require(principal_of(request), "deletion.request", matter_id)
    try:
        return {"report": deletion.preview(matter_id, "meeting_transcript", meeting_id)}
    except deletion.NotFound:
        raise _not_found()


@router.post("/matters/{matter_id}/meetings/{meeting_id}/transcript/delete")
@audited("deletion.request", object_type="deletion")
def request_transcript_deletion(matter_id: str, meeting_id: str, inp: ReasonIn, request: Request):
    return _request(principal_of(request), matter_id, "meeting_transcript", meeting_id, inp.reason)


@router.get("/matters/{matter_id}/deletions")
def matter_deletions(matter_id: str, request: Request):
    """Deletions waiting for approval in this matter, for the people who may request (and restore) them."""
    p = principal_of(request)
    require(p, "deletion.request", matter_id)
    return {"deletions": deletion.listing(p["org_id"], matter_id=matter_id, pending_only=True)}


@router.post("/deletions/{deletion_id}/restore")
@audited("deletion.restore", object_type="deletion", object_param="deletion_id")
def restore_deletion(deletion_id: str, request: Request):
    p = principal_of(request)
    d = deletion.row(p["org_id"], deletion_id)
    if not d or not policy.authorize(p, "deletion.request", d["matter_id"]).allowed:
        raise _not_found()
    note(matter_id=d["matter_id"])
    try:
        return {"deletion": deletion.restore(deletion_id, p["sub"])}
    except deletion.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))


# ── Records staff: deletion approvals ────────────────────────────────────────────
@router.get("/deletions")
def all_deletions(request: Request, status: Optional[Literal["pending", "all"]] = "all"):
    p = principal_of(request)
    require(p, "deletion.review")
    return {"deletions": deletion.listing(p["org_id"], pending_only=status == "pending")}


def _for_decision(p: dict, deletion_id: str) -> dict:
    require(p, "deletion.review")
    d = deletion.row(p["org_id"], deletion_id)
    if not d:
        raise _not_found()
    note(matter_id=d["matter_id"])
    return d


@router.post("/deletions/{deletion_id}/approve")
@audited("deletion.approve", object_type="deletion", object_param="deletion_id")
def approve_deletion(deletion_id: str, request: Request, inp: Optional[NoteIn] = None):
    p = principal_of(request)
    d = _for_decision(p, deletion_id)
    if d["requested_by"] == p["sub"]:
        raise HTTPException(status_code=403, detail=_SECOND)
    try:
        out = deletion.purge(deletion_id, p["sub"], (inp or NoteIn()).note)
    except holds.Held:
        note(reason="legal_hold")
        raise HTTPException(status_code=409, detail="This material is under legal hold; nothing was deleted and the request is blocked.")
    except deletion.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    note(details={"status": out["status"], "verified": bool(out["verification"] and out["verification"]["ok"])})
    return {"deletion": out}


@router.post("/deletions/{deletion_id}/reject")
@audited("deletion.reject", object_type="deletion", object_param="deletion_id")
def reject_deletion(deletion_id: str, request: Request, inp: Optional[NoteIn] = None):
    p = principal_of(request)
    _for_decision(p, deletion_id)
    try:
        return {"deletion": deletion.restore(deletion_id, p["sub"], status="rejected", note=(inp or NoteIn()).note)}
    except deletion.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))


# ── Legal holds ──────────────────────────────────────────────────────────────────
@router.get("/matter-directory")
def matter_directory(request: Request, q: str = ""):
    p = principal_of(request)
    require(p, "matters.directory")
    return {"matters": walls_repo.matter_directory(p["org_id"], q)}


@router.get("/holds")
def list_holds(request: Request):
    p = principal_of(request)
    require(p, "holds.read")
    return {"holds": holds.listing(p["org_id"]), "can_request": "records_admin" in policy.firm_roles_of(p["sub"])}


@router.post("/holds")
@audited("hold.request", object_type="hold")
def request_hold(inp: HoldIn, request: Request):
    p = principal_of(request)
    require(p, "holds.request")
    for mid in inp.matter_ids:
        if not walls_repo.org_has(p["org_id"], "matters", mid):
            raise _not_found()
    hid = holds.create(p["org_id"], inp.name.strip(), inp.reason.strip(), p["sub"], inp.matter_ids)
    note(object_id=hid, details={"matters": len(set(inp.matter_ids))})
    return {"hold": holds.listing(p["org_id"], hid)[0]}


def _hold(p: dict, hold_id: str, action: str, expected: str) -> dict:
    require(p, action)
    h = holds.row(p["org_id"], hold_id)
    if not h:
        raise _not_found()
    if h["status"] != expected:
        raise HTTPException(status_code=409, detail=f"This hold is {h['status'].replace('_', ' ')}.")
    return h


@router.post("/holds/{hold_id}/approve")
@audited("hold.approve", object_type="hold", object_param="hold_id")
def approve_hold(hold_id: str, request: Request):
    p = principal_of(request)
    h = _hold(p, hold_id, "holds.approve", "requested")
    if h["requested_by"] == p["sub"]:
        raise HTTPException(status_code=403, detail=_SECOND)
    holds.update(hold_id, status="active", approved_by=p["sub"], approved_at=policy.now_iso())
    return {"hold": holds.listing(p["org_id"], hold_id)[0]}


@router.post("/holds/{hold_id}/reject")
@audited("hold.reject", object_type="hold", object_param="hold_id")
def reject_hold(hold_id: str, request: Request):
    p = principal_of(request)
    _hold(p, hold_id, "holds.approve", "requested")
    holds.update(hold_id, status="rejected", approved_by=p["sub"], approved_at=policy.now_iso())
    return {"hold": holds.listing(p["org_id"], hold_id)[0]}


@router.post("/holds/{hold_id}/release")
@audited("hold.release.request", object_type="hold", object_param="hold_id")
def request_release(hold_id: str, request: Request):
    p = principal_of(request)
    _hold(p, hold_id, "holds.request", "active")
    holds.update(hold_id, status="release_requested", release_requested_by=p["sub"], release_requested_at=policy.now_iso())
    return {"hold": holds.listing(p["org_id"], hold_id)[0]}


@router.post("/holds/{hold_id}/release/approve")
@audited("hold.release.approve", object_type="hold", object_param="hold_id")
def approve_release(hold_id: str, request: Request):
    p = principal_of(request)
    h = _hold(p, hold_id, "holds.approve", "release_requested")
    if h["release_requested_by"] == p["sub"]:
        raise HTTPException(status_code=403, detail=_SECOND)
    holds.update(hold_id, status="released", released_by=p["sub"], released_at=policy.now_iso())
    return {"hold": holds.listing(p["org_id"], hold_id)[0]}


# ── Retention ────────────────────────────────────────────────────────────────────
def _retention_reader(p: dict) -> None:
    roles = policy.firm_roles_of(p["sub"])
    if not roles & (policy.ORG_ACTIONS["retention.manage"] | policy.ORG_ACTIONS["retention.approve"]):
        require(p, "retention.manage")          # refuses, and records the refusal


@router.get("/retention")
def get_retention(request: Request):
    p = principal_of(request)
    _retention_reader(p)
    from ...core.config import settings
    return {"policies": retention.policies(p["org_id"]), "runs": retention.runs(p["org_id"]),
            "apply_enabled": settings.LEGAL_RETENTION_APPLY}


@router.post("/retention/policies")
@audited("retention.policy.draft", object_type="retention_policy")
def draft_policy(inp: PolicyIn, request: Request):
    p = principal_of(request)
    require(p, "retention.manage")
    pid = retention.draft(p["org_id"], inp.object_type, inp.days_after_close, inp.note, p["sub"])
    note(object_id=pid, details={"object_type": inp.object_type, "days": inp.days_after_close})
    return {"policy": next(x for x in retention.policies(p["org_id"]) if x["id"] == pid)}


@router.post("/retention/policies/{policy_id}/approve")
@audited("retention.policy.approve", object_type="retention_policy", object_param="policy_id")
def approve_policy(policy_id: str, request: Request):
    p = principal_of(request)
    require(p, "retention.approve")
    pol = retention.policy_row(p["org_id"], policy_id)
    if not pol:
        raise _not_found()
    if pol["drafted_by"] == p["sub"]:
        raise HTTPException(status_code=403, detail=_SECOND)
    if pol["status"] != "draft":
        raise HTTPException(status_code=409, detail=f"This policy is {pol['status']}.")
    retention.set_status(policy_id, "approved", p["sub"])
    return {"policy": next(x for x in retention.policies(p["org_id"]) if x["id"] == policy_id)}


@router.post("/retention/policies/{policy_id}/retire")
@audited("retention.policy.retire", object_type="retention_policy", object_param="policy_id")
def retire_policy(policy_id: str, request: Request):
    p = principal_of(request)
    require(p, "retention.manage")
    if not retention.policy_row(p["org_id"], policy_id):
        raise _not_found()
    retention.set_status(policy_id, "retired")
    return {"policy": next(x for x in retention.policies(p["org_id"]) if x["id"] == policy_id)}


@router.post("/retention/run")
@audited("retention.run", object_type="retention_run")
def run_retention(inp: RunIn, request: Request):
    p = principal_of(request)
    require(p, "retention.manage")
    note(details={"mode": inp.mode})
    if inp.mode == "apply":
        why = retention.apply_allowed(p["org_id"])
        if why:
            raise HTTPException(status_code=409, detail=why)
    out = retention.run(p["org_id"], inp.mode, p["sub"])
    note(object_id=out["id"], details={"candidates": out["report"]["candidates"], "held_skipped": out["report"]["held_skipped"]})
    return {"run": out}
