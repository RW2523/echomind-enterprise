"""/api/legal/walls and /api/legal/break-glass (S1.4, SEC-02).

Two-person rule throughout: a wall, its release, an exception and a break-glass grant each need a
managing partner other than the person who asked. Break-glass requests name a matter by number,
because the requester cannot see it; the response is the same whether or not that number exists.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import policy
from ..matters import repo as matters_repo
from ..matters.routes import principal_of, require, require_law_edition
from . import repo

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])

_MAX_EXCEPTION_DAYS = 90
_SECOND_PERSON = "A different managing partner must approve this."


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


def _org_bound(principal: dict, ids: list[str], table: str) -> None:
    for oid in ids:
        if not repo.org_has(principal["org_id"], table, oid):
            raise _not_found()


# ── Schemas ──────────────────────────────────────────────────────────────────────
class WallIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=2000)
    screened: list[str] = Field(min_length=1, max_length=200)
    protected_matters: list[str] = Field(default_factory=list, max_length=500)
    protected_clients: list[str] = Field(default_factory=list, max_length=500)


class ExceptionIn(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    matter_id: str
    reason: str = Field(min_length=1, max_length=2000)
    expires_at: datetime


class BreakGlassIn(BaseModel):
    matter_number: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=2000)
    minutes: int = Field(ge=5, le=240)


class DecisionIn(BaseModel):
    note: Optional[str] = Field(default=None, max_length=2000)


# ── Walls ────────────────────────────────────────────────────────────────────────
@router.get("/walls")
def list_walls(request: Request):
    p = principal_of(request)
    require(p, "walls.read")
    return {"walls": repo.list_walls(p["org_id"])}


@router.get("/walls/matter-directory")
def matter_directory(request: Request, q: str = ""):
    """The firm's matters by number/title/client, for choosing what a wall protects (walls authority only)."""
    p = principal_of(request)
    require(p, "walls.read")
    return {"matters": repo.matter_directory(p["org_id"], q)}


@router.post("/walls")
def create_wall(inp: WallIn, request: Request):
    p = principal_of(request)
    require(p, "walls.request")
    if not inp.protected_matters and not inp.protected_clients:
        raise HTTPException(status_code=422, detail="A wall must protect at least one matter or client.")
    screened = []
    for name in dict.fromkeys(inp.screened):
        uid = repo.org_user(p["org_id"], name)
        if not uid:
            raise HTTPException(status_code=400, detail=f"No active colleague named {name!r}.")
        screened.append(uid)
    if "managing_partner" in policy.firm_roles_of(p["sub"]):
        _org_bound(p, inp.protected_matters, "matters")
        _org_bound(p, inp.protected_clients, "clients")
    else:
        # Anyone else may only protect what they can already see (no discovery through walls).
        for mid in inp.protected_matters:
            if not policy.authorize(p, "matter.read", mid).allowed:
                raise _not_found()
        for cid in inp.protected_clients:
            if not matters_repo.get_visible_client(p, cid):
                raise _not_found()
    wid = repo.create_wall(p["org_id"], inp.name, inp.rationale, p["sub"], screened,
                           list(dict.fromkeys(inp.protected_matters)), list(dict.fromkeys(inp.protected_clients)))
    return {"wall": repo.get_wall(p["org_id"], wid)}


def _wall_for_decision(p: dict, wall_id: str, expected_status: str) -> dict:
    require(p, "walls.approve")
    w = repo.get_wall_row(p["org_id"], wall_id)
    if not w:
        raise _not_found()
    if w["status"] != expected_status:
        raise HTTPException(status_code=409, detail=f"This wall is {w['status'].replace('_', ' ')}.")
    return w


def _bump_wall(wall_id: str) -> None:
    matters, clients = repo.protected_of(wall_id)
    matters_repo.bump_epochs(matters, clients)


@router.post("/walls/{wall_id}/approve")
def approve_wall(wall_id: str, request: Request):
    p = principal_of(request)
    w = _wall_for_decision(p, wall_id, "requested")
    if w["requested_by"] == p["sub"]:
        raise HTTPException(status_code=403, detail=_SECOND_PERSON)
    repo.set_wall_status(wall_id, "active", approved_by=p["sub"], approved_at=policy.now_iso())
    _bump_wall(wall_id)
    return {"wall": repo.get_wall(p["org_id"], wall_id)}


@router.post("/walls/{wall_id}/reject")
def reject_wall(wall_id: str, request: Request):
    p = principal_of(request)
    _wall_for_decision(p, wall_id, "requested")
    repo.set_wall_status(wall_id, "rejected", approved_by=p["sub"], approved_at=policy.now_iso())
    return {"wall": repo.get_wall(p["org_id"], wall_id)}


@router.post("/walls/{wall_id}/release")
def request_release(wall_id: str, request: Request):
    p = principal_of(request)
    _wall_for_decision(p, wall_id, "active")
    repo.set_wall_status(wall_id, "release_requested", release_requested_by=p["sub"], release_requested_at=policy.now_iso())
    return {"wall": repo.get_wall(p["org_id"], wall_id)}


@router.post("/walls/{wall_id}/release/approve")
def approve_release(wall_id: str, request: Request):
    p = principal_of(request)
    w = _wall_for_decision(p, wall_id, "release_requested")
    if w["release_requested_by"] == p["sub"]:
        raise HTTPException(status_code=403, detail=_SECOND_PERSON)
    repo.set_wall_status(wall_id, "released", released_by=p["sub"], released_at=policy.now_iso())
    _bump_wall(wall_id)
    return {"wall": repo.get_wall(p["org_id"], wall_id)}


# ── Exceptions ───────────────────────────────────────────────────────────────────
@router.post("/walls/{wall_id}/exceptions")
def request_exception(wall_id: str, inp: ExceptionIn, request: Request):
    p = principal_of(request)
    require(p, "walls.request")
    w = repo.get_wall_row(p["org_id"], wall_id)
    if not w or w["status"] not in ("active", "release_requested"):
        raise _not_found()
    expires = inp.expires_at if inp.expires_at.tzinfo else inp.expires_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if not now < expires <= now + timedelta(days=_MAX_EXCEPTION_DAYS):
        raise HTTPException(status_code=422, detail=f"An exception must end within {_MAX_EXCEPTION_DAYS} days.")
    uid = repo.org_user(p["org_id"], inp.username)
    if not uid or not repo.wall_covers(wall_id, uid, inp.matter_id):
        raise HTTPException(status_code=400, detail="That person is not screened from that matter by this wall.")
    iso = expires.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return {"exception": repo.create_exception(wall_id, uid, inp.matter_id, inp.reason, p["sub"], iso)}


def _exception(p: dict, wall_id: str, exc_id: str) -> dict:
    require(p, "walls.approve")
    if not repo.get_wall_row(p["org_id"], wall_id):
        raise _not_found()
    e = repo.get_exception(wall_id, exc_id)
    if not e:
        raise _not_found()
    return e


@router.post("/walls/{wall_id}/exceptions/{exc_id}/approve")
def approve_exception(wall_id: str, exc_id: str, request: Request):
    p = principal_of(request)
    e = _exception(p, wall_id, exc_id)
    if e["requested_by"] == p["sub"]:
        raise HTTPException(status_code=403, detail=_SECOND_PERSON)
    if e["approved_at"] or e["revoked_at"]:
        raise HTTPException(status_code=409, detail="This exception was already decided.")
    repo.approve_exception(exc_id, p["sub"])
    matters_repo.bump_epochs([e["matter_id"]])
    return {"ok": True}


@router.delete("/walls/{wall_id}/exceptions/{exc_id}")
def revoke_exception(wall_id: str, exc_id: str, request: Request):
    p = principal_of(request)
    e = _exception(p, wall_id, exc_id)
    repo.revoke_exception(exc_id, p["sub"])
    matters_repo.bump_epochs([e["matter_id"]])
    return {"ok": True}


# ── Break-glass ──────────────────────────────────────────────────────────────────
def _public_request(g: dict) -> dict:
    """What the requester sees: identical fields whether or not the number exists."""
    return {"id": g["id"], "matter_number": g["requested_number"], "status": g["status"], "minutes": g["minutes"],
            "reason": g["reason"], "requested_at": g["requested_at"], "expires_at": g["expires_at"]}


@router.post("/break-glass")
def request_break_glass(inp: BreakGlassIn, request: Request):
    p = principal_of(request)
    require(p, "breakglass.request")
    g = repo.create_break_glass(p["org_id"], p["sub"], inp.matter_number, inp.reason, inp.minutes)
    return {"request": _public_request(g)}


@router.get("/break-glass")
def list_break_glass(request: Request):
    p = principal_of(request)
    require(p, "breakglass.read")
    if "managing_partner" in policy.firm_roles_of(p["sub"]):
        out = []
        for g in repo.list_break_glass(p["org_id"]):
            m = repo.matter_by_number(p["org_id"], g["requested_number"])
            out.append({**_public_request(g), "requester": g["requester"], "decided_by": g["decided_by_name"],
                        "decision_note": g["decision_note"],
                        "matter": {"number": m["number"], "title": m["title"], "client": m["client"],
                                   "sensitivity": m["sensitivity"]} if m else None})
        return {"requests": out, "can_approve": True}
    return {"requests": [_public_request(g) for g in repo.list_break_glass(p["org_id"], p["sub"])], "can_approve": False}


def _bg_for_decision(p: dict, gid: str) -> dict:
    require(p, "breakglass.approve")
    g = repo.get_break_glass(p["org_id"], gid)
    if not g:
        raise _not_found()
    if g["status"] != "requested":
        raise HTTPException(status_code=409, detail=f"This request is already {g['status']}.")
    if g["user_id"] == p["sub"]:
        raise HTTPException(status_code=403, detail=_SECOND_PERSON)
    return g


@router.post("/break-glass/{gid}/approve")
def approve_break_glass(gid: str, request: Request, inp: Optional[DecisionIn] = None):
    p = principal_of(request)
    g = _bg_for_decision(p, gid)
    m = repo.matter_by_number(p["org_id"], g["requested_number"])
    if not m:
        raise HTTPException(status_code=404, detail="No matter has that number.")
    if m["sensitivity"] == "highly_restricted":
        raise HTTPException(status_code=409, detail="Emergency access is not available for highly restricted matters.")
    if policy.is_walled(g["user_id"], m["id"]):
        raise HTTPException(status_code=409, detail="The requester is screened from this matter by an ethical wall.")
    repo.decide_break_glass(gid, "active", p["sub"], (inp or DecisionIn()).note or "", matter_id=m["id"],
                            expires_at=repo.iso_in(g["minutes"]))
    matters_repo.bump_epochs([m["id"]])
    return {"request": _public_request(repo.get_break_glass(p["org_id"], gid))}


@router.post("/break-glass/{gid}/reject")
def reject_break_glass(gid: str, request: Request, inp: Optional[DecisionIn] = None):
    p = principal_of(request)
    _bg_for_decision(p, gid)
    repo.decide_break_glass(gid, "rejected", p["sub"], (inp or DecisionIn()).note or "")
    return {"request": _public_request(repo.get_break_glass(p["org_id"], gid))}


@router.post("/break-glass/{gid}/end")
def end_break_glass(gid: str, request: Request):
    """The requester, or any managing partner, ends emergency access early."""
    p = principal_of(request)
    g = repo.get_break_glass(p["org_id"], gid)
    is_mp = "managing_partner" in policy.firm_roles_of(p["sub"])
    if not g or (g["user_id"] != p["sub"] and not is_mp):
        raise _not_found()
    if g["status"] == "active":
        repo.end_break_glass(gid, p["sub"])
        if g["matter_id"]:
            matters_repo.bump_epochs([g["matter_id"]])
    return {"request": _public_request(repo.get_break_glass(p["org_id"], gid))}
