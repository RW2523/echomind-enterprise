"""/api/legal/clients and /api/legal/matters (S1.2, SEC-02).

Every route asks the policy engine first. A matter or client the caller cannot see answers 404
"Not Found", exactly like one that does not exist; 403 is used only where existence is already
known to the caller (a member lacking a right, or a firm-level action).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ...core import auth as authmod
from ...core.edition import is_law
from .. import policy
from ..audit.record import audited, note, policy_decision
from . import repo

MatterRole = Literal["lead", "attorney", "paralegal", "reviewer", "read_only"]
Sensitivity = Literal["standard", "restricted", "highly_restricted"]


def require_law_edition() -> None:
    if not is_law():
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])

_DETAIL = {
    "not_found": "Not Found",
    "forbidden": "You don't have permission to do that.",
    "matter_closed": "This matter is closed. Reopen it to make changes.",
}


def principal_of(request: Request) -> dict:
    p = authmod.user_from_request(request)
    if not p:  # the guard already enforces this; kept so routes never run anonymously
        raise HTTPException(status_code=401, detail="Not authenticated")
    return p


def require(principal: dict, action: str, matter_id: Optional[str] = None) -> policy.Decision:
    d = policy.authorize(principal, action, matter_id)
    policy_decision(principal, action, d, matter_id)
    if not d.allowed:
        raise HTTPException(status_code=d.status, detail=_DETAIL[d.reason])
    return d


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ── Schemas ──────────────────────────────────────────────────────────────────────
class ClientIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    code: Optional[str] = Field(default=None, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")


class ClientPatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    code: Optional[str] = Field(default=None, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    status: Optional[Literal["active", "archived"]] = None
    version: int


class ClientOut(BaseModel):
    id: str
    name: str
    code: Optional[str] = None
    status: str
    version: int


class ClientRef(BaseModel):
    id: str
    name: str
    code: Optional[str] = None


class MatterIn(BaseModel):
    client_id: str
    number: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=300)
    practice_area: Optional[str] = Field(default=None, max_length=120)
    jurisdiction: Optional[str] = Field(default=None, max_length=120)
    sensitivity: Sensitivity = "restricted"


class MatterPatch(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=300)
    practice_area: Optional[str] = Field(default=None, max_length=120)
    jurisdiction: Optional[str] = Field(default=None, max_length=120)
    status: Optional[Literal["open", "closed", "archived"]] = None
    sensitivity: Optional[Sensitivity] = None
    version: int


class MatterOut(BaseModel):
    id: str
    number: str
    title: str
    practice_area: Optional[str] = None
    jurisdiction: Optional[str] = None
    status: str
    sensitivity: str
    version: int
    created_at: str
    updated_at: Optional[str] = None
    client: ClientRef
    my_role: Optional[str] = None


class MembershipIn(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    role: MatterRole
    can_export: bool = False
    valid_to: Optional[datetime] = None


class MembershipPatch(BaseModel):
    role: Optional[MatterRole] = None
    can_export: Optional[bool] = None
    valid_to: Optional[datetime] = None


class MembershipOut(BaseModel):
    id: str
    user_id: str
    username: str
    display_name: str
    role: str
    can_export: bool
    valid_from: str
    valid_to: Optional[str] = None
    granted_at: str


class ClientList(BaseModel):
    clients: list[ClientOut]


class ClientEnvelope(BaseModel):
    client: ClientOut


class MatterList(BaseModel):
    matters: list[MatterOut]


class MatterEnvelope(BaseModel):
    matter: MatterOut


class MembershipList(BaseModel):
    memberships: list[MembershipOut]


class MembershipEnvelope(BaseModel):
    membership: MembershipOut


class Ok(BaseModel):
    ok: bool = True


# ── Clients ──────────────────────────────────────────────────────────────────────
@router.get("/clients", response_model=ClientList)
def list_clients(request: Request):
    return {"clients": repo.list_visible_clients(principal_of(request))}


@router.post("/clients", response_model=ClientEnvelope)
@audited("client.create", object_type="client")
def create_client(inp: ClientIn, request: Request):
    p = principal_of(request)
    require(p, "clients.create")
    try:
        c = repo.create_client(p["org_id"], inp.name, inp.code, p["sub"])
    except repo.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    note(object_id=c["id"], details={"code": c["code"]})
    return {"client": c}


@router.get("/clients/{client_id}", response_model=ClientEnvelope)
def get_client(client_id: str, request: Request):
    c = repo.get_visible_client(principal_of(request), client_id)
    if not c:
        raise _not_found()
    return {"client": c}


@router.patch("/clients/{client_id}", response_model=ClientEnvelope)
@audited("client.update", object_type="client", object_param="client_id")
def update_client(client_id: str, inp: ClientPatch, request: Request):
    p = principal_of(request)
    if not repo.get_visible_client(p, client_id):
        raise _not_found()
    require(p, "clients.update")
    note(details={"fields": sorted(inp.model_dump(exclude={"version"}, exclude_none=True)), "status": inp.status})
    try:
        return {"client": repo.update_client(client_id, inp.model_dump(exclude={"version"}), inp.version)}
    except repo.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))


# ── Matters ──────────────────────────────────────────────────────────────────────
@router.get("/matters", response_model=MatterList)
def list_matters(request: Request):
    return {"matters": repo.list_visible_matters(principal_of(request))}


@router.post("/matters", response_model=MatterEnvelope)
@audited("matter.create", object_type="matter")
def create_matter(inp: MatterIn, request: Request):
    p = principal_of(request)
    require(p, "matters.create")
    if not repo.get_visible_client(p, inp.client_id):
        raise _not_found()
    try:
        m = repo.create_matter(p["org_id"], inp.client_id, inp.number, inp.title, p["sub"],
                               practice_area=inp.practice_area, jurisdiction=inp.jurisdiction, sensitivity=inp.sensitivity)
    except repo.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    note(object_id=m["id"], matter_id=m["id"], details={"sensitivity": m["sensitivity"]})
    return {"matter": m}


@router.get("/matters/{matter_id}", response_model=MatterEnvelope)
@audited("matter.read", object_type="matter", object_param="matter_id")
def get_matter(matter_id: str, request: Request):
    d = require(principal_of(request), "matter.read", matter_id)
    return {"matter": repo.get_matter(matter_id, my_role=d.role)}


@router.patch("/matters/{matter_id}", response_model=MatterEnvelope)
@audited("matter.update", object_type="matter", object_param="matter_id")
def update_matter(matter_id: str, inp: MatterPatch, request: Request):
    d = require(principal_of(request), "matter.update", matter_id)
    current = repo.get_matter(matter_id)
    changes = inp.model_dump(exclude={"version"}, exclude_none=True)
    note(details={"fields": sorted(changes), "status": changes.get("status"), "sensitivity": changes.get("sensitivity")})
    if current["status"] != "open" and set(changes) - {"status"}:
        raise HTTPException(status_code=409, detail=_DETAIL["matter_closed"])
    try:
        m = repo.update_matter(matter_id, changes, inp.version)
    except repo.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"matter": {**m, "my_role": d.role}}


# ── Memberships ──────────────────────────────────────────────────────────────────
@router.get("/matters/{matter_id}/memberships", response_model=MembershipList)
def list_memberships(matter_id: str, request: Request):
    require(principal_of(request), "matter.read", matter_id)
    return {"memberships": repo.list_live_memberships(matter_id)}


@router.post("/matters/{matter_id}/memberships", response_model=MembershipEnvelope)
@audited("membership.add", object_type="membership")
def add_membership(matter_id: str, inp: MembershipIn, request: Request):
    p = principal_of(request)
    require(p, "members.manage", matter_id)
    note(details={"username": inp.username, "role": inp.role, "can_export": inp.can_export})
    colleague = repo.active_colleague(p["org_id"], inp.username)
    if not colleague:
        raise HTTPException(status_code=400, detail="No active colleague with that username.")
    try:
        mm = repo.add_membership(matter_id, colleague["id"], inp.role, granted_by=p["sub"],
                                 can_export=inp.can_export, valid_to=_iso(inp.valid_to))
    except repo.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    note(object_id=mm["id"])
    return {"membership": mm}


@router.patch("/matters/{matter_id}/memberships/{membership_id}", response_model=MembershipEnvelope)
@audited("membership.change", object_type="membership", object_param="membership_id")
def change_membership(matter_id: str, membership_id: str, inp: MembershipPatch, request: Request):
    p = principal_of(request)
    require(p, "members.manage", matter_id)
    note(details={"fields": sorted(inp.model_dump(exclude_none=True)), "role": inp.role, "can_export": inp.can_export})
    try:
        mm = repo.change_membership(matter_id, membership_id, by=p["sub"], role=inp.role,
                                    can_export=inp.can_export, valid_to=_iso(inp.valid_to))
    except LookupError:
        raise _not_found()
    except repo.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    note(details={"username": mm["username"]})
    return {"membership": mm}


@router.delete("/matters/{matter_id}/memberships/{membership_id}", response_model=Ok)
@audited("membership.revoke", object_type="membership", object_param="membership_id")
def remove_membership(matter_id: str, membership_id: str, request: Request):
    p = principal_of(request)
    require(p, "members.manage", matter_id)
    try:
        repo.revoke_membership(matter_id, membership_id, by=p["sub"])
    except LookupError:
        raise _not_found()
    except repo.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"ok": True}
