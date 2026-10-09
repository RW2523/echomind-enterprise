"""/api/legal/users: firm accounts and firm roles (S1.2; completes S1.1's revoke-on-role-change).

Firm administrators manage firm roles. Removing any role signs the person out everywhere, so a
reduced set of rights applies at once to every open session, stream and socket. Adding a role does
not interrupt anyone. An administrator cannot remove their own firm_admin role (no lock-out).
"""
from __future__ import annotations

import time
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...core.db import get_conn
from ..audit.record import audited, note
from ..matters.routes import principal_of, require, require_law_edition
from ..policy import firm_roles_of
from . import sessions
from .sessions import _iso

FirmRole = Literal["firm_admin", "managing_partner", "attorney", "paralegal", "reviewer", "records_admin", "read_only"]

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


class UserOut(BaseModel):
    username: str
    display_name: str
    status: str
    roles: list[str]
    mfa_enabled: bool


class UserList(BaseModel):
    users: list[UserOut]


class RolesIn(BaseModel):
    roles: list[FirmRole]


class RolesOut(BaseModel):
    user: UserOut
    sessions_revoked: int


def _user_out(user_id: str) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT username, COALESCE(display_name, username), COALESCE(status,'active'), mfa_enabled "
                           "FROM users WHERE id=?", (user_id,)).fetchone()
    return {"username": row[0], "display_name": row[1], "status": row[2], "roles": sorted(firm_roles_of(user_id)),
            "mfa_enabled": bool(row[3])}


def _org_user_id(org_id: str, username: str) -> Optional[str]:
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM users WHERE username=? AND org_id=?", (username, org_id)).fetchone()
    return row[0] if row else None


@router.get("/users", response_model=UserList)
def list_users(request: Request):
    """Firm admins manage accounts; managing partners need the list to choose whom a wall screens."""
    p = principal_of(request)
    require(p, "users.read")
    with get_conn() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM users WHERE org_id=? ORDER BY username", (p["org_id"],))]
    return {"users": [_user_out(i) for i in ids]}


@router.put("/users/{username}/roles", response_model=RolesOut)
@audited("roles.set", object_type="user")
def set_roles(username: str, inp: RolesIn, request: Request):
    p = principal_of(request)
    require(p, "roles.manage")
    target = _org_user_id(p["org_id"], username)
    if not target:
        raise HTTPException(status_code=404, detail="Not Found")
    new = set(inp.roles)
    if target == p["sub"] and "firm_admin" not in new:
        raise HTTPException(status_code=400, detail="You cannot remove your own firm_admin role.")
    old = set(firm_roles_of(target))
    note(object_id=target, details={"username": username, "added": sorted(new - old), "removed": sorted(old - new)})
    with get_conn() as conn:
        for role in old - new:
            conn.execute("DELETE FROM user_roles WHERE user_id=? AND role=?", (target, role))
        for role in new - old:
            conn.execute("INSERT INTO user_roles (user_id, role, granted_by, granted_at) VALUES (?,?,?,?)",
                         (target, role, p["sub"], _iso(time.time())))
        conn.commit()
    revoked = sessions.revoke_all_for_user(target, reason="role_changed") if old - new else 0
    return {"user": _user_out(target), "sessions_revoked": revoked}
