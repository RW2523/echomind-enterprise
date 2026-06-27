"""Phase 0b auth routes: /api/auth/{config,login,logout,me}. Active only when AUTH_ENABLED."""
import logging

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ...core.config import settings
from ...core import auth as authmod
from ...core import audit as auditmod

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str


@router.get("/config")
def auth_config():
    """Public: lets the frontend decide whether to show the login gate."""
    return {"auth_enabled": settings.AUTH_ENABLED}


@router.post("/login")
def login(inp: LoginIn, response: Response):
    user = authmod.authenticate(inp.username, inp.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = authmod.make_token(user["id"], user["role"], user["username"], tenant=user.get("tenant", ""))
    response.set_cookie(
        "echomind_token", token, httponly=True, samesite="lax",
        max_age=settings.AUTH_TOKEN_TTL_MIN * 60,
    )
    return {"token": token, "user": user}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("echomind_token")
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    payload = authmod.user_from_request(request)
    if not payload:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"user": {"id": payload.get("sub"), "username": payload.get("username"), "role": payload.get("role"), "tenant": payload.get("tenant", "")}}


# ── RBAC: admin-only user management ─────────────────────────────────────────────
class CreateUserIn(BaseModel):
    username: str
    password: str
    role: str = "user"
    tenant: str = ""


def _require_admin(request: Request) -> dict:
    payload = authmod.user_from_request(request)
    if not payload:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if (payload.get("role") or "") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return payload


@router.get("/users")
def admin_list_users(request: Request):
    _require_admin(request)
    return {"users": authmod.list_users()}


@router.post("/users")
def admin_create_user(request: Request, inp: CreateUserIn):
    _require_admin(request)
    role = inp.role if inp.role in ("admin", "user") else "user"
    try:
        user = authmod.create_user(inp.username, inp.password, role=role, tenant=inp.tenant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"user": user}


@router.delete("/users/{username}")
def admin_delete_user(request: Request, username: str):
    payload = _require_admin(request)
    if username == payload.get("username"):
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    authmod.delete_user(username)
    return {"ok": True, "deleted": username}


@router.get("/audit")
def admin_audit(request: Request):
    """Admin-only: recent activity (who did what)."""
    _require_admin(request)
    return {"events": auditmod.recent_activity(150)}


@router.get("/usage")
def admin_usage(request: Request):
    """Admin-only: usage metering summary (per-user request counts)."""
    _require_admin(request)
    return auditmod.usage_summary()
