"""Phase 0b auth routes: /api/auth/{config,login,logout,me}. Active only when AUTH_ENABLED."""
import logging

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from ...core.config import settings
from ...core import auth as authmod

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
    token = authmod.make_token(user["id"], user["role"], user["username"])
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
    return {"user": {"id": payload.get("sub"), "username": payload.get("username"), "role": payload.get("role")}}
