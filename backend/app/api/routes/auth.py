"""Phase 0b auth routes: /api/auth/{config,login,logout,me}. Active only when AUTH_ENABLED.

In the EchoMind Law edition (S1.1, SEC-01) the same routes run on server-side sessions with
mandatory MFA, lockout and CSRF, and the Law-only routes below (MFA, sessions, password,
on-behalf-of tokens, account disable/enable) become available. Legacy editions are unchanged.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from ...core.config import settings
from ...core import auth as authmod
from ...core import audit as auditmod
from ...core.edition import is_law
from ...legal.audit.record import audited, note

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str


@router.get("/config")
def auth_config(request: Request):
    """Public: lets the frontend decide whether to show the login gate, and which edition's
    interface to load (the EchoMind Law workspace when the edition is "law")."""
    from ...core.edition import current_edition
    out = {"auth_enabled": settings.AUTH_ENABLED, "edition": current_edition() or "standard"}
    if is_law():
        from ...legal.identity import oidc
        if oidc.enabled():
            out["oidc"] = {"label": settings.LEGAL_OIDC_LABEL}       # D-076: the sign-in page offers the firm's account
        from ...legal.identity import demo
        if demo.available(request):
            out["demo_signin"] = True                                    # D-077: the sign-in page lists the demo people
    return out


@router.post("/login")
@audited("auth.login", session_actor=False)
def login(inp: LoginIn, request: Request, response: Response):
    if is_law():
        return _law_login(inp, request, response)
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
@audited("auth.logout")
def logout(request: Request, response: Response):
    if is_law():
        from ...legal.identity import SESSION_COOKIE, sessions
        sessions.revoke(request.cookies.get(SESSION_COOKIE, ""), reason="logout")
        _clear_law_cookies(response)
        return {"ok": True}
    # Revoke server-side too: clearing the cookie alone left the token valid until expiry (NC-2026-012).
    authmod.revoke_token(authmod.token_from_request(request))
    response.delete_cookie("echomind_token")
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    payload = authmod.user_from_request(request)
    if not payload:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if is_law():
        return _law_me(payload)
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
    if is_law():
        # Law edition: administration is the firm_admin firm role (S1.2), not the legacy flag.
        from ...legal.policy import firm_roles_of
        if "firm_admin" not in firm_roles_of(payload["sub"]):
            raise HTTPException(status_code=403, detail="Admin privileges required")
        return payload
    if (payload.get("role") or "") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return payload


@router.get("/users")
def admin_list_users(request: Request):
    _require_admin(request)
    return {"users": authmod.list_users()}


@router.post("/users")
@audited("user.create", object_type="user")
def admin_create_user(request: Request, inp: CreateUserIn):
    admin = _require_admin(request)
    if is_law():
        # Law edition: `role` names a firm role ("admin" is accepted as firm_admin); the legacy
        # users.role flag is kept only for compatibility.
        from ...legal.identity import accounts
        firm_role = "firm_admin" if inp.role == "admin" else inp.role
        if firm_role not in accounts.FIRM_ROLES and firm_role != "user":
            raise HTTPException(status_code=400, detail=f"Unknown role. Use one of: {', '.join(accounts.FIRM_ROLES)}")
        _enforce_password_policy(inp.password, inp.username)
        try:
            user = authmod.create_user(inp.username, inp.password, role="admin" if firm_role == "firm_admin" else "user")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if firm_role != "user":
            accounts.grant_role(user["id"], firm_role, granted_by=admin["sub"])
        note(object_id=user["id"], details={"username": user["username"], "role": firm_role})
        # D-021: the administrator chose this password, so the person must replace it; and binding
        # an authenticator needs this one-time link, not just the password.
        accounts.require_password_change(user["id"])
        return {"user": {**user, "roles": accounts.roles_of(user["id"])},
                "enrollment": _enrollment_link(user["id"], admin["sub"])}
    role = inp.role if inp.role in ("admin", "user") else "user"
    try:
        user = authmod.create_user(inp.username, inp.password, role=role, tenant=inp.tenant)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"user": user}


@router.delete("/users/{username}")
@audited("user.delete", object_type="user")
def admin_delete_user(request: Request, username: str):
    payload = _require_admin(request)
    if username == payload.get("username"):
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    if is_law():
        from ...legal.identity import accounts, sessions
        acct = accounts.get_by_username(username)
        note(object_id=acct["id"] if acct else None, details={"username": username})
        if acct:
            sessions.revoke_all_for_user(acct["id"], reason="account_deleted")
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


# ═════════════════════════════════════════════════════════════════════════════════
# EchoMind Law edition (S1.1, SEC-01)
# ═════════════════════════════════════════════════════════════════════════════════
_SIGN_IN_FAILED = "Sign-in failed. Check your username and password, or try again later."
# Verified against for unknown usernames so a miss costs the same time as a wrong password.
_DUMMY_HASH = authmod.hash_password("echomind-law-timing-equaliser")


def _require_law() -> None:
    if not is_law():
        raise HTTPException(status_code=404, detail="Not Found")


def _client_meta(request: Request) -> tuple[str, str]:
    from ...legal.netaddr import client_address
    return (client_address(request) or ""), (request.headers.get("user-agent") or "")      # the client's, through nginx (D-075)


def _set_law_cookies(response: Response, token: str, csrf: str, mfa_verified: bool) -> None:
    from ...legal.identity import CSRF_COOKIE, SESSION_COOKIE
    max_age = int(settings.LEGAL_SESSION_ABSOLUTE_HOURS * 3600) if mfa_verified else settings.LEGAL_MFA_PENDING_MIN * 60
    # __Host- prefix: Secure, Path=/, no Domain. The CSRF cookie is readable by the SPA, which echoes
    # it in X-CSRF-Token; the session cookie is HttpOnly and never visible to scripts.
    response.set_cookie(SESSION_COOKIE, token, max_age=max_age, path="/", secure=True, httponly=True, samesite="strict")
    response.set_cookie(CSRF_COOKIE, csrf, max_age=max_age, path="/", secure=True, httponly=False, samesite="strict")


def _clear_law_cookies(response: Response) -> None:
    from ...legal.identity import CSRF_COOKIE, SESSION_COOKIE
    for name, http_only in ((SESSION_COOKIE, True), (CSRF_COOKIE, False)):
        response.delete_cookie(name, path="/", secure=True, httponly=http_only, samesite="strict")


def _public_user(acct: dict) -> dict:
    from ...legal.identity import accounts
    return {"id": acct["id"], "username": acct["username"], "role": acct["role"], "tenant": acct.get("tenant", ""),
            "roles": accounts.roles_of(acct["id"])}


def _enforce_password_policy(password: str, username: str) -> None:
    from ...legal.identity import passwords
    problems = passwords.problems(password, username)
    if problems:
        raise HTTPException(status_code=400, detail=passwords.describe(problems))


def _law_login(inp: LoginIn, request: Request, response: Response) -> dict:
    from ...legal.identity import accounts, sessions
    acct = accounts.get_by_username(inp.username)
    if acct is None:
        authmod.verify_password(inp.password, _DUMMY_HASH)
        note(reason="unknown_user")               # the name typed is not recorded
        raise HTTPException(status_code=401, detail=_SIGN_IN_FAILED)
    note(actor_user_id=acct["id"], actor_username=acct["username"])
    if accounts.is_locked(acct) or acct["status"] != "active":
        note(reason="disabled" if acct["status"] != "active" else "locked")
        authmod.verify_password(inp.password, acct["password_hash"])
        logger.warning("Auth: sign-in refused for locked or disabled account %s", acct["id"])
        raise HTTPException(status_code=401, detail=_SIGN_IN_FAILED)
    if not authmod.verify_password(inp.password, acct["password_hash"]):
        accounts.register_failure(acct["id"])
        note(reason="bad_password")
        raise HTTPException(status_code=401, detail=_SIGN_IN_FAILED)
    ip, ua = _client_meta(request)
    # Failure counters reset only once sign-in is complete, so knowing the password does not buy
    # unlimited guesses at the MFA code.
    if acct["mfa_enabled"] or settings.LEGAL_MFA_REQUIRED:
        token, csrf = sessions.create(acct["id"], mfa_verified=False, ip=ip, user_agent=ua)
        _set_law_cookies(response, token, csrf, mfa_verified=False)
        status = "mfa_required" if acct["mfa_enabled"] else "mfa_enrollment_required"
        note(details={"status": status})
        return {"status": status}
    accounts.register_success(acct["id"])
    token, csrf = sessions.create(acct["id"], mfa_verified=True, ip=ip, user_agent=ua)
    _set_law_cookies(response, token, csrf, mfa_verified=True)
    note(details={"status": "ok"})
    return {"status": "ok", "user": _public_user(acct)}


# ── One-click demo sign-in (D-077) ─────────────────────────────────────────────────
@router.get("/demo")
def demo_people(request: Request):
    """The demo seed's fictional people, for the sign-in page; 404 unless demo sign-in is on for this address."""
    from ...legal.identity import accounts, demo
    if not demo.available(request):
        raise HTTPException(status_code=404, detail="Not Found")
    people = []
    for name in demo.people():
        acct = accounts.get_by_username(name)
        if acct and acct["status"] == "active":
            people.append({"username": name, "roles": sorted(accounts.roles_of(acct["id"]))})
    return {"people": people}


@router.post("/demo/{username}")
@audited("auth.demo_signin", session_actor=False)
def demo_signin(username: str, request: Request, response: Response):
    """Enter as a demo person: a full session, no password or code. Never the administrator or a real account."""
    from ...legal.identity import accounts, demo, sessions
    if not demo.available(request) or username not in demo.people():
        raise HTTPException(status_code=404, detail="Not Found")
    acct = accounts.get_by_username(username)
    if acct is None:
        raise HTTPException(status_code=404, detail="Not Found")
    note(actor_user_id=acct["id"], actor_username=acct["username"])
    if accounts.is_locked(acct) or acct["status"] != "active":
        raise HTTPException(status_code=403, detail="This demo person is not active.")
    accounts.register_success(acct["id"])
    ip, ua = _client_meta(request)
    token, csrf = sessions.create(acct["id"], mfa_verified=True, ip=ip, user_agent=ua)
    _set_law_cookies(response, token, csrf, mfa_verified=True)
    return {"status": "ok", "user": _public_user(acct)}


# ── Single sign-on with the firm's identity provider (D-076) ────────────────────────
def _require_oidc() -> None:
    from ...legal.identity import oidc
    if not is_law() or not oidc.enabled():
        raise HTTPException(status_code=404, detail="Not Found")


def _without_state_cookie(r: RedirectResponse) -> RedirectResponse:
    from ...legal.identity import oidc
    r.delete_cookie(oidc.STATE_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
    return r


@router.get("/oidc/start")
@audited("auth.oidc.start", session_actor=False)
def oidc_start(request: Request):
    """Send the browser to the firm's identity provider, with a state bound to it by its own cookie."""
    _require_oidc()
    from ...legal.identity import oidc
    try:
        url, browser = oidc.start()
    except Exception as e:
        note(reason="discovery", outcome="deny")
        logger.warning("Auth: the identity provider could not be reached (%s)", type(e).__name__)
        return RedirectResponse("/signin?oidc_error=unavailable", status_code=302)
    r = RedirectResponse(url, status_code=302)
    r.set_cookie(oidc.STATE_COOKIE, browser, max_age=oidc.STATE_TTL, path="/", secure=True, httponly=True, samesite="lax")
    return r


@router.get("/oidc/callback")
@audited("auth.oidc", session_actor=False)
def oidc_callback(request: Request, code: str = "", state: str = ""):
    """The provider sends the browser back here. An existing, active account is signed in; this server's own code is
    still asked for unless the firm trusts the provider's MFA and the token says it was used. Refusals redirect to the
    sign-in page with a general message; the audit trail keeps the reason, never a token or a claim."""
    _require_oidc()
    from ...legal.identity import accounts, oidc, sessions

    def refused(reason: str) -> RedirectResponse:
        note(reason=reason, outcome="deny")
        kind = "account" if reason in ("no_account", "account_unavailable") else "failed"
        return _without_state_cookie(RedirectResponse(f"/signin?oidc_error={kind}", status_code=302))

    try:
        claims = oidc.finish(code, state, request.cookies.get(oidc.STATE_COOKIE))
    except oidc.Refused as e:
        return refused(e.reason)
    name = oidc.username_of(claims)
    acct = accounts.get_by_username(name) if name else None
    if acct is None:
        return refused("no_account")                    # nothing is created here: an administrator adds people
    note(actor_user_id=acct["id"], actor_username=acct["username"])
    if accounts.is_locked(acct) or acct["status"] != "active":
        return refused("account_unavailable")
    full = oidc.mfa_by_provider(claims)
    if full:
        accounts.register_success(acct["id"])
    ip, ua = _client_meta(request)
    token, csrf = sessions.create(acct["id"], mfa_verified=full, ip=ip, user_agent=ua)
    where = "/home" if full else ("/signin?step=code" if acct["mfa_enabled"] else "/signin?step=enrol")
    r = _without_state_cookie(RedirectResponse(where, status_code=302))
    _set_law_cookies(r, token, csrf, mfa_verified=full)
    note(details={"provider_mfa": full})
    return r


def _law_me(payload: dict) -> dict:
    from ...legal.identity import accounts
    acct = accounts.get(payload["sub"])
    s = payload["session"]
    return {"user": _public_user(acct),
            "session": {"auth": payload["auth"], "mfa_enrolled": acct["mfa_enabled"],
                        "must_change_password": bool(payload.get("must_change_password")),
                        "idle_expires_at": s["idle_expires_at"], "abs_expires_at": s["abs_expires_at"]},
            "features": {"intake": bool(settings.LEGAL_INTAKE), "intake_links": bool(settings.LEGAL_INTAKE and settings.LEGAL_INTAKE_LINKS),
                         "summaries": bool(settings.LEGAL_SUMMARIES), "time_entries": bool(settings.LEGAL_TIME_ENTRIES),
                         "panel": bool(settings.LEGAL_PANEL), "conversation": bool(settings.LEGAL_CONVERSATION),
                         "regulatory": bool(settings.LEGAL_REGULATORY), "translation": bool(settings.LEGAL_TRANSLATION),
                         "demo_signin": bool(settings.LEGAL_DEMO_SIGNIN)}}


def _complete_sign_in(request: Request, response: Response, user_id: str) -> dict:
    """Upgrade a partial session to a full one under a fresh token (no session fixation)."""
    from ...legal.identity import SESSION_COOKIE, accounts, sessions
    accounts.register_success(user_id)
    current = request.cookies.get(SESSION_COOKIE, "")
    s = sessions.lookup(current, slide=False)
    if s and s["mfa_verified"]:
        return {"status": "ok", "user": _public_user(accounts.get(user_id))}
    token, csrf = sessions.rotate(current, mfa_verified=True)
    _set_law_cookies(response, token, csrf, mfa_verified=True)
    return {"status": "ok", "user": _public_user(accounts.get(user_id))}


def _failed_code(request: Request, user_id: str) -> None:
    from ...legal.identity import SESSION_COOKIE, accounts, sessions
    accounts.register_failure(user_id)
    note(reason="bad_code")
    if accounts.is_locked(accounts.get(user_id)):
        note(reason="locked_out")
        sessions.revoke(request.cookies.get(SESSION_COOKIE, ""), reason="locked_out")
    raise HTTPException(status_code=401, detail="That code was not accepted.")


class CodeIn(BaseModel):
    code: str


@router.post("/mfa/verify")
@audited("auth.mfa.verify")
def mfa_verify(inp: CodeIn, request: Request, response: Response):
    """Second sign-in step: a current authenticator code, or an unused recovery code."""
    _require_law()
    from ...legal.identity import mfa
    p = authmod.user_from_request(request, allow_partial=True)
    if not mfa.verify_login_code(p["sub"], inp.code):
        _failed_code(request, p["sub"])
    return _complete_sign_in(request, response, p["sub"])


class EnrollIn(BaseModel):
    enrollment_token: Optional[str] = None


class ConfirmIn(BaseModel):
    code: str
    enrollment_token: Optional[str] = None


_LINK_REQUIRED = "An enrolment link from your administrator is required to set up an authenticator."


def _check_link(user_id: str, token: Optional[str]) -> None:
    from ...legal.identity import mfa
    if settings.LEGAL_ENROLLMENT_LINK_REQUIRED and not mfa.enrollment_token_valid(user_id, token or ""):
        raise HTTPException(status_code=403, detail=_LINK_REQUIRED)


@router.post("/mfa/enroll")
@audited("auth.mfa.enroll")
def mfa_enroll(request: Request, inp: Optional[EnrollIn] = None):
    """Start authenticator enrolment; returns the shared secret once. Needs a valid enrolment link
    unless the firm has relaxed that (D-021)."""
    _require_law()
    from ...legal.identity import mfa
    p = authmod.user_from_request(request, allow_partial=True)
    if mfa.is_enrolled(p["sub"]):
        raise HTTPException(status_code=409, detail="An authenticator is already enrolled for this account.")
    _check_link(p["sub"], (inp or EnrollIn()).enrollment_token)
    secret = mfa.begin_enrollment(p["sub"])
    return {"secret": secret, "otpauth_uri": mfa.otpauth_uri(secret, p["username"])}


@router.post("/mfa/confirm")
@audited("auth.mfa.confirm")
def mfa_confirm(inp: ConfirmIn, request: Request, response: Response):
    """Finish enrolment with a code from the new authenticator; returns recovery codes once.
    The enrolment link is used up only once the code is accepted."""
    _require_law()
    from ...legal.identity import mfa
    p = authmod.user_from_request(request, allow_partial=True)
    _check_link(p["sub"], inp.enrollment_token)
    if not mfa.confirm_enrollment(p["sub"], inp.code):
        _failed_code(request, p["sub"])
    if settings.LEGAL_ENROLLMENT_LINK_REQUIRED:
        mfa.consume_enrollment_token(p["sub"], inp.enrollment_token or "")
    codes = mfa.issue_recovery_codes(p["sub"])
    return {**_complete_sign_in(request, response, p["sub"]), "recovery_codes": codes}


@router.get("/sessions")
def list_sessions(request: Request):
    _require_law()
    from ...legal.identity import sessions
    p = authmod.user_from_request(request)
    current = p["sid"][:16]
    return {"sessions": [{**s, "current": s["id"] == current} for s in sessions.list_for_user(p["sub"])]}


@router.delete("/sessions/{session_id}")
@audited("session.revoke", object_type="session", object_param="session_id")
def revoke_session(session_id: str, request: Request, response: Response):
    _require_law()
    from ...legal.identity import sessions
    p = authmod.user_from_request(request)
    if not sessions.revoke_by_public_id(p["sub"], session_id, reason="revoked_by_user"):
        raise HTTPException(status_code=404, detail="Not Found")
    if session_id == p["sid"][:16]:
        _clear_law_cookies(response)
    return {"ok": True}


class PasswordIn(BaseModel):
    current_password: str
    new_password: str


@router.post("/password")
@audited("auth.password.change")
def change_password(inp: PasswordIn, request: Request):
    _require_law()
    from ...legal.identity import SESSION_COOKIE, accounts
    p = authmod.user_from_request(request)
    acct = accounts.get(p["sub"])
    if not authmod.verify_password(inp.current_password, acct["password_hash"]):
        note(reason="bad_password")
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    _enforce_password_policy(inp.new_password, acct["username"])
    revoked = accounts.change_password(acct["id"], inp.new_password, keep_token=request.cookies.get(SESSION_COOKIE, ""))
    return {"ok": True, "revoked_sessions": revoked}


class OboIn(BaseModel):
    audience: str


@router.post("/obo-token")
@audited("auth.obo.issue")
def obo_token(inp: OboIn, request: Request):
    """Exchange a browser session for a short-lived token a service (voice) uses on the user's behalf."""
    _require_law()
    from ...legal.identity import obo
    p = authmod.user_from_request(request)
    if p["auth"] != "session":
        raise HTTPException(status_code=401, detail="Not authenticated")
    note(details={"audience": inp.audience})
    if inp.audience != "voice":
        raise HTTPException(status_code=400, detail="Unknown audience")
    return {"token": obo.issue(p["session"], audience="voice"), "expires_in": settings.LEGAL_OBO_TTL_SEC}


def _admin_target(request: Request, username: str) -> dict:
    _require_law()
    from ...legal.identity import accounts
    _require_admin(request)
    acct = accounts.get_by_username(username)
    if not acct:
        raise HTTPException(status_code=404, detail="Not Found")
    note(object_type="user", object_id=acct["id"], details={"username": acct["username"]})
    return acct


def _enrollment_link(user_id: str, created_by: str) -> dict:
    from ...legal.identity import mfa
    token, expires = mfa.issue_enrollment_token(user_id, created_by=created_by)
    return {"token": token, "expires_at": expires, "path": f"/signin?enrol={token}"}


@router.post("/users/{username}/enrollment-link")
@audited("user.enrollment_link", object_type="user")
def admin_enrollment_link(username: str, request: Request):
    """Issue a one-time authenticator enrolment link (shown once; replaces older links)."""
    from ...legal.identity import mfa
    acct = _admin_target(request, username)
    if mfa.is_enrolled(acct["id"]):
        raise HTTPException(status_code=409, detail="This account already has an authenticator. Reset it first.")
    return {"enrollment": _enrollment_link(acct["id"], authmod.user_from_request(request)["sub"])}


@router.post("/users/{username}/reset-mfa")
@audited("user.mfa_reset", object_type="user")
def admin_reset_mfa(username: str, request: Request):
    """Forget a lost authenticator: signs the person out everywhere and issues a new enrolment link."""
    from ...legal.identity import mfa, sessions
    acct = _admin_target(request, username)
    mfa.reset(acct["id"])
    sessions.revoke_all_for_user(acct["id"], reason="mfa_reset")
    return {"ok": True, "enrollment": _enrollment_link(acct["id"], authmod.user_from_request(request)["sub"])}


@router.post("/users/{username}/revoke-sessions")
@audited("user.sessions_revoke", object_type="user")
def admin_revoke_sessions(username: str, request: Request):
    from ...legal.identity import sessions
    acct = _admin_target(request, username)
    return {"ok": True, "revoked": sessions.revoke_all_for_user(acct["id"], reason="revoked_by_admin")}


@router.post("/users/{username}/disable")
@audited("user.disable", object_type="user")
def admin_disable_user(username: str, request: Request):
    from ...legal.identity import accounts
    acct = _admin_target(request, username)
    if acct["id"] == authmod.user_from_request(request)["sub"]:
        raise HTTPException(status_code=400, detail="You cannot disable your own account")
    accounts.set_status(username, "disabled")
    return {"ok": True}


@router.post("/users/{username}/enable")
@audited("user.enable", object_type="user")
def admin_enable_user(username: str, request: Request):
    from ...legal.identity import accounts
    _admin_target(request, username)
    accounts.set_status(username, "active")
    accounts.register_success(accounts.get_by_username(username)["id"])  # clear any lockout
    return {"ok": True}
