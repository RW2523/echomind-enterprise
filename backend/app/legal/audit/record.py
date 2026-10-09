"""Writing audit events (S1.5, SEC-04; decision D-008).

Three ways an event is written, so that every Law action leaves exactly one:

- `@audited("matter.create")` on a route: one event per call, allow on success, deny (401/403/404)
  or error otherwise. Routes add the object and safe details with `note(...)`.
- `policy_decision(...)`, called by the routes' `require()`: inside an audited route the decision
  feeds that route's event; elsewhere it writes an event for every refusal and every access made
  under break-glass.
- `request_finished(...)`, from the Law guard: any state-changing request that wrote nothing else
  (the existing chat, document and meeting routes until S1.7 scopes them) gets an `api.request`
  event naming the route, never its body.

Events hold identifiers, actions, outcomes and counts. Free text never enters `details`: only
values under the keys in SAFE_TEXT_KEYS are kept as strings; any other string becomes
{"chars": n}.
"""
from __future__ import annotations

import functools
import inspect
import json
import logging
import re
import uuid
from contextvars import ContextVar
from typing import Any, Optional

from fastapi import HTTPException

from ...core.edition import is_law
from .. import policy
from ..orgs import default_org_id
from . import chain

logger = logging.getLogger(__name__)

# String values kept verbatim in details: identifiers and controlled vocabulary, never prose.
SAFE_TEXT_KEYS = frozenset({
    "action", "code", "fields", "format", "method", "number", "outcome", "policy", "role", "roles",
    "route", "sensitivity", "status", "username", "user_id", "audience", "added", "removed", "wall",
    "sha256", "kind", "mime", "modes", "citations", "method", "transcription", "label", "object_type", "mode", "verified",
})
_MAX_SAFE_TEXT = 120
_IDENT = re.compile(r"^[A-Za-z0-9_.:@-]{1,64}$")
_DENY_STATUSES = frozenset({401, 403, 404})

_ctx: ContextVar[Optional[dict]] = ContextVar("law_audit", default=None)


# ── Request context ──────────────────────────────────────────────────────────────
def begin(request) -> Any:
    """Open the audit context for one HTTP request (called by the Law guard)."""
    return _ctx.set({"ip": request.client.host if request.client else None, "recorded": False, "notes": {}})


def end(token) -> None:
    _ctx.reset(token)


def note(**fields) -> None:
    """Add to the current route's event: object_type, object_id, matter_id, reason, details,
    actor_user_id, actor_username. Details merge."""
    ctx = _ctx.get()
    if ctx is None:
        return
    details = fields.pop("details", None)
    ctx["notes"].update(fields)
    if details:
        ctx["notes"].setdefault("details", {}).update(details)


# ── Sanitising ───────────────────────────────────────────────────────────────────
def _safe(value: Any, key: str = "") -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if key in SAFE_TEXT_KEYS and len(value) <= _MAX_SAFE_TEXT:
            return value
        return {"chars": len(value)}
    if isinstance(value, dict):
        return {str(k): _safe(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(v, key) for v in value]
    return {"chars": len(str(value))}


def safe_details(details: Optional[dict]) -> Optional[str]:
    if not details:
        return None
    return json.dumps(_safe(details), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _ident(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    return s if _IDENT.match(s) else "invalid"


# ── Writing ──────────────────────────────────────────────────────────────────────
def record(action: str, outcome: str = "allow", *, principal: Optional[dict] = None,
           actor_user_id: Optional[str] = None, actor_username: Optional[str] = None,
           object_type: Optional[str] = None, object_id: Any = None, matter_id: Any = None,
           reason: Optional[str] = None, details: Optional[dict] = None) -> None:
    """Append one event. Raises if it cannot be written: callers fail closed."""
    ctx = _ctx.get()
    p = principal or {}
    uid = actor_user_id or p.get("sub")
    session = p.get("sid") or ""
    auth = p.get("auth")
    if auth == "session" and not p.get("mfa_verified", True):
        auth = "partial"
    chain.append({
        "id": str(uuid.uuid4()),
        "ts": policy.now_iso(),
        "org_id": p.get("org_id") or default_org_id(),
        "actor_user_id": uid,
        "actor_username": actor_username or p.get("username"),
        "actor_roles": ",".join(sorted(policy.firm_roles_of(uid))) if uid else None,
        "auth": auth,
        "session_ref": session[:16] or None,
        "ip": (ctx or {}).get("ip"),
        "action": action,
        "object_type": object_type,
        "object_id": _ident(object_id),
        "matter_id": _ident(matter_id),
        "outcome": outcome,
        "reason": reason,
        "details_json": safe_details(details),
    })
    if ctx is not None:
        ctx["recorded"] = True


def system_event(action: str, outcome: str = "allow", **kw) -> None:
    """An event with no person behind it (start-up, scheduled jobs, the CLI)."""
    record(action, outcome, **kw)


# ── Policy decisions ─────────────────────────────────────────────────────────────
def policy_decision(principal: dict, action: str, decision: "policy.Decision", matter_id: Optional[str]) -> None:
    ctx = _ctx.get()
    if ctx is not None and ctx.get("action"):
        ctx["decision"] = (action, decision, matter_id)      # the audited route's own event carries it
        return
    under_break_glass = decision.allowed and decision.role == policy.BREAK_GLASS_ROLE
    if decision.allowed and not under_break_glass:
        return
    record(action, "allow" if decision.allowed else "deny", principal=principal,
           matter_id=matter_id if action in policy.MATTER_ACTIONS else None,
           object_type="matter" if action in policy.MATTER_ACTIONS else None,
           object_id=matter_id if action in policy.MATTER_ACTIONS else None,
           reason="break_glass" if under_break_glass else decision.reason)


# ── Route decorator ──────────────────────────────────────────────────────────────
def _principal_for(request) -> Optional[dict]:
    if request is None:
        return None
    from ...core import auth as authmod
    try:
        return authmod.user_from_request(request, allow_partial=True)
    except Exception:
        return None


def _finish(ctx: dict, action: str, object_type: Optional[str], kwargs: dict, object_param: Optional[str],
            principal: Optional[dict], outcome: str, status: Optional[int]) -> None:
    n = ctx["notes"]
    decided = ctx.get("decision")
    d = decided[1] if decided else None
    if n.get("reason"):
        reason = n["reason"]
    elif outcome != "allow" and d is not None and not d.allowed:
        reason = d.reason
    elif outcome == "allow" and d is not None and d.role == policy.BREAK_GLASS_ROLE:
        reason = "break_glass"
    elif outcome != "allow":
        reason = f"http_{status}"
    else:
        reason = None
    matter_id = n.get("matter_id") or kwargs.get("matter_id") or (decided[2] if decided else None)
    details = dict(n.get("details") or {})
    if decided and decided[0] != action:
        details.setdefault("policy", decided[0])
    record(action, outcome, principal=principal,
           actor_user_id=n.get("actor_user_id"), actor_username=n.get("actor_username"),
           object_type=n.get("object_type", object_type),
           object_id=n.get("object_id") or (kwargs.get(object_param) if object_param else None),
           matter_id=matter_id, reason=reason, details=details or None)


def _audit_failed(action: str, exc: Exception) -> HTTPException:
    logger.error("Audit: could not record %s (%s)", action, type(exc).__name__)
    return HTTPException(status_code=500, detail="This action could not be recorded in the audit trail.")


def audited(action: str, *, object_type: Optional[str] = None, object_param: Optional[str] = None,
            session_actor: bool = True):
    """Record one event per call of the decorated route (Law edition only). If the event cannot be
    written the caller gets a 500: a read returns no data; a change is reported as failed.
    session_actor=False (sign-in): the actor comes only from note(), never from a cookie the
    browser still holds from an earlier session."""
    def wrap(fn):
        def _setup(kwargs):
            request = kwargs.get("request")
            ctx = _ctx.get()
            token = None
            if ctx is None:
                token = _ctx.set({"ip": request.client.host if request is not None and request.client else None,
                                  "recorded": False, "notes": {}})
                ctx = _ctx.get()
            ctx.update(action=action, notes={}, decision=None)
            return ctx, token, _principal_for(request) if session_actor else None

        def _close(ctx, token, kwargs, principal, outcome, status):
            try:
                _finish(ctx, action, object_type, kwargs, object_param, principal, outcome, status)
            except Exception as e:
                raise _audit_failed(action, e) from None
            finally:
                ctx["action"] = None
                if token is not None:
                    _ctx.reset(token)

        def _outcome(e: Exception) -> tuple[str, Optional[int]]:
            if isinstance(e, HTTPException):
                return ("deny" if e.status_code in _DENY_STATUSES else "error"), e.status_code
            return "error", 500

        if inspect.iscoroutinefunction(fn):
            async def _aclose(ctx, token, kwargs, principal, outcome, status):
                from starlette.concurrency import run_in_threadpool
                try:      # the database write runs off the event loop
                    await run_in_threadpool(_finish, ctx, action, object_type, kwargs, object_param, principal, outcome, status)
                except Exception as e:
                    raise _audit_failed(action, e) from None
                finally:
                    ctx["action"] = None
                    if token is not None:
                        _ctx.reset(token)

            @functools.wraps(fn)
            async def inner(*args, **kwargs):
                if not is_law():
                    return await fn(*args, **kwargs)
                ctx, token, principal = _setup(kwargs)
                try:
                    result = await fn(*args, **kwargs)
                except Exception as e:
                    await _aclose(ctx, token, kwargs, principal, *_outcome(e))
                    raise
                await _aclose(ctx, token, kwargs, principal, "allow", None)
                return result
        else:
            @functools.wraps(fn)
            def inner(*args, **kwargs):
                if not is_law():
                    return fn(*args, **kwargs)
                ctx, token, principal = _setup(kwargs)
                try:
                    result = fn(*args, **kwargs)
                except Exception as e:
                    _close(ctx, token, kwargs, principal, *_outcome(e))
                    raise
                _close(ctx, token, kwargs, principal, "allow", None)
                return result
        # FastAPI reads the signature through the wrapper; resolve postponed annotations against the
        # route's own module, not this one.
        inner.__signature__ = inspect.signature(fn, eval_str=True)
        inner.__audit_action__ = action
        return inner
    return wrap


# ── The guard's fallback ─────────────────────────────────────────────────────────
_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def request_finished(request, status: int) -> None:
    """A state-changing request that wrote no event of its own (including one refused before its
    route ran, e.g. invalid input) gets a generic event naming the route — never its content."""
    ctx = _ctx.get()
    if ctx is None or ctx["recorded"] or request.method not in _MUTATING:
        return
    route = request.scope.get("route")
    endpoint = request.scope.get("endpoint")
    action = getattr(endpoint, "__audit_action__", None) or "api.request"
    outcome = "allow" if status < 400 else ("deny" if status in _DENY_STATUSES else "error")
    record(action, outcome, principal=_principal_for(request),
           reason=None if status < 400 else f"http_{status}",
           details={"method": request.method, "route": getattr(route, "path", None) or "unmatched", "status": status})


def guard_refused(request, status: int, reason: str) -> None:
    """A signed-in person's request refused by the Law guard itself (e.g. a failed CSRF check)."""
    principal = _principal_for(request)
    if principal is None:
        return       # anonymous refusals are not events: anyone could flood the trail with them
    record("request.refused", "deny", principal=principal, reason=reason,
           details={"method": request.method, "status": status})
