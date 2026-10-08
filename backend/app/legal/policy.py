"""EchoMind Law policy engine (S1.2, SEC-02; IMPLEMENTATION_PLAN §4).

Deny by default. Two layers:
- firm roles (user_roles) grant firm-level actions only, never matter content;
- matter roles (live matter_memberships) grant actions inside one matter.

`decide()` is the pure decision over facts; `authorize()` loads those facts with one query and
calls it. A matter the principal is not a live member of answers exactly like a matter that does
not exist (404 "not_found"), through the same single query, so existence cannot be inferred.
Ethical walls and break-glass are added to this engine in S1.4.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from ..core.db import get_conn
from .identity.sessions import _iso

ALL_MATTER_ROLES = frozenset({"lead", "attorney", "paralegal", "reviewer", "read_only"})
_CONTRIBUTORS = frozenset({"lead", "attorney", "paralegal"})

MATTER_ACTIONS: dict[str, frozenset] = {
    "matter.read": ALL_MATTER_ROLES,
    "documents.read": ALL_MATTER_ROLES,
    "research.query": ALL_MATTER_ROLES,
    "meetings.read": ALL_MATTER_ROLES,
    "matter.update": frozenset({"lead"}),
    "members.manage": frozenset({"lead"}),
    "audit.read.matter": frozenset({"lead"}),
    "documents.import": _CONTRIBUTORS,
    "documents.version": _CONTRIBUTORS,
    "labels.set": frozenset({"lead", "attorney", "reviewer"}),
    "labels.suggest": frozenset({"lead", "attorney", "paralegal", "reviewer"}),
    "meetings.create": _CONTRIBUTORS,
    "consent.capture": _CONTRIBUTORS,
    "capture.control": _CONTRIBUTORS,
    "ai.draft": _CONTRIBUTORS,
    "artifacts.approve": frozenset({"lead", "attorney", "reviewer"}),
    "export.request": _CONTRIBUTORS,
    "export.approve": frozenset({"lead", "attorney"}),
    "deletion.request": frozenset({"lead", "attorney"}),
}

# Content changes refused on closed/archived matters (reading, membership, export and records continue).
BLOCKED_WHEN_CLOSED = frozenset({"documents.import", "documents.version", "labels.set", "labels.suggest",
                                 "meetings.create", "consent.capture", "capture.control", "ai.draft",
                                 "artifacts.approve"})

ORG_ACTIONS: dict[str, frozenset] = {
    "clients.create": frozenset({"managing_partner", "attorney"}),
    "matters.create": frozenset({"managing_partner", "attorney"}),
    "clients.update": frozenset({"managing_partner"}),
    "users.manage": frozenset({"firm_admin"}),
    "roles.manage": frozenset({"firm_admin"}),
    "connectors.manage": frozenset({"firm_admin"}),
    "system.settings": frozenset({"firm_admin"}),
    "audit.read.org": frozenset({"firm_admin", "managing_partner", "records_admin"}),
    "retention.manage": frozenset({"records_admin"}),
}


@dataclass(frozen=True)
class MatterFacts:
    status: str
    sensitivity: str


@dataclass(frozen=True)
class MembershipFacts:
    role: str
    can_export: bool


@dataclass(frozen=True)
class Decision:
    allowed: bool
    status: int = 200          # HTTP status to answer with when refused
    reason: str = "ok"
    matter_id: Optional[str] = None
    role: Optional[str] = None


_NOT_FOUND = Decision(False, 404, "not_found")
_FORBIDDEN = Decision(False, 403, "forbidden")


def decide(action: str, firm_roles: frozenset, matter: Optional[MatterFacts],
           membership: Optional[MembershipFacts]) -> Decision:
    if action in ORG_ACTIONS:
        return Decision(True) if firm_roles & ORG_ACTIONS[action] else _FORBIDDEN
    if action not in MATTER_ACTIONS:
        raise ValueError(f"unknown policy action {action!r}")
    if matter is None or membership is None:
        return _NOT_FOUND                                    # never reveal that the matter exists
    if membership.role not in MATTER_ACTIONS[action]:
        return _FORBIDDEN
    if matter.status != "open" and action in BLOCKED_WHEN_CLOSED:
        return Decision(False, 409, "matter_closed")
    if action == "export.request" and not membership.can_export:
        return _FORBIDDEN
    return Decision(True, role=membership.role)


# ── Loading facts ────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return _iso(time.time())


_LIVE_MEMBERSHIP = ("mm.revoked_at IS NULL AND mm.valid_from <= :now "
                    "AND (mm.valid_to IS NULL OR mm.valid_to > :now)")


def firm_roles_of(user_id: str) -> frozenset:
    with get_conn() as conn:
        return frozenset(r[0] for r in conn.execute("SELECT role FROM user_roles WHERE user_id=?", (user_id,)))


def authorize(principal: dict, action: str, matter_id: Optional[str] = None) -> Decision:
    """Decide for an authenticated principal. Matter facts and the live membership come from ONE
    query whether or not the matter exists or the principal is a member (no timing/shape signal)."""
    roles = firm_roles_of(principal["sub"])
    if action in ORG_ACTIONS:
        return decide(action, roles, None, None)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT m.status, m.sensitivity, mm.role, mm.can_export FROM (SELECT :mid AS id) q "
            "LEFT JOIN matters m ON m.id = q.id AND m.org_id = :org "
            f"LEFT JOIN matter_memberships mm ON mm.matter_id = m.id AND mm.user_id = :uid AND {_LIVE_MEMBERSHIP}",
            {"mid": matter_id or "", "org": principal.get("org_id") or "", "uid": principal["sub"], "now": now_iso()},
        ).fetchone()
    matter = MatterFacts(row[0], row[1]) if row and row[0] else None
    membership = MembershipFacts(row[2], bool(row[3])) if row and row[2] else None
    d = decide(action, roles, matter, membership)
    return Decision(d.allowed, d.status, d.reason, matter_id if d.allowed else None, d.role)


def visible_matters_sql() -> str:
    """WHERE fragment (with :uid, :org, :now) restricting `matters m` to the principal's live memberships."""
    return (f"m.org_id = :org AND EXISTS (SELECT 1 FROM matter_memberships mm WHERE mm.matter_id = m.id "
            f"AND mm.user_id = :uid AND {_LIVE_MEMBERSHIP})")
