"""EchoMind Law policy engine (S1.2, SEC-02; IMPLEMENTATION_PLAN §4).

Deny by default. Two layers:
- firm roles (user_roles) grant firm-level actions only, never matter content;
- matter roles (live matter_memberships) grant actions inside one matter.

`decide()` is the pure decision over facts; `authorize()` loads those facts with one query and
calls it. A matter the principal is not a live member of answers exactly like a matter that does
not exist (404 "not_found"), through the same single query, so existence cannot be inferred.

Ethical walls (S1.4) bind everyone, including a matter's lead: a walled matter is "not_found"
unless an approved, unexpired exception exists. Break-glass gives a non-member time-limited,
read-only access once a different managing partner approved it; it never crosses a wall and never
reaches a highly restricted matter.
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
    # S4.1 (D-054): the matter's party register, which conflict searches read. Kept on closed matters too.
    "parties.manage": frozenset({"lead", "attorney"}),
    # S4.4 (D-068): deciding whether a regulatory update concerns the matter.
    "regulatory.decide": frozenset({"lead", "attorney", "paralegal", "reviewer"}),
}

# What an approved break-glass grant allows (read only).
BREAK_GLASS_ACTIONS = frozenset({"matter.read", "documents.read", "research.query", "meetings.read"})
BREAK_GLASS_ROLE = "break_glass"

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
    # Ethical walls (S1.4). Managing partners are the walls authority: they see every wall and, for
    # that purpose, the firm's matter directory (numbers, titles, clients) — decision D-022.
    "walls.read": frozenset({"managing_partner"}),
    "walls.request": frozenset({"managing_partner", "attorney"}),
    "walls.approve": frozenset({"managing_partner"}),
    "breakglass.request": frozenset({"firm_admin"}),
    "breakglass.read": frozenset({"firm_admin", "managing_partner"}),
    "breakglass.approve": frozenset({"managing_partner"}),
    "users.read": frozenset({"firm_admin", "managing_partner"}),
    # Records (S1.9; P-002/P-003). Records administrators and managing partners see the firm's
    # matter directory to place holds and review deletions (D-027); content stays matter-only.
    "matters.directory": frozenset({"managing_partner", "records_admin"}),
    "holds.read": frozenset({"records_admin", "managing_partner"}),
    "holds.request": frozenset({"records_admin"}),
    "holds.approve": frozenset({"records_admin", "managing_partner"}),
    "deletion.review": frozenset({"records_admin", "managing_partner"}),
    "retention.approve": frozenset({"managing_partner"}),
    # Firm templates (S3.1, D-045; P-008): written by lawyers, approved by a managing partner other than the author.
    "templates.manage": frozenset({"managing_partner", "attorney"}),
    "templates.approve": frozenset({"managing_partner"}),
    # Playbooks (S3.2, D-046): positions written by lawyers, approved by a managing partner other than the author.
    "playbooks.manage": frozenset({"managing_partner", "attorney"}),
    "playbooks.approve": frozenset({"managing_partner"}),
    # Due-diligence checklists (S3.7, D-051): written by lawyers, approved by a managing partner other than the author.
    "diligence.manage": frozenset({"managing_partner", "attorney"}),
    "diligence.approve": frozenset({"managing_partner"}),
    # Intake (S4.1, D-054; OPT-01): front desk and lawyers take enquiries; lawyers approve replies; the conflicts
    # team (and managing partners, the walls authority) see screening matches and decide; records administrators purge.
    "intake.manage": frozenset({"intake", "paralegal", "attorney", "managing_partner"}),
    "intake.approve": frozenset({"attorney", "managing_partner"}),
    "conflicts.review": frozenset({"conflicts", "managing_partner"}),
    "intake.purge": frozenset({"records_admin"}),
    # S4.4 regulatory updates (OPT-04, D-068): public law, read by the firm's lawyers.
    "regulatory.read": frozenset({"managing_partner", "attorney", "paralegal", "reviewer"}),
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
           membership: Optional[MembershipFacts], *, walled: bool = False, break_glass: bool = False) -> Decision:
    if action in ORG_ACTIONS:
        return Decision(True) if firm_roles & ORG_ACTIONS[action] else _FORBIDDEN
    if action not in MATTER_ACTIONS:
        raise ValueError(f"unknown policy action {action!r}")
    if matter is None or walled:
        return _NOT_FOUND                                    # walls bind everyone, leads included
    if membership is None:
        if break_glass and matter.sensitivity != "highly_restricted":
            if action in BREAK_GLASS_ACTIONS:
                return Decision(True, role=BREAK_GLASS_ROLE)
            return _FORBIDDEN                                # read-only emergency access
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

# :uid is screened from matter `m` by an active wall (on the matter or its client) without an
# approved, unexpired, unrevoked exception for that matter.
_WALLED = ("EXISTS (SELECT 1 FROM wall_screened ws JOIN ethical_walls w ON w.id = ws.wall_id "
           "AND w.status IN ('active','release_requested') JOIN wall_protected wp ON wp.wall_id = w.id "
           "WHERE ws.user_id = :uid AND ((wp.object_type = 'matter' AND wp.object_id = m.id) "
           "OR (wp.object_type = 'client' AND wp.object_id = m.client_id)) "
           "AND NOT EXISTS (SELECT 1 FROM wall_exceptions we WHERE we.wall_id = w.id AND we.user_id = :uid "
           "AND we.matter_id = m.id AND we.approved_at IS NOT NULL AND we.revoked_at IS NULL AND we.expires_at > :now))")

# :uid is screened from client `c` itself by an active wall (client-level protection).
_CLIENT_WALLED = ("EXISTS (SELECT 1 FROM wall_screened ws JOIN ethical_walls w ON w.id = ws.wall_id "
                  "AND w.status IN ('active','release_requested') JOIN wall_protected wp ON wp.wall_id = w.id "
                  "WHERE ws.user_id = :uid AND wp.object_type = 'client' AND wp.object_id = c.id)")

# :uid holds an active, approved break-glass grant for matter `m`.
_BREAK_GLASS = ("EXISTS (SELECT 1 FROM break_glass_grants bg WHERE bg.user_id = :uid AND bg.matter_id = m.id "
                "AND bg.status = 'active' AND bg.expires_at > :now)")


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
            f"SELECT m.status, m.sensitivity, mm.role, mm.can_export, {_WALLED}, {_BREAK_GLASS} FROM (SELECT :mid AS id) q "
            "LEFT JOIN matters m ON m.id = q.id AND m.org_id = :org "
            f"LEFT JOIN matter_memberships mm ON mm.matter_id = m.id AND mm.user_id = :uid AND {_LIVE_MEMBERSHIP}",
            {"mid": matter_id or "", "org": principal.get("org_id") or "", "uid": principal["sub"], "now": now_iso()},
        ).fetchone()
    matter = MatterFacts(row[0], row[1]) if row and row[0] else None
    membership = MembershipFacts(row[2], bool(row[3])) if row and row[2] else None
    d = decide(action, roles, matter, membership, walled=bool(row and row[4]), break_glass=bool(row and row[5]))
    return Decision(d.allowed, d.status, d.reason, matter_id if d.allowed else None, d.role)


def visible_matters_sql() -> str:
    """WHERE fragment (with :uid, :org, :now) restricting `matters m` to what the principal may see:
    a live membership or an active break-glass grant (never highly restricted), and no wall."""
    return (f"m.org_id = :org AND NOT {_WALLED} AND (EXISTS (SELECT 1 FROM matter_memberships mm "
            f"WHERE mm.matter_id = m.id AND mm.user_id = :uid AND {_LIVE_MEMBERSHIP}) "
            f"OR ({_BREAK_GLASS} AND m.sensitivity <> 'highly_restricted'))")


def my_role_sql() -> str:
    """Column expression (with :uid, :now) for the principal's role on matter `m`."""
    return (f"COALESCE((SELECT mm.role FROM matter_memberships mm WHERE mm.matter_id = m.id AND mm.user_id = :uid "
            f"AND {_LIVE_MEMBERSHIP}), CASE WHEN {_BREAK_GLASS} THEN '{BREAK_GLASS_ROLE}' END)")


def is_walled(user_id: str, matter_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(f"SELECT {_WALLED} FROM matters m WHERE m.id = :mid",
                           {"uid": user_id, "mid": matter_id, "now": now_iso()}).fetchone()
    return bool(row and row[0])
