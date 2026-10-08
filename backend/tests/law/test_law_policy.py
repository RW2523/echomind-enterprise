"""S1.2 / SEC-02 — the policy engine, as a full decision matrix.

The expected permissions below are written out from IMPLEMENTATION_PLAN.md §4.1 independently of
the engine's own tables: if either drifts from the plan, this test fails.
"""
from __future__ import annotations

import itertools

import pytest

from app.legal.policy import MatterFacts, MembershipFacts, decide

ALL = {"lead", "attorney", "paralegal", "reviewer", "read_only"}
MATTER_SPEC = {
    "matter.read": ALL, "documents.read": ALL, "research.query": ALL, "meetings.read": ALL,
    "matter.update": {"lead"}, "members.manage": {"lead"}, "audit.read.matter": {"lead"},
    "documents.import": {"lead", "attorney", "paralegal"}, "documents.version": {"lead", "attorney", "paralegal"},
    "labels.set": {"lead", "attorney", "reviewer"}, "labels.suggest": {"lead", "attorney", "paralegal", "reviewer"},
    "meetings.create": {"lead", "attorney", "paralegal"}, "consent.capture": {"lead", "attorney", "paralegal"},
    "capture.control": {"lead", "attorney", "paralegal"}, "ai.draft": {"lead", "attorney", "paralegal"},
    "artifacts.approve": {"lead", "attorney", "reviewer"},
    "export.request": {"lead", "attorney", "paralegal"}, "export.approve": {"lead", "attorney"},
    "deletion.request": {"lead", "attorney"},
}
# Changing the content of a closed or archived matter is refused; reading, membership, export and
# records work (deletion requests) continue.
BLOCKED_WHEN_CLOSED = {"documents.import", "documents.version", "labels.set", "labels.suggest", "meetings.create",
                       "consent.capture", "capture.control", "ai.draft", "artifacts.approve"}
FIRM_ROLES = ["firm_admin", "managing_partner", "attorney", "paralegal", "reviewer", "records_admin", "read_only"]
ORG_SPEC = {
    "clients.create": {"managing_partner", "attorney"}, "matters.create": {"managing_partner", "attorney"},
    "clients.update": {"managing_partner"},
    "users.manage": {"firm_admin"}, "roles.manage": {"firm_admin"}, "connectors.manage": {"firm_admin"},
    "system.settings": {"firm_admin"},
    "audit.read.org": {"firm_admin", "managing_partner", "records_admin"},
    "retention.manage": {"records_admin"},
}

OPEN = MatterFacts(status="open", sensitivity="restricted")
CLOSED = MatterFacts(status="closed", sensitivity="restricted")
ARCHIVED = MatterFacts(status="archived", sensitivity="highly_restricted")


@pytest.mark.parametrize("action, role, matter",
                         list(itertools.product(sorted(MATTER_SPEC), sorted(ALL), [OPEN, CLOSED, ARCHIVED])))
def test_matter_action_matrix(action, role, matter):
    d = decide(action, frozenset(), matter, MembershipFacts(role=role, can_export=True))
    if role not in MATTER_SPEC[action]:
        assert (d.allowed, d.status) == (False, 403), d
    elif matter.status != "open" and action in BLOCKED_WHEN_CLOSED:
        assert (d.allowed, d.status, d.reason) == (False, 409, "matter_closed"), d
    else:
        assert d.allowed, d


@pytest.mark.parametrize("action", sorted(MATTER_SPEC))
@pytest.mark.parametrize("firm_roles", [frozenset(), frozenset(FIRM_ROLES)])
def test_non_members_and_missing_matters_look_identical(action, firm_roles):
    """Even holding every firm role, without a live membership the matter does not exist (404)."""
    no_member = decide(action, firm_roles, OPEN, None)
    missing = decide(action, firm_roles, None, None)
    assert (no_member.allowed, no_member.status, no_member.reason) == (False, 404, "not_found")
    assert (missing.status, missing.reason) == (no_member.status, no_member.reason)


@pytest.mark.parametrize("action, firm_role", list(itertools.product(sorted(ORG_SPEC), FIRM_ROLES + [None])))
def test_org_action_matrix(action, firm_role):
    roles = frozenset({firm_role}) if firm_role else frozenset()
    d = decide(action, roles, None, None)
    if firm_role in ORG_SPEC[action]:
        assert d.allowed, d
    else:
        assert (d.allowed, d.status) == (False, 403), d


def test_export_needs_the_export_right_too():
    assert not decide("export.request", frozenset(), OPEN, MembershipFacts("lead", can_export=False)).allowed
    assert decide("export.request", frozenset(), OPEN, MembershipFacts("lead", can_export=True)).allowed


def test_unknown_actions_fail_closed():
    with pytest.raises(ValueError):
        decide("matter.teleport", frozenset({"firm_admin"}), OPEN, MembershipFacts("lead", True))


def test_spec_and_engine_cover_the_same_actions():
    from app.legal import policy
    assert set(policy.MATTER_ACTIONS) == set(MATTER_SPEC)
    assert set(ORG_SPEC) <= set(policy.ORG_ACTIONS)
