"""S1.2 / SEC-02, spec test T01 — clients, matters and memberships over HTTP.

World (fictional): client Alpha with M-101 and M-114, client Beta with M-202 (highly restricted)
and M-215. alice leads M-101/M-114; bob leads M-202/M-215; carol is read-only on M-101 and her
M-215 membership was revoked; dave's M-202 membership expired; admin is firm_admin only; mp is a
managing partner with no memberships.
"""
from __future__ import annotations

import sqlite3
import uuid

from conftest import _ts, as_user, csrf_headers, numbers


# ── Visibility ───────────────────────────────────────────────────────────────────
def test_each_person_sees_exactly_their_live_matters(world):
    expected = {"alice": {"M-101", "M-114"}, "bob": {"M-202", "M-215"}, "carol": {"M-101"},
                "dave": set(), "mp": set(), "admin": set()}
    for name, want in expected.items():
        assert numbers(as_user(world, name).get("/api/legal/matters")) == want, name


def test_a_hidden_matter_is_indistinguishable_from_a_missing_one(world):
    alice = as_user(world, "alice")
    hidden, missing = world["matters"]["M-202"]["id"], str(uuid.uuid4())
    for method, suffix, body in [("GET", "", None), ("PATCH", "", {"title": "x", "version": 1}),
                                 ("GET", "/memberships", None),
                                 ("POST", "/memberships", {"username": "alice", "role": "lead"})]:
        r1 = alice.request(method, f"/api/legal/matters/{hidden}{suffix}", json=body, headers=csrf_headers(alice))
        r2 = alice.request(method, f"/api/legal/matters/{missing}{suffix}", json=body, headers=csrf_headers(alice))
        assert (r1.status_code, r1.json()) == (r2.status_code, r2.json()) == (404, {"detail": "Not Found"}), (method, suffix)


def test_hidden_and_missing_matters_run_the_same_queries(world, monkeypatch):
    from contextlib import contextmanager
    from app.legal import policy
    statements = []
    real = policy.get_conn

    @contextmanager
    def traced():
        with real() as conn:
            conn.set_trace_callback(statements.append)
            yield conn

    monkeypatch.setattr(policy, "get_conn", traced)
    p = {"sub": world["users"]["alice"]["id"], "org_id": world["matters"]["M-202"]["org_id"]}
    policy.authorize(p, "matter.read", world["matters"]["M-202"]["id"])
    hidden = [s for s in statements if not s.startswith("PRAGMA")]
    statements.clear()
    policy.authorize(p, "matter.read", str(uuid.uuid4()))
    missing = [s for s in statements if not s.startswith("PRAGMA")]
    assert len(hidden) == len(missing) == 2              # firm roles + one combined matter/membership query
    assert [s.split(" WHERE ")[0][:60] for s in hidden] == [s.split(" WHERE ")[0][:60] for s in missing]


def test_revoked_and_expired_memberships_grant_nothing(world):
    assert as_user(world, "carol").get(f"/api/legal/matters/{world['matters']['M-215']['id']}").status_code == 404
    assert as_user(world, "dave").get(f"/api/legal/matters/{world['matters']['M-202']['id']}").status_code == 404


def test_firm_admin_has_no_matter_content_access(world):
    admin = as_user(world, "admin")
    assert admin.get(f"/api/legal/matters/{world['matters']['M-101']['id']}").status_code == 404
    assert admin.get("/api/legal/clients").json()["clients"] == []


def test_clients_are_visible_only_through_matters(world):
    alice, bob = as_user(world, "alice"), as_user(world, "bob")
    assert [c["name"] for c in alice.get("/api/legal/clients").json()["clients"]] == ["Alpha Holdings"]
    assert [c["name"] for c in bob.get("/api/legal/clients").json()["clients"]] == ["Beta Corp"]
    assert alice.get(f"/api/legal/clients/{world['clients']['Beta']['id']}").status_code == 404


def test_matter_detail_carries_my_role_and_no_hidden_counts(world):
    d = as_user(world, "carol").get(f"/api/legal/matters/{world['matters']['M-101']['id']}").json()["matter"]
    assert d["number"] == "M-101" and d["my_role"] == "read_only" and d["client"]["name"] == "Alpha Holdings"
    assert "member_count" not in d and "matter_count" not in d


def test_other_organisations_are_invisible_even_with_a_stray_membership(world):
    from app.legal.matters import repo
    from app.core.db import get_conn
    other_org = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute("INSERT INTO organizations (id, name, created_at) VALUES (?,?,?)", (other_org, "Other firm", _ts(0)))
        conn.commit()
    c = repo.create_client(other_org, "Gamma", "GAMMA", world["users"]["alice"]["id"])
    stray = repo.create_matter(other_org, c["id"], "X-1", "Other firm matter", world["users"]["alice"]["id"])
    alice = as_user(world, "alice")
    assert "X-1" not in numbers(alice.get("/api/legal/matters"))
    assert alice.get(f"/api/legal/matters/{stray['id']}").status_code == 404


# ── Rights inside a matter ───────────────────────────────────────────────────────
def test_members_without_the_right_get_403_not_404(world):
    carol = as_user(world, "carol")
    mid = world["matters"]["M-101"]["id"]
    assert carol.patch(f"/api/legal/matters/{mid}", json={"title": "x", "version": 1}, headers=csrf_headers(carol)).status_code == 403
    assert carol.post(f"/api/legal/matters/{mid}/memberships", json={"username": "bob", "role": "attorney"},
                      headers=csrf_headers(carol)).status_code == 403


def test_lead_manages_memberships_and_removal_takes_effect_at_once(world, fresh_db):
    alice, bob = as_user(world, "alice"), as_user(world, "bob")
    mid = world["matters"]["M-114"]["id"]
    epoch0 = sqlite3.connect(fresh_db).execute("SELECT acl_epoch FROM matters WHERE id=?", (mid,)).fetchone()[0]
    r = alice.post(f"/api/legal/matters/{mid}/memberships", json={"username": "bob", "role": "attorney"}, headers=csrf_headers(alice))
    assert r.status_code == 200, r.text
    assert "M-114" in numbers(bob.get("/api/legal/matters"))
    members = alice.get(f"/api/legal/matters/{mid}/memberships").json()["memberships"]
    assert {(m["username"], m["role"]) for m in members} == {("alice", "lead"), ("bob", "attorney")}
    r = alice.delete(f"/api/legal/matters/{mid}/memberships/{r.json()['membership']['id']}", headers=csrf_headers(alice))
    assert r.status_code == 200
    assert bob.get(f"/api/legal/matters/{mid}").status_code == 404
    epoch2 = sqlite3.connect(fresh_db).execute("SELECT acl_epoch FROM matters WHERE id=?", (mid,)).fetchone()[0]
    assert epoch2 == epoch0 + 2


def test_role_change_keeps_history(world, fresh_db):
    alice = as_user(world, "alice")
    mid = world["matters"]["M-101"]["id"]
    carol_mm = next(m for m in alice.get(f"/api/legal/matters/{mid}/memberships").json()["memberships"] if m["username"] == "carol")
    r = alice.patch(f"/api/legal/matters/{mid}/memberships/{carol_mm['id']}", json={"role": "paralegal"}, headers=csrf_headers(alice))
    assert r.status_code == 200 and r.json()["membership"]["role"] == "paralegal"
    rows = sqlite3.connect(fresh_db).execute(
        "SELECT role, revoked_at IS NOT NULL, reason FROM matter_memberships WHERE matter_id=? AND user_id=? ORDER BY granted_at",
        (mid, world["users"]["carol"]["id"])).fetchall()
    assert [(r[0], bool(r[1])) for r in rows] == [("read_only", True), ("paralegal", False)]


def test_the_last_lead_cannot_be_removed_or_demoted(world):
    alice = as_user(world, "alice")
    mid = world["matters"]["M-114"]["id"]
    me = alice.get(f"/api/legal/matters/{mid}/memberships").json()["memberships"][0]
    assert alice.delete(f"/api/legal/matters/{mid}/memberships/{me['id']}", headers=csrf_headers(alice)).status_code == 409
    assert alice.patch(f"/api/legal/matters/{mid}/memberships/{me['id']}", json={"role": "attorney"},
                       headers=csrf_headers(alice)).status_code == 409


def test_memberships_require_an_active_colleague(world):
    from app.legal.identity import accounts
    alice = as_user(world, "alice")
    mid = world["matters"]["M-114"]["id"]
    for who in ("nobody", ):
        assert alice.post(f"/api/legal/matters/{mid}/memberships", json={"username": who, "role": "attorney"},
                          headers=csrf_headers(alice)).status_code == 400
    accounts.set_status("dave", "disabled")
    assert alice.post(f"/api/legal/matters/{mid}/memberships", json={"username": "dave", "role": "attorney"},
                      headers=csrf_headers(alice)).status_code == 400


# ── Creating and changing ────────────────────────────────────────────────────────
def test_attorney_creates_client_and_matter_and_becomes_lead(world):
    alice = as_user(world, "alice")
    c = alice.post("/api/legal/clients", json={"name": "Delta Shipping", "code": "DELTA"}, headers=csrf_headers(alice))
    assert c.status_code == 200, c.text
    m = alice.post("/api/legal/matters", json={"client_id": c.json()["client"]["id"], "number": "M-301", "title": "Charter dispute",
                                               "practice_area": "Shipping", "jurisdiction": "England & Wales"}, headers=csrf_headers(alice))
    assert m.status_code == 200, m.text
    assert m.json()["matter"]["my_role"] == "lead" and m.json()["matter"]["sensitivity"] == "restricted"
    dup = alice.post("/api/legal/matters", json={"client_id": c.json()["client"]["id"], "number": "M-301", "title": "Again"},
                     headers=csrf_headers(alice))
    assert dup.status_code == 409


def test_who_may_create(world):
    for name, want in (("carol", 403), ("admin", 403), ("mp", 200)):
        c = as_user(world, name)
        r = c.post("/api/legal/clients", json={"name": f"Client of {name}"}, headers=csrf_headers(c))
        assert r.status_code == want, (name, r.text)


def test_matter_cannot_be_opened_under_an_invisible_client(world):
    alice = as_user(world, "alice")
    r = alice.post("/api/legal/matters", json={"client_id": world["clients"]["Beta"]["id"], "number": "M-999", "title": "x"},
                   headers=csrf_headers(alice))
    assert r.status_code == 404


def test_optimistic_concurrency_on_matter_updates(world):
    alice = as_user(world, "alice")
    mid = world["matters"]["M-101"]["id"]
    r = alice.patch(f"/api/legal/matters/{mid}", json={"title": "MSA renewal 2027", "version": 1}, headers=csrf_headers(alice))
    assert r.status_code == 200 and r.json()["matter"]["version"] == 2
    stale = alice.patch(f"/api/legal/matters/{mid}", json={"title": "Lost update", "version": 1}, headers=csrf_headers(alice))
    assert stale.status_code == 409


def test_closed_matters_only_accept_reopening(world):
    alice = as_user(world, "alice")
    mid = world["matters"]["M-101"]["id"]
    assert alice.patch(f"/api/legal/matters/{mid}", json={"status": "closed", "version": 1}, headers=csrf_headers(alice)).status_code == 200
    assert alice.patch(f"/api/legal/matters/{mid}", json={"title": "x", "version": 2}, headers=csrf_headers(alice)).status_code == 409
    assert alice.patch(f"/api/legal/matters/{mid}", json={"status": "open", "version": 2}, headers=csrf_headers(alice)).status_code == 200


# ── Firm roles (carried over from S1.1: revoke on role change) ───────────────────
def test_firm_admin_manages_roles_and_removal_signs_the_user_out(world):
    admin, alice = as_user(world, "admin"), as_user(world, "alice")
    listed = {u["username"]: u for u in admin.get("/api/legal/users").json()["users"]}
    assert listed["alice"]["roles"] == ["attorney"] and listed["alice"]["mfa_enabled"] is True
    r = admin.put("/api/legal/users/alice/roles", json={"roles": ["attorney", "reviewer"]}, headers=csrf_headers(admin))
    assert r.status_code == 200 and alice.get("/api/auth/me").status_code == 200        # added a role: stays signed in
    r = admin.put("/api/legal/users/alice/roles", json={"roles": ["reviewer"]}, headers=csrf_headers(admin))
    assert r.status_code == 200 and r.json()["sessions_revoked"] >= 1
    assert alice.get("/api/auth/me").status_code == 401                                 # removed a role: signed out


def test_role_management_is_firm_admin_only_and_cannot_lock_out_the_admin(world):
    alice, admin = as_user(world, "alice"), as_user(world, "admin")
    assert alice.get("/api/legal/users").status_code == 403
    assert as_user(world, "mp").get("/api/legal/users").status_code == 200          # walls authority may list
    assert as_user(world, "mp").put("/api/legal/users/alice/roles", json={"roles": ["attorney"]},
                                    headers=csrf_headers(as_user(world, "mp"))).status_code == 403
    assert alice.put("/api/legal/users/alice/roles", json={"roles": ["managing_partner"]}, headers=csrf_headers(alice)).status_code == 403
    assert admin.put("/api/legal/users/admin2/roles", json={"roles": []}, headers=csrf_headers(admin)).status_code == 400
    assert admin.put("/api/legal/users/alice/roles", json={"roles": ["wizard"]}, headers=csrf_headers(admin)).status_code == 422


# ── Commercial-readiness review F6 (2026-10-28): two people for the firm's highest power ─────────────────────────────
def test_an_administrator_cannot_make_managing_partners_once_the_firm_has_one(world):
    """Two managing partners approve walls and emergency access; an administrator who could make them could approve their
    own requests. Only a managing partner appoints another (never themselves); the very first may come from an administrator."""
    admin, mp = as_user(world, "admin"), as_user(world, "mp")
    r = admin.put("/api/legal/users/carol/roles", json={"roles": ["paralegal", "managing_partner"]}, headers=csrf_headers(admin))
    assert r.status_code == 403 and "managing partner" in r.json()["detail"]
    r = admin.post("/api/auth/users", headers=csrf_headers(admin),
                   json={"username": "puppet", "password": "A long enough pass phrase 42!", "role": "managing_partner"})
    assert r.status_code == 403
    assert mp.post("/api/legal/users/mp/appoint-managing-partner", headers=csrf_headers(mp)).status_code == 400   # not self
    assert as_user(world, "alice").post("/api/legal/users/carol/appoint-managing-partner",
                                        headers=csrf_headers(as_user(world, "alice"))).status_code == 403
    r = mp.post("/api/legal/users/carol/appoint-managing-partner", headers=csrf_headers(mp))
    assert r.status_code == 200 and "managing_partner" in r.json()["user"]["roles"]
    r = admin.put("/api/legal/users/carol/roles", json={"roles": ["paralegal"]}, headers=csrf_headers(admin))
    assert r.status_code == 200                                                    # taking the power away needs no second person


def test_the_first_managing_partner_may_come_from_an_administrator(world, fresh_db):
    import sqlite3
    with sqlite3.connect(fresh_db) as c:
        c.execute("DELETE FROM user_roles WHERE role='managing_partner'")
    admin = as_user(world, "admin")
    r = admin.put("/api/legal/users/carol/roles", json={"roles": ["managing_partner"]}, headers=csrf_headers(admin))
    assert r.status_code == 200, r.text
