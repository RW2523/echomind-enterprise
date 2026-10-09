"""S1.4 / SEC-02 — ethical walls, exceptions and break-glass.

Walls bind everyone, including a matter's lead; they take effect only after a second managing
partner approves; a walled matter answers exactly like a missing one. Break-glass gives a firm
admin time-limited read-only access, approved by someone else, never across a wall and never to a
highly restricted matter.
"""
from __future__ import annotations

import time

import pytest

from conftest import _ts, as_user, csrf_headers, numbers


def _wall(world, requester="mp", screened=("alice",), matters=(), clients=(), approve_by="mp2"):
    c = as_user(world, requester)
    r = c.post("/api/legal/walls", headers=csrf_headers(c), json={
        "name": "Conflict screen", "rationale": "Alice previously advised the counterparty.",
        "screened": list(screened), "protected_matters": [world["matters"][m]["id"] for m in matters],
        "protected_clients": [world["clients"][k]["id"] for k in clients]})
    assert r.status_code == 200, r.text
    wall = r.json()["wall"]
    if approve_by:
        a = as_user(world, approve_by)
        ok = a.post(f"/api/legal/walls/{wall['id']}/approve", headers=csrf_headers(a))
        assert ok.status_code == 200, ok.text
        wall = ok.json()["wall"]
    return wall


def _mid(world, number):
    return world["matters"][number]["id"]


# ── Walls ────────────────────────────────────────────────────────────────────────
def test_an_approved_wall_hides_a_matter_even_from_its_lead(world):
    alice = as_user(world, "alice")
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 200
    _wall(world, matters=["M-101"])
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}").json() == {"detail": "Not Found"}
    assert numbers(alice.get("/api/legal/matters")) == {"M-114"}
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}/memberships").status_code == 404


def test_a_client_wall_covers_every_matter_of_that_client(world):
    alice = as_user(world, "alice")
    _wall(world, clients=["Alpha"])
    assert numbers(alice.get("/api/legal/matters")) == set()
    assert alice.get("/api/legal/clients").json()["clients"] == []
    carol = as_user(world, "carol")                                   # not screened
    assert numbers(carol.get("/api/legal/matters")) == {"M-101"}


def test_a_wall_needs_a_second_managing_partner(world):
    wall = _wall(world, matters=["M-101"], approve_by=None)
    alice = as_user(world, "alice")
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 200   # requested only
    mp = as_user(world, "mp")
    assert mp.post(f"/api/legal/walls/{wall['id']}/approve", headers=csrf_headers(mp)).status_code == 403
    assert as_user(world, "alice").post(f"/api/legal/walls/{wall['id']}/approve",
                                        headers=csrf_headers(alice)).status_code == 403
    mp2 = as_user(world, "mp2")
    assert mp2.post(f"/api/legal/walls/{wall['id']}/approve", headers=csrf_headers(mp2)).json()["wall"]["status"] == "active"
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 404


def test_releasing_a_wall_also_needs_two_people(world):
    wall = _wall(world, matters=["M-101"])
    mp, mp2, alice = as_user(world, "mp"), as_user(world, "mp2"), as_user(world, "alice")
    assert mp.post(f"/api/legal/walls/{wall['id']}/release", headers=csrf_headers(mp)).status_code == 200
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 404     # still holds
    assert mp.post(f"/api/legal/walls/{wall['id']}/release/approve", headers=csrf_headers(mp)).status_code == 403
    assert mp2.post(f"/api/legal/walls/{wall['id']}/release/approve", headers=csrf_headers(mp2)).status_code == 200
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 200


def test_an_approved_unexpired_exception_lets_one_person_in(world):
    wall = _wall(world, matters=["M-101"])
    mp, mp2, alice = as_user(world, "mp"), as_user(world, "mp2"), as_user(world, "alice")
    r = mp.post(f"/api/legal/walls/{wall['id']}/exceptions", headers=csrf_headers(mp), json={
        "username": "alice", "matter_id": _mid(world, "M-101"), "reason": "Needed for hand-over", "expires_at": _ts(2)})
    assert r.status_code == 200, r.text
    exc = r.json()["exception"]
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 404     # not yet approved
    assert mp.post(f"/api/legal/walls/{wall['id']}/exceptions/{exc['id']}/approve", headers=csrf_headers(mp)).status_code == 403
    assert mp2.post(f"/api/legal/walls/{wall['id']}/exceptions/{exc['id']}/approve", headers=csrf_headers(mp2)).status_code == 200
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 200
    assert mp2.delete(f"/api/legal/walls/{wall['id']}/exceptions/{exc['id']}", headers=csrf_headers(mp2)).status_code == 200
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 404


def test_an_expired_exception_grants_nothing(world, fresh_db):
    import sqlite3
    wall = _wall(world, matters=["M-101"])
    mp, mp2, alice = as_user(world, "mp"), as_user(world, "mp2"), as_user(world, "alice")
    exc = mp.post(f"/api/legal/walls/{wall['id']}/exceptions", headers=csrf_headers(mp), json={
        "username": "alice", "matter_id": _mid(world, "M-101"), "reason": "x", "expires_at": _ts(1)}).json()["exception"]
    mp2.post(f"/api/legal/walls/{wall['id']}/exceptions/{exc['id']}/approve", headers=csrf_headers(mp2))
    with sqlite3.connect(fresh_db) as c:
        c.execute("UPDATE wall_exceptions SET expires_at=? WHERE id=?", (_ts(-1), exc["id"]))
    assert alice.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 404


def test_exception_expiry_is_bounded(world):
    wall = _wall(world, matters=["M-101"])
    mp = as_user(world, "mp")
    r = mp.post(f"/api/legal/walls/{wall['id']}/exceptions", headers=csrf_headers(mp), json={
        "username": "alice", "matter_id": _mid(world, "M-101"), "reason": "x", "expires_at": _ts(400)})
    assert r.status_code == 422


def test_walls_are_visible_to_managing_partners_only(world):
    _wall(world, matters=["M-101"])
    for name in ("alice", "admin", "carol"):
        assert as_user(world, name).get("/api/legal/walls").status_code == 403
    walls = as_user(world, "mp2").get("/api/legal/walls").json()["walls"]
    assert walls[0]["screened"] == ["alice"] and walls[0]["protected"][0]["label"].startswith("M-101")


def test_an_attorney_can_only_protect_matters_they_can_see(world):
    bob = as_user(world, "bob")
    r = bob.post("/api/legal/walls", headers=csrf_headers(bob), json={
        "name": "x", "rationale": "y", "screened": ["carol"], "protected_matters": [_mid(world, "M-101")], "protected_clients": []})
    assert r.status_code == 404


# ── Break-glass ──────────────────────────────────────────────────────────────────
def _break_glass(world, number="M-101", minutes=60, requester="admin", approver="mp"):
    req = as_user(world, requester)
    r = req.post("/api/legal/break-glass", headers=csrf_headers(req),
                 json={"matter_number": number, "reason": "Lead unreachable; regulator deadline today", "minutes": minutes})
    assert r.status_code == 200, r.text
    g = r.json()["request"]
    if approver:
        a = as_user(world, approver)
        return g, a.post(f"/api/legal/break-glass/{g['id']}/approve", headers=csrf_headers(a))
    return g, None


def test_break_glass_gives_time_limited_read_only_access(world):
    admin = as_user(world, "admin")
    assert admin.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 404
    _, ok = _break_glass(world)
    assert ok.status_code == 200, ok.text
    d = admin.get(f"/api/legal/matters/{_mid(world, 'M-101')}").json()["matter"]
    assert d["my_role"] == "break_glass"
    assert "M-101" in numbers(admin.get("/api/legal/matters"))
    assert admin.get(f"/api/legal/matters/{_mid(world, 'M-101')}/memberships").status_code == 200
    assert admin.patch(f"/api/legal/matters/{_mid(world, 'M-101')}", json={"title": "x", "version": 1},
                       headers=csrf_headers(admin)).status_code == 403
    assert admin.post(f"/api/legal/matters/{_mid(world, 'M-101')}/memberships", json={"username": "admin2", "role": "lead"},
                      headers=csrf_headers(admin)).status_code == 403


def test_break_glass_cannot_be_self_approved_or_used_after_it_ends(world):
    g, _ = _break_glass(world, approver=None)
    admin = as_user(world, "admin")
    assert admin.post(f"/api/legal/break-glass/{g['id']}/approve", headers=csrf_headers(admin)).status_code == 403
    mp = as_user(world, "mp")
    assert mp.post(f"/api/legal/break-glass/{g['id']}/approve", headers=csrf_headers(mp)).status_code == 200
    assert admin.post(f"/api/legal/break-glass/{g['id']}/end", headers=csrf_headers(admin)).status_code == 200
    assert admin.get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 404


def test_break_glass_expires(world, fresh_db):
    import sqlite3
    g, ok = _break_glass(world)
    with sqlite3.connect(fresh_db) as c:
        c.execute("UPDATE break_glass_grants SET expires_at=? WHERE id=?", (_ts(-1), g["id"]))
    assert as_user(world, "admin").get(f"/api/legal/matters/{_mid(world, 'M-101')}").status_code == 404


def test_break_glass_never_reaches_highly_restricted_matters(world):
    _, r = _break_glass(world, number="M-202")
    assert r.status_code == 409 and "highly restricted" in r.json()["detail"]


def test_break_glass_never_crosses_a_wall(world):
    _wall(world, screened=["admin2"], matters=["M-101"])
    _, r = _break_glass(world)
    assert r.status_code == 409 and "wall" in r.json()["detail"]


def test_break_glass_duration_is_capped(world):
    admin = as_user(world, "admin")
    r = admin.post("/api/legal/break-glass", headers=csrf_headers(admin),
                   json={"matter_number": "M-101", "reason": "x", "minutes": 600})
    assert r.status_code == 422


def test_requesting_break_glass_reveals_nothing_about_the_number(world):
    admin = as_user(world, "admin")
    def req(n):
        r = admin.post("/api/legal/break-glass", headers=csrf_headers(admin), json={"matter_number": n, "reason": "x", "minutes": 30})
        return r.status_code, sorted(r.json()["request"].keys()), r.json()["request"]["status"]
    assert req("M-101") == req("NO-SUCH-MATTER")


def test_only_firm_admins_request_and_managing_partners_approve_break_glass(world):
    alice = as_user(world, "alice")
    assert alice.post("/api/legal/break-glass", headers=csrf_headers(alice),
                      json={"matter_number": "M-202", "reason": "x", "minutes": 30}).status_code == 403
    g, _ = _break_glass(world, approver=None)
    assert alice.post(f"/api/legal/break-glass/{g['id']}/approve", headers=csrf_headers(alice)).status_code == 403


def test_policy_still_answers_hidden_and_missing_with_the_same_query_shape(world, monkeypatch):
    from contextlib import contextmanager
    import uuid
    from app.legal import policy
    _wall(world, matters=["M-114"])
    statements = []
    real = policy.get_conn

    @contextmanager
    def traced():
        with real() as conn:
            conn.set_trace_callback(statements.append)
            yield conn

    monkeypatch.setattr(policy, "get_conn", traced)
    p = {"sub": world["users"]["alice"]["id"], "org_id": world["matters"]["M-114"]["org_id"]}
    for target in (_mid(world, "M-114"), _mid(world, "M-202"), str(uuid.uuid4())):   # walled, not member, missing
        statements.clear()
        assert policy.authorize(p, "matter.read", target).reason == "not_found"
        assert len([s for s in statements if not s.startswith("PRAGMA")]) == 2
