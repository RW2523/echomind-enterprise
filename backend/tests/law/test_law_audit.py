"""S1.5 / SEC-04 — the hash-chained, append-only audit trail."""
from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

from conftest import as_user, csrf_headers


def _events(db, **where):
    q = "SELECT action, outcome, actor_username, matter_id, object_id, reason, details_json FROM audit_events"
    if where:
        q += " WHERE " + " AND ".join(f"{k}=?" for k in where)
    with sqlite3.connect(db) as c:
        return [dict(zip(("action", "outcome", "actor", "matter_id", "object_id", "reason", "details"), r))
                for r in c.execute(q + " ORDER BY seq", tuple(where.values()))]


def _actions(db):
    return [(e["action"], e["outcome"]) for e in _events(db)]


# ── The chain itself ─────────────────────────────────────────────────────────────
def test_events_chain_and_verify(law_app, fresh_db):
    from app.legal.audit import chain, record
    for i in range(3):
        record.system_event("test.event", details={"n": i})
    v = chain.verify()
    assert v["ok"] and v["events"] == 3 and len(v["head"]) == 64


def test_the_database_refuses_edits_and_deletions(law_app, fresh_db):
    from app.legal.audit import record
    record.system_event("test.event")
    with sqlite3.connect(fresh_db) as c:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            c.execute("UPDATE audit_events SET action='forged'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            c.execute("DELETE FROM audit_events")


def test_tampering_with_direct_database_access_is_detected(law_app, fresh_db):
    from app.legal.audit import chain, record
    for i in range(4):
        record.system_event("test.event", details={"n": i})
    with sqlite3.connect(fresh_db) as c:              # an attacker with file access drops the trigger
        c.execute("DROP TRIGGER audit_no_update")
        c.execute("UPDATE audit_events SET outcome='allow', action='innocent' WHERE seq=2")
    v = chain.verify()
    assert v["ok"] is False and v["broken_at"] == 2


def test_details_never_carry_free_text(law_app, fresh_db):
    from app.legal.audit import record
    record.system_event("test.event", details={"title": "Board investigation of Beta Shipping", "count": 3,
                                               "fields": ["title", "status"], "code": "M-202"})
    d = json.loads(_events(fresh_db)[-1]["details"])
    assert d == {"title": {"chars": 36}, "count": 3, "fields": ["title", "status"], "code": "M-202"}


def test_a_daily_anchor_is_written_next_to_the_data(law_app, fresh_db):
    import os
    import stat
    from app.core.config import settings
    from app.legal.audit import record
    record.system_event("test.event")
    d = os.path.join(settings.DATA_DIR, "legal", "audit_anchor")
    files = os.listdir(d)
    assert len(files) == 1 and stat.S_IMODE(os.stat(os.path.join(d, files[0])).st_mode) == 0o600


# ── What gets recorded ───────────────────────────────────────────────────────────
def test_sign_in_events_are_recorded_without_secrets(world, fresh_db):
    from fastapi.testclient import TestClient
    c = TestClient(world["app"], base_url="https://testserver")
    c.post("/api/auth/login", json={"username": "alice", "password": "wrong password typed here"})
    c.post("/api/auth/login", json={"username": "no.such.person", "password": "x" * 12})
    as_user(world, "bob")
    ev = _events(fresh_db)
    assert ("auth.login", "deny") in [(e["action"], e["outcome"]) for e in ev if e["actor"] == "alice"]
    unknown = [e for e in ev if e["action"] == "auth.login" and e["reason"] == "unknown_user"]
    assert unknown and unknown[0]["actor"] is None
    assert ("auth.mfa.verify", "allow") in [(e["action"], e["outcome"]) for e in ev if e["actor"] == "bob"]
    blob = json.dumps(ev)
    assert "wrong password typed here" not in blob and "no.such.person" not in blob


def test_refusals_are_recorded_but_the_answer_does_not_change(world, fresh_db):
    alice = as_user(world, "alice")
    hidden = world["matters"]["M-202"]["id"]
    r = alice.get(f"/api/legal/matters/{hidden}")
    assert r.json() == {"detail": "Not Found"}
    denies = [e for e in _events(fresh_db, action="matter.read", outcome="deny") if e["actor"] == "alice"]
    assert denies and denies[-1]["matter_id"] == hidden and denies[-1]["reason"] == "not_found"


def test_changes_to_matters_members_walls_and_roles_are_recorded(world, fresh_db):
    alice, mp, mp2, admin = (as_user(world, n) for n in ("alice", "mp", "mp2", "admin"))
    m = alice.post("/api/legal/matters", headers=csrf_headers(alice), json={
        "client_id": world["clients"]["Alpha"]["id"], "number": "M-777", "title": "ZEPHYR-CANARY confidential deal"}).json()["matter"]
    alice.post(f"/api/legal/matters/{m['id']}/memberships", headers=csrf_headers(alice), json={"username": "carol", "role": "paralegal"})
    alice.patch(f"/api/legal/matters/{m['id']}", headers=csrf_headers(alice), json={"status": "closed", "version": 1})
    w = mp.post("/api/legal/walls", headers=csrf_headers(mp), json={"name": "ZEPHYR-CANARY wall", "rationale": "secret reason",
                "screened": ["bob"], "protected_matters": [m["id"]], "protected_clients": []}).json()["wall"]
    mp2.post(f"/api/legal/walls/{w['id']}/approve", headers=csrf_headers(mp2))
    admin.put("/api/legal/users/carol/roles", headers=csrf_headers(admin), json={"roles": ["reviewer"]})
    acts = _actions(fresh_db)
    for expected in [("matter.create", "allow"), ("membership.add", "allow"), ("matter.update", "allow"),
                     ("wall.request", "allow"), ("wall.approve", "allow"), ("roles.set", "allow")]:
        assert expected in acts, expected
    assert "ZEPHYR-CANARY" not in json.dumps(_events(fresh_db)) and "secret reason" not in json.dumps(_events(fresh_db))


def test_every_break_glass_access_is_recorded(world, fresh_db):
    admin, mp = as_user(world, "admin"), as_user(world, "mp")
    g = admin.post("/api/legal/break-glass", headers=csrf_headers(admin), json={"matter_number": "M-101", "reason": "x", "minutes": 30}).json()["request"]
    mp.post(f"/api/legal/break-glass/{g['id']}/approve", headers=csrf_headers(mp))
    admin.get(f"/api/legal/matters/{world['matters']['M-101']['id']}")
    reads = [e for e in _events(fresh_db, action="matter.read", outcome="allow") if e["actor"] == "admin2"]
    assert reads and reads[-1]["reason"] == "break_glass"
    assert ("breakglass.approve", "allow") in _actions(fresh_db)


# ── Reading the trail ────────────────────────────────────────────────────────────
def test_org_audit_is_for_governance_roles_and_hides_matters_from_firm_admins(world):
    alice = as_user(world, "alice")
    alice.get(f"/api/legal/matters/{world['matters']['M-101']['id']}")
    assert alice.get("/api/legal/audit").status_code == 403
    rows = as_user(world, "admin").get("/api/legal/audit?action=matter.read").json()["events"]
    assert rows and all(r["matter"] is None or r["matter"].get("number") is None for r in rows)
    rows = as_user(world, "mp").get("/api/legal/audit?action=matter.read").json()["events"]
    assert any((r["matter"] or {}).get("number") == "M-101" for r in rows)


def test_matter_activity_is_for_the_lead(world):
    mid = world["matters"]["M-101"]["id"]
    alice, carol, bob = as_user(world, "alice"), as_user(world, "carol"), as_user(world, "bob")
    alice.get(f"/api/legal/matters/{mid}")
    r = alice.get(f"/api/legal/matters/{mid}/activity")
    assert r.status_code == 200 and r.json()["events"]
    assert {(e["matter"] or {}).get("number") for e in r.json()["events"]} == {"M-101"}
    assert carol.get(f"/api/legal/matters/{mid}/activity").status_code == 403         # member, not lead
    assert bob.get(f"/api/legal/matters/{mid}/activity").status_code == 404            # not a member


def test_verify_and_export_endpoints(world):
    admin = as_user(world, "admin")
    v = admin.get("/api/legal/audit/verify").json()
    assert v["ok"] is True and v["events"] > 0
    r = admin.get("/api/legal/audit/export?format=jsonl")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(x) for x in r.text.strip().splitlines()]
    assert lines and {"seq", "hash", "prev_hash", "action"} <= set(lines[0])
    assert admin.get("/api/legal/audit/export?format=csv").text.startswith("seq,ts,")
    assert ("audit.export", "allow") in [(e["action"], e["outcome"]) for e in admin.get("/api/legal/audit?action=audit.export").json()["events"]]


# ── Completeness and failure behaviour ───────────────────────────────────────────
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


def test_every_law_state_changing_route_writes_a_named_event(law_app):
    """Structural guarantee: a new Law route without @audited fails here, not in an incident review."""
    from fastapi.routing import APIRoute
    missing = [f"{sorted(r.methods)} {r.path}" for r in law_app.routes
               if isinstance(r, APIRoute) and r.methods & _MUTATING
               and (r.path.startswith("/api/legal/") or r.path.startswith("/api/auth/"))
               and not getattr(r.endpoint, "__audit_action__", None)]
    assert not missing, missing


def test_one_action_one_event(world, fresh_db):
    alice, carol = as_user(world, "alice"), as_user(world, "carol")
    mid = world["matters"]["M-101"]["id"]

    def count():
        with sqlite3.connect(fresh_db) as c:
            return c.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]

    n = count()
    alice.patch(f"/api/legal/matters/{mid}", headers=csrf_headers(alice), json={"jurisdiction": "England", "version": 1})
    assert count() == n + 1
    r = carol.patch(f"/api/legal/matters/{mid}", headers=csrf_headers(carol), json={"title": "x", "version": 2})
    assert r.status_code == 403 and count() == n + 2
    last = _events(fresh_db)[-1]
    assert (last["action"], last["outcome"], last["reason"], last["matter_id"]) == ("matter.update", "deny", "forbidden", mid)


def test_invalid_input_and_csrf_failures_are_recorded(world, fresh_db):
    alice = as_user(world, "alice")
    r = alice.post("/api/legal/matters", headers=csrf_headers(alice), json={"number": "M-9"})     # 422 before the route runs
    assert r.status_code == 422
    assert _events(fresh_db)[-1]["action"] == "matter.create" and _events(fresh_db)[-1]["outcome"] == "error"
    r = alice.post("/api/legal/matters", json={"number": "M-9"})                                   # no CSRF header
    assert r.status_code == 403
    last = _events(fresh_db)[-1]
    assert (last["action"], last["outcome"], last["reason"], last["actor"]) == ("request.refused", "deny", "csrf_failed", "alice")


def test_other_state_changing_requests_get_a_generic_event(world, fresh_db):
    alice = as_user(world, "alice")
    r = alice.post("/api/chat/create", headers=csrf_headers(alice), json={"title": "ZEPHYR-CANARY chat"})
    assert r.status_code == 200
    generic = [e for e in _events(fresh_db) if e["action"] == "api.request"]
    assert generic and json.loads(generic[-1]["details"]) == {"method": "POST", "route": "/api/chat/create", "status": 200}
    assert "ZEPHYR-CANARY" not in json.dumps(_events(fresh_db))


def test_break_glass_reads_on_any_route_are_recorded(world, fresh_db):
    admin, mp = as_user(world, "admin"), as_user(world, "mp")
    g = admin.post("/api/legal/break-glass", headers=csrf_headers(admin), json={"matter_number": "M-114", "reason": "x", "minutes": 30}).json()["request"]
    mp.post(f"/api/legal/break-glass/{g['id']}/approve", headers=csrf_headers(mp))
    mid = world["matters"]["M-114"]["id"]
    assert admin.get(f"/api/legal/matters/{mid}/memberships").status_code == 200
    ev = [e for e in _events(fresh_db) if e["actor"] == "admin2" and e["matter_id"] == mid and e["outcome"] == "allow"]
    assert any(e["reason"] == "break_glass" for e in ev)


def test_when_the_trail_cannot_be_written_nothing_is_returned(world, monkeypatch):
    alice = as_user(world, "alice")
    from app.legal.audit import chain

    def broken(fields):
        raise OSError("disk full")
    monkeypatch.setattr(chain, "append", broken)
    r = alice.get(f"/api/legal/matters/{world['matters']['M-101']['id']}")
    assert r.status_code == 500 and "MSA" not in r.text


def test_a_removed_tail_is_caught_by_the_anchor(law_app, fresh_db):
    from app.legal.audit import chain, record
    for i in range(3):
        record.system_event("test.event")
    chain.anchor_now()
    with sqlite3.connect(fresh_db) as c:
        c.execute("DROP TRIGGER audit_no_delete")
        c.execute("DELETE FROM audit_events WHERE seq = (SELECT MAX(seq) FROM audit_events)")
    v = chain.verify()
    assert v["ok"] is False and v["problem"] == "events_missing_after_anchor"


def test_a_rewritten_chain_is_caught_by_the_anchor(law_app, fresh_db):
    """Edit an event and re-hash everything after it: the chain is self-consistent, the anchor is not."""
    from app.legal.audit import chain, record
    for i in range(3):
        record.system_event("test.event")
    chain.anchor_now()
    with sqlite3.connect(fresh_db) as c:
        c.execute("DROP TRIGGER audit_no_update")
        rows = [dict(zip(chain.COLUMNS, r)) for r in c.execute(f"SELECT {', '.join(chain.COLUMNS)} FROM audit_events ORDER BY seq")]
        prev = chain.GENESIS
        for r in rows:
            if r["seq"] == 2:
                r["action"] = "innocent"
            h = chain.digest(prev, r)
            c.execute("UPDATE audit_events SET action=?, prev_hash=?, hash=? WHERE seq=?", (r["action"], prev, h, r["seq"]))
            prev = h
    v = chain.verify()
    assert v["ok"] is False and v["problem"] == "anchor_mismatch"


def test_the_verifier_cli(law_app, fresh_db, capsys):
    from app.legal.audit import record
    from app.legal.audit.__main__ import main
    record.system_event("test.event")
    assert main(["verify"]) == 0
    with sqlite3.connect(fresh_db) as c:
        c.execute("DROP TRIGGER audit_no_update")
        c.execute("UPDATE audit_events SET reason='x'")
    assert main(["verify"]) == 1


def test_canary_text_reaches_neither_the_trail_nor_the_logs(world, fresh_db, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    alice, mp = as_user(world, "alice"), as_user(world, "mp")
    canary = "QUOKKA-7731 privileged settlement terms"
    m = alice.post("/api/legal/matters", headers=csrf_headers(alice), json={
        "client_id": world["clients"]["Alpha"]["id"], "number": "M-778", "title": canary}).json()["matter"]
    mp.post("/api/legal/walls", headers=csrf_headers(mp), json={"name": canary, "rationale": canary, "screened": ["bob"],
                                                                "protected_matters": [m["id"]], "protected_clients": []})
    as_user(world, "admin").post("/api/legal/break-glass", headers=csrf_headers(as_user(world, "admin")),
                                 json={"matter_number": "M-101", "reason": canary, "minutes": 10})
    assert "QUOKKA" not in json.dumps(_events(fresh_db))
    assert "QUOKKA" not in caplog.text
