"""One-click demo sign-in (D-077): the owner asked to enter the demo without passwords or codes.

Off unless LEGAL_DEMO_SIGNIN. Only the fictional people of the demo seed (the names in legal/demo_credentials.json) can be
entered this way; the administrator and any account added later keep their password and code. Each entry is a full
session, audited as auth.demo_signin. By default only from this server or private addresses
(LEGAL_DEMO_SIGNIN_FROM=private); the owner may choose "anywhere". The pages say it is on.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from conftest import LAW_TEST_ORIGIN

PUBLIC, PRIVATE = {"X-Real-IP": "203.0.113.9"}, {"X-Real-IP": "10.0.0.42"}


@pytest.fixture
def demo(world, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_DEMO_SIGNIN", True)
    monkeypatch.setattr(settings, "LEGAL_DEMO_SIGNIN_FROM", "private")
    monkeypatch.setattr(settings, "LEGAL_TRUSTED_PROXIES", "testclient")
    folder = os.path.join(settings.DATA_DIR, "legal")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "demo_credentials.json"), "w") as f:     # the demo seed's people (fictional)
        json.dump({n: {"password": "x", "totp_secret": "y"} for n in ("alice", "bob", "carol")}, f)
    return world


def browser(w, headers=PRIVATE):
    from fastapi.testclient import TestClient
    c = TestClient(w["app"], base_url=LAW_TEST_ORIGIN)
    c.headers.update(headers)
    return c


def test_off_unless_the_owner_turns_it_on(world):
    c = browser(world)
    assert "demo_signin" not in c.get("/api/auth/config").json()
    assert c.get("/api/auth/demo").status_code == 404
    assert c.post("/api/auth/demo/alice").status_code == 404


def test_the_page_lists_the_demo_people_and_one_click_signs_in_fully(demo, fresh_db):
    c = browser(demo)
    assert c.get("/api/auth/config").json()["demo_signin"] is True
    people = c.get("/api/auth/demo").json()["people"]
    assert [p["username"] for p in people] == ["alice", "bob", "carol"]
    assert {p["username"]: p["roles"] for p in people}["carol"] == ["paralegal"]
    r = c.post("/api/auth/demo/alice")
    assert r.status_code == 200 and r.json()["user"]["username"] == "alice"
    assert c.get("/api/legal/matters").status_code == 200                       # a full session: no code asked
    me = c.get("/api/auth/me").json()
    assert me["user"]["username"] == "alice" and me["features"]["demo_signin"] is True
    with sqlite3.connect(fresh_db) as db:
        rows = db.execute("SELECT outcome, actor_username, ip FROM audit_events WHERE action='auth.demo_signin'").fetchall()
    assert rows == [("allow", "alice", "10.0.0.42")]


def test_only_the_demo_people_never_the_administrator_or_anyone_added_later(demo):
    c = browser(demo)
    for name in ("admin2", "mp", "nobody"):
        assert c.post(f"/api/auth/demo/{name}").status_code == 404
    assert c.get("/api/legal/matters").status_code == 401


def test_a_disabled_demo_person_cannot_be_entered(demo):
    from app.legal.identity import accounts
    accounts.set_status("bob", "disabled")
    assert browser(demo).post("/api/auth/demo/bob").status_code == 403


def test_by_default_only_from_this_server_or_a_private_address(demo, monkeypatch):
    from app.core.config import settings
    outside = browser(demo, PUBLIC)
    assert "demo_signin" not in outside.get("/api/auth/config").json()
    assert outside.get("/api/auth/demo").status_code == 404
    assert outside.post("/api/auth/demo/alice").status_code == 404
    monkeypatch.setattr(settings, "LEGAL_DEMO_SIGNIN_FROM", "anywhere")          # the owner's choice for the pilot
    assert outside.post("/api/auth/demo/alice").status_code == 200


def test_the_owner_s_open_demo_names_the_person_the_page_enters_as(demo, monkeypatch):
    """LEGAL_DEMO_AUTO: the sign-in page enters as this demo person by itself. Only a demo person, only when demo sign-in is
    available to this caller; the entry itself is the same audited demo session."""
    from app.core.config import settings
    c = browser(demo)
    assert "auto" not in c.get("/api/auth/demo").json()                          # off by default: the page lists the people
    monkeypatch.setattr(settings, "LEGAL_DEMO_AUTO", "alice")
    assert c.get("/api/auth/demo").json()["auto"] == "alice"
    monkeypatch.setattr(settings, "LEGAL_DEMO_AUTO", "admin")                     # never the administrator or a real account
    assert "auto" not in c.get("/api/auth/demo").json()
    monkeypatch.setattr(settings, "LEGAL_DEMO_AUTO", "alice")
    assert browser(demo, PUBLIC).get("/api/auth/demo").status_code == 404        # not from a public address unless "anywhere"
    monkeypatch.setattr(settings, "LEGAL_DEMO_SIGNIN", False)
    assert c.get("/api/auth/demo").status_code == 404                             # nothing without demo sign-in



# ── Commercial-readiness review, 2026-10-28 (F1, F10) ─────────────────────────────────────────────────────────────────
def _seed_names(names):
    from app.core.config import settings
    with open(os.path.join(settings.DATA_DIR, "legal", "demo_credentials.json"), "w") as f:
        json.dump({n: {"password": "x", "totp_secret": "y"} for n in names}, f)


def test_no_one_with_a_firm_wide_power_is_one_click_away(demo):
    """Administration, records and the managing partners (who make up the two-person rules) keep their password and code."""
    _seed_names(["alice", "bob", "carol", "mp", "admin2"])
    c = browser(demo)
    assert [p["username"] for p in c.get("/api/auth/demo").json()["people"]] == ["alice", "bob", "carol"]
    for name in ("mp", "admin2"):
        assert c.post(f"/api/auth/demo/{name}").status_code == 404
    from app.legal.identity import accounts
    accounts.grant_role(demo["users"]["bob"]["id"], "records_admin", granted_by="test")     # a power granted later counts too
    assert "bob" not in [p["username"] for p in c.get("/api/auth/demo").json()["people"]]
    assert c.post("/api/auth/demo/bob").status_code == 404


def test_another_site_cannot_sign_a_visitor_in(demo):
    c = browser(demo)
    assert c.post("/api/auth/demo/alice", headers={"Origin": "https://evil.example"}).status_code == 403
    assert c.get("/api/legal/matters").status_code == 401
    assert c.post("/api/auth/demo/alice", headers={"Origin": LAW_TEST_ORIGIN}).status_code == 200


def test_turning_demo_sign_in_off_ends_the_sessions_it_made(demo, monkeypatch):
    from app.core.config import settings
    from app.legal.identity import demo as demo_mod
    c = browser(demo)
    assert c.post("/api/auth/demo/alice").status_code == 200 and c.get("/api/legal/matters").status_code == 200
    assert demo_mod.end_leftover_sessions() == 0                                     # still on: nothing ends
    monkeypatch.setattr(settings, "LEGAL_DEMO_SIGNIN", False)
    assert demo_mod.end_leftover_sessions() >= 1
    assert c.get("/api/legal/matters").status_code == 401
