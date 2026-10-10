"""S4.1 — intake assistant with conflict screening (OPT-01; decision D-054).

Off unless the firm turns it on (LEGAL_INTAKE; enquirer links also need LEGAL_INTAKE_LINKS). The intake
is administrative and scripted: no model reads or writes anything. Before any question, the enquirer
accepts the disclaimer. Names given are screened against the firm's clients and the parties on its
matters:
- front-desk staff and the enquirer learn only "possible conflict: refer to the conflicts team";
- the conflicts team sees the matches and decides.

A call is recorded only after the caller's consent; withdrawing consent removes the recording. Replies
come from fixed templates and are approved by a lawyer who did not write them, and a reply may not
name another client or party of the firm. A cleared intake can become a client and matter. A records
administrator can purge it.
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from conftest import LAW_TEST_ORIGIN, as_user, csrf_headers, make_user

ANSWERS = {"your_name": "Hana Kowalczyk", "organisation": "Kestrel Audio GmbH", "email": "hana@kestrel.example",
           "phone": "", "matter_type": "commercial", "other_parties": ["Pinewood Retail Ltd"],
           "summary": "We want to review a distribution agreement before renewal.", "deadline": "", "heard_from": "referral"}


@pytest.fixture
def ik(world, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_INTAKE", True)
    monkeypatch.setattr(settings, "LEGAL_INTAKE_LINKS", True)
    world["users"]["rita"] = make_user("rita", mfa=True, firm_roles=["intake"])
    world["users"]["conan"] = make_user("conan", mfa=True, firm_roles=["conflicts"])
    world["users"]["rhea"] = make_user("rhea", mfa=True, firm_roles=["records_admin"])
    m202 = world["matters"]["M-202"]["id"]                       # highly restricted, client Beta Corp, led by bob
    bob = as_user(world, "bob")
    r = bob.post(f"/api/legal/matters/{m202}/parties", headers=csrf_headers(bob), json={"name": "Orrin Shipping Ltd", "role": "adverse"})
    assert r.status_code == 200, r.text
    world.update(base="/api/legal/intake")
    return world


def new_intake(w, who="rita", channel="phone") -> dict:
    c = as_user(w, who)
    r = c.post(w["base"], headers=csrf_headers(c), json={"channel": channel})
    assert r.status_code == 200, r.text
    return r.json()["intake"]


def submit(w, intake, answers=None, who="rita"):
    c = as_user(w, who)
    c.post(f"{w['base']}/{intake['id']}/disclaimer", headers=csrf_headers(c), json={"version": intake["disclaimer"]["version"]})
    return c.post(f"{w['base']}/{intake['id']}/submit", headers=csrf_headers(c), json={"answers": answers or ANSWERS})


def get(w, intake, who):
    c = as_user(w, who)
    return c.get(f"{w['base']}/{intake['id']}")


def decide(w, intake, decision="cleared", who="conan"):
    c = as_user(w, who)
    return c.post(f"{w['base']}/{intake['id']}/decision", headers=csrf_headers(c), json={"decision": decision, "note": "Checked the register."})


def anonymous(w) -> TestClient:
    return TestClient(w["app"], base_url=LAW_TEST_ORIGIN)


# ── Off by default ───────────────────────────────────────────────────────────────
def test_intake_is_off_unless_the_firm_turns_it_on(world, monkeypatch):
    from app.core.config import settings
    world["users"]["rita"] = make_user("rita", mfa=True, firm_roles=["intake"])
    rita = as_user(world, "rita")
    assert settings.LEGAL_INTAKE is False and settings.LEGAL_INTAKE_LINKS is False
    assert rita.get("/api/legal/intake").status_code == 404
    assert rita.post("/api/legal/intake", headers=csrf_headers(rita), json={"channel": "phone"}).status_code == 404
    assert anonymous(world).get("/api/intake-link/abcdefabcdefabcdefabcdefabcdef12").status_code == 404
    assert rita.get("/api/auth/me").json()["features"] == {"intake": False, "intake_links": False}


# ── The party register ───────────────────────────────────────────────────────────
def test_matter_parties_are_kept_by_the_matter_s_lawyers(ik):
    alice, carol, bob = as_user(ik, "alice"), as_user(ik, "carol"), as_user(ik, "bob")
    m101 = ik["matters"]["M-101"]["id"]
    r = alice.post(f"/api/legal/matters/{m101}/parties", headers=csrf_headers(alice), json={"name": "Northwind Logistics Ltd", "role": "adverse"})
    assert r.status_code == 200 and r.json()["party"]["role"] == "adverse"
    pid = r.json()["party"]["id"]
    assert [p["name"] for p in carol.get(f"/api/legal/matters/{m101}/parties").json()["parties"]] == ["Northwind Logistics Ltd"]
    assert carol.post(f"/api/legal/matters/{m101}/parties", headers=csrf_headers(carol), json={"name": "X Ltd", "role": "related"}).status_code == 403
    assert bob.get(f"/api/legal/matters/{m101}/parties").status_code == 404                 # not a member: no such matter
    assert alice.post(f"/api/legal/matters/{m101}/parties", headers=csrf_headers(alice), json={"name": " ", "role": "adverse"}).status_code == 422
    assert alice.post(f"/api/legal/matters/{m101}/parties", headers=csrf_headers(alice), json={"name": "Y Ltd", "role": "friend"}).status_code == 422
    r = alice.post(f"/api/legal/matters/{m101}/parties/{pid}/remove", headers=csrf_headers(alice), json={"reason": "Added in error"})
    assert r.status_code == 200
    assert alice.get(f"/api/legal/matters/{m101}/parties").json()["parties"] == []
    from app.core.db import get_conn
    with get_conn() as conn:                                     # removal keeps the history
        assert conn.execute("SELECT removed_at IS NOT NULL, removal_reason FROM legal_matter_parties WHERE id=?", (pid,)).fetchone() == (1, "Added in error")


# ── The script and the disclaimer ────────────────────────────────────────────────
def test_an_intake_starts_with_the_disclaimer_and_follows_the_script(ik):
    rita = as_user(ik, "rita")
    i = new_intake(ik)
    assert i["status"] == "open" and i["channel"] == "phone" and i["reference"].startswith("INT-")
    assert "not legal advice" in i["disclaimer"]["text"] and i["disclaimer"]["version"]
    assert [q["key"] for q in i["script"]["questions"]][:3] == ["your_name", "organisation", "email"]
    r = rita.post(f"{ik['base']}/{i['id']}/submit", headers=csrf_headers(rita), json={"answers": ANSWERS})
    assert r.status_code == 409 and "disclaimer" in r.json()["detail"]
    assert rita.post(f"{ik['base']}/{i['id']}/disclaimer", headers=csrf_headers(rita), json={"version": "old"}).status_code == 409
    for bad in ({**ANSWERS, "your_name": ""}, {**ANSWERS, "email": "", "phone": ""}, {**ANSWERS, "email": "not-an-email"},
                {**ANSWERS, "matter_type": "murder"}, {**ANSWERS, "other_parties": [f"P{n} Ltd" for n in range(11)]},
                {**ANSWERS, "summary": "x" * 601}, {**ANSWERS, "deadline": "next week"}, {**ANSWERS, "unknown": "y"}):
        r = submit(ik, i, bad)
        assert r.status_code == 422, bad
    r = submit(ik, i)
    assert r.status_code == 200 and r.json()["intake"]["status"] == "submitted"
    assert r.json()["intake"]["answers"]["organisation"] == "Kestrel Audio GmbH"


def test_the_assistant_writes_nothing_of_its_own(ik):
    """Scripted: the intake never calls a model."""
    from app.legal.research import llm

    class Fail:
        async def chat(self, *a, **k):
            raise AssertionError("intake called a model")
        chat_raw = chat
    llm._override = Fail()
    try:
        assert submit(ik, new_intake(ik)).status_code == 200
    finally:
        llm._override = None


# ── Screening ────────────────────────────────────────────────────────────────────
def test_staff_see_only_a_referral_and_the_conflicts_team_sees_the_matches(ik):
    i = new_intake(ik)
    r = submit(ik, i, {**ANSWERS, "organisation": "Beta Corporation", "other_parties": ["Orrin Shipping Limited"]})
    assert r.status_code == 200
    staff = r.json()["intake"]
    assert staff["screening"] == {"result": "possible_conflict", "message": "Possible conflict: refer to the conflicts team.", "decision": None}
    text = json.dumps(get(ik, i, "rita").json())
    for secret in ("M-202", "Board investigation", "Beta Corp\"", "highly_restricted", "matches"):
        assert secret not in text, secret
    assert "matches" not in get(ik, i, "alice").json()                          # lawyers who are not conflicts officers neither
    seen = get(ik, i, "conan").json()["matches"]
    got = {(m["searched"], m["matched"], m["kind"], m.get("matter_number")) for m in seen}
    assert ("Beta Corporation", "Beta Corp", "client", None) in got
    assert ("Orrin Shipping Limited", "Orrin Shipping Ltd", "adverse", "M-202") in got
    assert all(m["client"] == "Beta Corp" for m in seen)
    assert get(ik, i, "mp").json()["matches"]                                   # managing partners are the walls authority


@pytest.mark.parametrize("name, hit", [
    ("Smith, John", True), ("Alpha Holding", True), ("Alfa Holdings", True), ("ALPHA HOLDINGS LIMITED", True),
    ("The Orrin Shipping Company", True), ("Gamma Trading", False), ("Alphabet Soup Ltd", False), ("Ltd", False),
])
def test_names_match_across_order_spelling_and_legal_form(ik, name, hit):
    alice = as_user(ik, "alice")
    alice.post(f"/api/legal/matters/{ik['matters']['M-101']['id']}/parties", headers=csrf_headers(alice), json={"name": "John Smith", "role": "related"})
    from app.legal.intake import screening
    from app.legal.orgs import default_org_id
    assert bool(screening.screen(default_org_id(), [name])) is hit, name


def test_a_removed_party_no_longer_matches_and_a_screening_names_nothing_to_the_enquirer(ik):
    i = new_intake(ik)
    r = submit(ik, i, {**ANSWERS, "other_parties": ["Gamma Trading"]})
    assert r.json()["intake"]["screening"] == {"result": "no_match", "message": "No match found by the automatic check. The conflicts team confirms before we act.",
                                               "decision": None}


# ── The conflicts team ───────────────────────────────────────────────────────────
def test_the_conflicts_team_decides_and_staff_see_only_the_outcome(ik):
    i = new_intake(ik)
    submit(ik, i, {**ANSWERS, "other_parties": ["Orrin Shipping Ltd"]})
    assert decide(ik, i, who="rita").status_code == 403
    assert decide(ik, i, who="alice").status_code == 403
    r = decide(ik, i, "conflict")
    assert r.status_code == 200 and r.json()["intake"]["status"] == "conflict"
    s = get(ik, i, "rita").json()["intake"]
    assert s["screening"]["decision"] == "conflict" and "note" not in json.dumps(s["screening"])
    assert decide(ik, i, "cleared", who="mp").status_code == 200                 # a later decision is a new entry
    trail = [e["action"] for e in get(ik, i, "conan").json()["intake"]["events"]]
    assert trail[-2:] == ["conflict", "cleared"]


# ── Consent before recording ─────────────────────────────────────────────────────
def test_a_call_is_recorded_only_after_the_caller_consents(ik, tmp_path):
    from app.core.config import settings
    rita = as_user(ik, "rita")
    i = new_intake(ik)
    audio = b"RIFF\x00\x00\x00\x00WAVEfmt fictional call audio"
    up = lambda: rita.post(f"{ik['base']}/{i['id']}/recording", headers=csrf_headers(rita), files={"file": ("call.wav", audio, "audio/wav")})
    r = up()
    assert r.status_code == 409 and "consent" in r.json()["detail"]
    assert rita.post(f"{ik['base']}/{i['id']}/consent", headers=csrf_headers(rita), json={"method": "telepathy"}).status_code == 422
    r = rita.post(f"{ik['base']}/{i['id']}/consent", headers=csrf_headers(rita), json={"method": "verbal"})
    assert r.status_code == 200 and r.json()["intake"]["consent"]["method"] == "verbal"
    assert rita.post(f"{ik['base']}/{i['id']}/recording", headers=csrf_headers(rita),
                     files={"file": ("call.txt", b"text", "text/plain")}).status_code == 422
    assert up().status_code == 200
    assert rita.get(f"{ik['base']}/{i['id']}/recording").content == audio
    import glob
    stored = glob.glob(f"{settings.DATA_DIR}/legal/intake/**/*.bin", recursive=True)
    assert len(stored) == 1 and b"fictional call audio" not in open(stored[0], "rb").read()       # encrypted at rest
    r = rita.post(f"{ik['base']}/{i['id']}/consent/withdraw", headers=csrf_headers(rita))
    assert r.status_code == 200 and r.json()["intake"]["recording"] is None
    assert rita.get(f"{ik['base']}/{i['id']}/recording").status_code == 404
    assert glob.glob(f"{settings.DATA_DIR}/legal/intake/**/*.bin", recursive=True) == []
    assert up().status_code == 409                                              # withdrawn: no new recording


# ── Replies ──────────────────────────────────────────────────────────────────────
def test_replies_come_from_templates_and_a_lawyer_who_did_not_write_them_approves(ik):
    rita, alice, mp = as_user(ik, "rita"), as_user(ik, "alice"), as_user(ik, "mp")
    i = new_intake(ik)
    submit(ik, i)
    base = f"{ik['base']}/{i['id']}/replies"
    assert rita.post(base, headers=csrf_headers(rita), json={"template": "free_text"}).status_code == 422
    r = rita.post(base, headers=csrf_headers(rita), json={"template": "acknowledge"})
    assert r.status_code == 200
    reply = r.json()["reply"]
    assert reply["status"] == "draft" and "Hana Kowalczyk" in reply["body"] and i["reference"] in reply["body"]
    approve = lambda c, rep, sha=None: c.post(f"{base}/{rep['id']}/approve", headers=csrf_headers(c), json={"body_sha256": sha or rep["body_sha256"]})
    assert approve(rita, reply).status_code == 403                               # front desk staff are not lawyers
    assert approve(alice, reply, "0" * 64).status_code == 409                    # not the text the lawyer read
    r = approve(alice, reply)
    assert r.status_code == 200 and r.json()["reply"]["status"] == "released" and r.json()["reply"]["approved_by"] == "alice"
    r = alice.post(base, headers=csrf_headers(alice), json={"template": "can_help"})
    assert r.status_code == 409 and "conflicts team" in r.json()["detail"]
    decide(ik, i)
    can = alice.post(base, headers=csrf_headers(alice), json={"template": "can_help"}).json()["reply"]
    assert approve(alice, can).status_code == 409                                # not by its author
    assert approve(mp, can).status_code == 200
    replies = get(ik, i, "rita").json()["intake"]["replies"]
    assert [(x["template"], x["status"]) for x in replies] == [("acknowledge", "withdrawn"), ("can_help", "released")]


def test_a_reply_never_names_another_client_or_party_of_the_firm(ik):
    rita = as_user(ik, "rita")
    i = new_intake(ik)
    submit(ik, i)
    base = f"{ik['base']}/{i['id']}/replies"
    for text in ("Please send the contract you signed with Beta Corp.", "Is your dispute with Orrin Shipping?"):
        r = rita.post(base, headers=csrf_headers(rita), json={"template": "more_info", "request": text})
        assert r.status_code == 422 and "another client or party" in r.json()["detail"], text
    r = rita.post(base, headers=csrf_headers(rita), json={"template": "more_info", "request": "Please send the agreement Kestrel Audio GmbH signed."})
    assert r.status_code == 200 and "Kestrel Audio GmbH" in r.json()["reply"]["body"]


def test_cannot_act_gives_no_reason(ik):
    alice = as_user(ik, "alice")
    i = new_intake(ik)
    submit(ik, i, {**ANSWERS, "other_parties": ["Orrin Shipping Ltd"]})
    decide(ik, i, "conflict")
    body = alice.post(f"{ik['base']}/{i['id']}/replies", headers=csrf_headers(alice), json={"template": "cannot_act"}).json()["reply"]["body"]
    assert "not able to help" in body and "conflict" not in body.casefold() and "Orrin" not in body


# ── The enquirer's own link ──────────────────────────────────────────────────────
def test_an_enquirer_answers_through_a_single_use_link_and_sees_only_the_approved_reply(ik):
    rita, alice = as_user(ik, "rita"), as_user(ik, "alice")
    r = rita.post(f"{ik['base']}/links", headers=csrf_headers(rita), json={})
    assert r.status_code == 200
    link, intake = r.json()["link"], r.json()["intake"]
    token = link.rsplit("/", 1)[1]
    assert link == f"/intake/{token}" and len(token) >= 40 and intake["status"] == "invited"
    from app.core.db import get_conn
    with get_conn() as conn:
        assert token not in json.dumps(conn.execute("SELECT * FROM legal_intakes").fetchall())       # only a hash is kept
    assert token not in json.dumps(rita.get(f"{ik['base']}/{intake['id']}").json())
    anon = anonymous(ik)
    page = anon.get(f"/api/intake-link/{token}").json()
    assert page["status"] == "invited" and "not legal advice" in page["disclaimer"]["text"] and page["reply"] is None
    assert set(page) == {"status", "reference", "disclaimer", "script", "reply"}
    assert anon.post(f"/api/intake-link/{token}/submit", json={"answers": ANSWERS}).status_code == 409
    assert anon.post(f"/api/intake-link/{token}/accept", json={"version": page["disclaimer"]["version"]}).status_code == 200
    r = anon.post(f"/api/intake-link/{token}/submit", json={"answers": {**ANSWERS, "other_parties": ["Orrin Shipping Ltd"]}})
    assert r.status_code == 200 and set(r.json()) == {"status", "reference", "disclaimer", "script", "reply"}
    assert "conflicts team" not in json.dumps(r.json()) and "screening" not in json.dumps(r.json())   # the disclaimer's general word only
    assert anon.post(f"/api/intake-link/{token}/submit", json={"answers": ANSWERS}).status_code == 409       # once
    assert get(ik, intake, "rita").json()["intake"]["screening"]["result"] == "possible_conflict"
    reply = alice.post(f"{ik['base']}/{intake['id']}/replies", headers=csrf_headers(alice), json={"template": "acknowledge"}).json()["reply"]
    assert anon.get(f"/api/intake-link/{token}").json()["reply"] is None                                     # drafts are not shown
    as_user(ik, "mp").post(f"{ik['base']}/{intake['id']}/replies/{reply['id']}/approve", headers=csrf_headers(as_user(ik, "mp")),
                           json={"body_sha256": reply["body_sha256"]})
    assert anon.get(f"/api/intake-link/{token}").json()["reply"] == {"body": reply["body"]}
    assert anon.get("/api/intake-link/" + "x" * 43).status_code == 404
    assert anon.get(f"/api/legal/intake/{intake['id']}").status_code == 401                               # the link opens nothing else


def test_a_link_expires_and_closes_when_links_are_turned_off(ik, monkeypatch):
    from app.core.config import settings
    from app.core.db import get_conn
    rita = as_user(ik, "rita")
    r = rita.post(f"{ik['base']}/links", headers=csrf_headers(rita), json={}).json()
    token, iid = r["link"].rsplit("/", 1)[1], r["intake"]["id"]
    anon = anonymous(ik)
    monkeypatch.setattr(settings, "LEGAL_INTAKE_LINKS", False)
    assert anon.get(f"/api/intake-link/{token}").status_code == 404
    assert rita.post(f"{ik['base']}/links", headers=csrf_headers(rita), json={}).status_code == 404
    monkeypatch.setattr(settings, "LEGAL_INTAKE_LINKS", True)
    assert anon.get(f"/api/intake-link/{token}").status_code == 200
    with get_conn() as conn:
        conn.execute("UPDATE legal_intakes SET link_expires_at='2020-01-01T00:00:00.000Z' WHERE id=?", (iid,))
        conn.commit()
    assert anon.get(f"/api/intake-link/{token}").status_code == 404


def test_a_link_is_rate_limited(ik):
    rita = as_user(ik, "rita")
    token = rita.post(f"{ik['base']}/links", headers=csrf_headers(rita), json={}).json()["link"].rsplit("/", 1)[1]
    anon = anonymous(ik)
    codes = [anon.get(f"/api/intake-link/{token}").status_code for _ in range(31)]
    assert codes[:30] == [200] * 30 and codes[30] == 429


# ── Conversion ───────────────────────────────────────────────────────────────────
def test_a_cleared_intake_becomes_a_client_and_matter_with_its_parties(ik):
    rita, alice = as_user(ik, "rita"), as_user(ik, "alice")
    i = new_intake(ik)
    submit(ik, i)
    body = {"client_name": "Kestrel Audio GmbH", "client_code": "KESTREL", "matter_number": "M-301", "matter_title": "Distribution renewal"}
    r = alice.post(f"{ik['base']}/{i['id']}/convert", headers=csrf_headers(alice), json=body)
    assert r.status_code == 409 and "conflicts team" in r.json()["detail"]
    decide(ik, i)
    assert rita.post(f"{ik['base']}/{i['id']}/convert", headers=csrf_headers(rita), json=body).status_code == 403
    r = alice.post(f"{ik['base']}/{i['id']}/convert", headers=csrf_headers(alice), json=body)
    assert r.status_code == 200, r.text
    mid = r.json()["matter"]["id"]
    assert r.json()["intake"]["status"] == "converted" and r.json()["intake"]["matter_number"] == "M-301"
    m = alice.get(f"/api/legal/matters/{mid}").json()["matter"]
    assert m["client"]["name"] == "Kestrel Audio GmbH" and m["my_role"] == "lead"
    assert [(p["name"], p["role"]) for p in alice.get(f"/api/legal/matters/{mid}/parties").json()["parties"]] == [("Pinewood Retail Ltd", "adverse")]
    assert alice.post(f"{ik['base']}/{i['id']}/convert", headers=csrf_headers(alice), json=body).status_code == 409   # once


# ── Records and audit ────────────────────────────────────────────────────────────
def test_a_records_administrator_purges_an_intake_and_the_audit_keeps_no_personal_data(ik):
    from app.core.db import get_conn
    rita = as_user(ik, "rita")
    i = new_intake(ik)
    submit(ik, i, {**ANSWERS, "other_parties": ["Orrin Shipping Ltd"]})
    rita.post(f"{ik['base']}/{i['id']}/consent", headers=csrf_headers(rita), json={"method": "verbal"})
    rita.post(f"{ik['base']}/{i['id']}/recording", headers=csrf_headers(rita), files={"file": ("call.wav", b"RIFFfictional", "audio/wav")})
    rita.post(f"{ik['base']}/{i['id']}/replies", headers=csrf_headers(rita), json={"template": "acknowledge"})
    decide(ik, i, "conflict")
    assert rita.delete(f"{ik['base']}/{i['id']}", headers=csrf_headers(rita)).status_code == 403
    rhea = as_user(ik, "rhea")
    r = rhea.delete(f"{ik['base']}/{i['id']}", headers=csrf_headers(rhea))
    assert r.status_code == 200 and r.json()["intake"]["status"] == "purged"
    with get_conn() as conn:
        everything = json.dumps([conn.execute(f"SELECT * FROM {t}").fetchall() for t in
                                 ("legal_intakes", "legal_intake_matches", "legal_intake_replies", "legal_intake_events", "legal_intake_recordings")])
        audit = json.dumps(conn.execute("SELECT details_json, reason FROM audit_events WHERE action LIKE 'intake.%' OR action LIKE 'matter.party.%'").fetchall())
    for personal in ("Hana", "Kowalczyk", "kestrel.example", "Kestrel Audio", "Orrin", "Checked the register"):
        assert personal not in everything, personal
        assert personal not in audit, personal
    assert get(ik, i, "rita").json()["intake"]["answers"] == {}


def test_intake_records_are_kept_through_an_attempted_rollback(ik, fresh_db):
    from app.legal.migrations import runner
    submit(ik, new_intake(ik))
    with sqlite3.connect(fresh_db) as c:
        with pytest.raises(runner.MigrationError, match="intake"):
            runner.migrate_down(c, 30)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            c.execute("UPDATE legal_intake_events SET action='changed'")
