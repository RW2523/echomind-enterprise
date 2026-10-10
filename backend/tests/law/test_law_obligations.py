"""S3.6 — the obligations and deadlines tracker (DOC-05; decision D-050).

Obligations are read from a contract by rules (no model): notice periods, renewal windows, termination
and consent rights, change of control, payment, notification and post-termination duties. Each keeps
the exact words it was read from. A date is computed only from dates the document gives, the way of
counting is said, and nothing is shown as confirmed until a lawyer confirms it. Edits are versions with
reasons. A calendar file is off unless the firm turns it on.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from conftest import as_user, csrf_headers, index_all, upload

FX = Path(__file__).resolve().parents[3] / "eval" / "fixtures" / "law" / "Master Services Agreement (DEMO).md"


@pytest.fixture
def ob(docs):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    msa = upload(alice, mid, FX.name, FX.read_text(), mime="text/markdown", status="executed").json()["document"]
    index_all()
    docs.update(mid=mid, msa=msa, base=f"/api/legal/matters/{mid}", text=FX.read_text())
    return docs


def extract(w, by="alice"):
    c = as_user(w, by)
    r = c.post(f"{w['base']}/documents/{w['msa']['id']}/obligations/extract", headers=csrf_headers(c))
    assert r.status_code == 200, r.text
    return r.json()["obligations"]


def of(obs, clause):
    return [o for o in obs if o["clause"] == clause]


def patch(w, by, o, **body):
    c = as_user(w, by)
    return c.patch(f"{w['base']}/obligations/{o['id']}", headers=csrf_headers(c), json={"expected_version": o["version"], **body})


def test_obligations_are_read_from_the_agreement_with_their_exact_words(ob):
    obs = extract(ob)
    kinds = {(o["clause"], o["kind"]) for o in obs}
    assert {("2.2", "renewal"), ("4.2", "payment"), ("5.1", "confidentiality"), ("6.1", "notification"), ("6.2", "post_termination"),
            ("12.1", "termination"), ("12.2", "termination"), ("14.1", "consent"), ("14.2", "change_of_control"),
            ("14.3", "termination")} <= kinds
    for o in obs:
        assert o["status"] == "proposed" and o["source"] == "rules" and o["confirmed_by"] is None
        e = o["evidence"]
        assert ob["text"][e["start"]:e["end"]] == e["quote"]
    breach = of(obs, "6.1")[0]
    assert breach["party"] == "Provider" and "forty-eight (48) hours" in breach["when"] and "becoming aware" in breach["when"]
    assert breach["due_date"] is None and "event" in breach["due_note"]
    pay = of(obs, "4.2")[0]
    assert "thirty (30) days" in pay["when"] and "invoice date" in pay["when"]


def test_the_non_renewal_deadline_is_computed_from_the_agreement_s_own_dates_and_says_how(ob):
    renewal = of(extract(ob), "2.2")[0]
    assert renewal["due_date"] == "2028-12-01"
    note = renewal["due_note"]
    assert "2 March 2026" in note and "36 months" in note and "1 March 2029" in note and "90 days" in note
    assert "confirm" in note.lower()                     # the counting convention is for a lawyer to confirm


def test_a_date_is_never_guessed(docs):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    text = ("SUPPLY TERMS (fictional)\n\n1. Term\n1.1 This Agreement continues for an initial term of two (2) years.\n"
            "1.2 It renews for one year unless either party gives notice of non-renewal at least sixty (60) days before the end of the term.\n")
    d = upload(alice, mid, "terms.txt", text, status="executed").json()["document"]
    index_all()
    obs = alice.post(f"/api/legal/matters/{mid}/documents/{d['id']}/obligations/extract", headers=csrf_headers(alice)).json()["obligations"]
    r = of(obs, "1.2")[0]
    assert r["kind"] == "renewal" and r["due_date"] is None and "no start date" in r["due_note"].lower()


def test_nothing_is_confirmed_until_a_lawyer_confirms_it(ob):
    o = of(extract(ob), "6.1")[0]
    assert patch(ob, "carol", o, action="confirm").status_code == 403                 # read-only
    r = patch(ob, "alice", o, action="confirm", owner="alice")
    assert r.status_code == 200, r.text
    got = r.json()["obligation"]
    assert got["status"] == "confirmed" and got["confirmed_by"] == "alice" and got["owner"] == "alice" and got["version"] == 2
    listed = as_user(ob, "carol").get(f"{ob['base']}/obligations").json()["obligations"]
    assert next(x for x in listed if x["id"] == o["id"])["status"] == "confirmed"


def test_edits_are_versions_with_reasons_and_the_audit_holds_no_content(ob, fresh_db):
    o = of(extract(ob), "2.2")[0]
    assert patch(ob, "alice", o, action="edit", due_date="2028-11-30").status_code == 422           # a reason is required
    r = patch(ob, "alice", o, action="edit", due_date="2028-11-30", reason="Counted as 90 clear days")
    assert r.status_code == 200, r.text
    e = r.json()["obligation"]
    assert e["due_date"] == "2028-11-30" and e["version"] == 2 and e["due_note"].startswith("Set by alice")
    assert patch(ob, "alice", o, action="confirm").status_code == 409                               # stale version
    hist = as_user(ob, "alice").get(f"{ob['base']}/obligations/{o['id']}").json()["history"]
    assert [h["action"] for h in hist] == ["proposed", "edit"]
    assert hist[1]["reason"] == "Counted as 90 clear days" and hist[1]["changes"] == {"due_date": ["2028-12-01", "2028-11-30"]}
    with sqlite3.connect(fresh_db) as c:
        rows = c.execute("SELECT action, details_json FROM audit_events WHERE action LIKE 'obligation.%'").fetchall()
    assert {a for a, _ in rows} >= {"obligation.extract", "obligation.update"}
    assert not any("non-renewal" in (d or "") or "90 clear days" in (d or "") for _, d in rows)


def test_a_due_date_is_shown_in_its_timezone_and_marked_overdue_by_it(ob):
    o = of(extract(ob), "2.2")[0]
    assert o["timezone"] == "UTC" and o["due_at"] == "2028-12-01T23:59:59+00:00" and o["overdue"] is False
    assert patch(ob, "alice", o, action="edit", timezone="Mars/Olympus", reason="x").status_code == 422
    e = patch(ob, "alice", o, action="edit", timezone="Europe/Stockholm", reason="The client is in Sweden").json()["obligation"]
    assert e["due_at"] == "2028-12-01T23:59:59+01:00"
    e2 = patch(ob, "alice", e, action="edit", timezone="America/New_York", due_date="2026-07-01", reason="Summer date").json()["obligation"]
    assert e2["due_at"] == "2026-07-01T23:59:59-04:00" and e2["overdue"] is True                   # 2026-10-10 is after it


def test_dismissing_needs_a_reason_and_reading_again_keeps_every_decision(ob):
    obs = extract(ob)
    o = of(obs, "14.1")[0]
    assert patch(ob, "alice", o, action="dismiss").status_code == 422
    assert patch(ob, "alice", o, action="dismiss", reason="A consent right, not a deadline").json()["obligation"]["status"] == "dismissed"
    again = extract(ob)
    assert len(again) == len(obs)
    assert of(again, "14.1")[0]["status"] == "dismissed"


def test_a_lawyer_adds_an_obligation_from_the_exact_words(ob):
    alice = as_user(ob, "alice")
    text = ob["text"]
    a = text.index("Undisputed amounts not paid when due")
    b = text.index("until paid.", a) + len("until paid.")
    r = alice.post(f"{ob['base']}/obligations", headers=csrf_headers(alice),
                   json={"document_id": ob["msa"]["id"], "start": a, "end": b, "kind": "payment", "action": "Interest on late payment", "party": "Client"})
    assert r.status_code == 200, r.text
    o = r.json()["obligation"]
    assert o["source"] == "person" and o["evidence"]["quote"].startswith("Undisputed amounts") and o["clause"] == "4.3"
    bad = alice.post(f"{ob['base']}/obligations", headers=csrf_headers(alice),
                     json={"document_id": ob["msa"]["id"], "start": len(text) + 10, "end": len(text) + 20, "kind": "payment", "action": "x"})
    assert bad.status_code == 422


def test_the_calendar_file_is_off_unless_the_firm_turns_it_on(ob, monkeypatch):
    from app.core.config import settings
    o = of(extract(ob), "2.2")[0]
    patch(ob, "alice", o, action="confirm", owner="alice")
    alice = as_user(ob, "alice")
    assert alice.get(f"{ob['base']}/obligations.ics").status_code == 404
    monkeypatch.setattr(settings, "LEGAL_OBLIGATIONS_ICS", True)
    r = alice.get(f"{ob['base']}/obligations.ics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/calendar")
    body = r.text
    assert body.count("BEGIN:VEVENT") == 1 and "DTSTART;VALUE=DATE:20281201" in body             # confirmed and dated only


def test_obligations_are_shown_only_with_their_document_and_purged_with_it(ob, fresh_db):
    from app.legal.documents import repo
    from app.legal.records import deletion
    extract(ob)
    assert as_user(ob, "carol").get(f"{ob['base']}/obligations").json()["obligations"]
    repo.set_acl(ob["msa"]["id"], {ob["users"]["alice"]["id"]})
    assert as_user(ob, "carol").get(f"{ob['base']}/obligations").json()["obligations"] == []
    assert as_user(ob, "bob").get(f"{ob['base']}/obligations").status_code == 404
    org, users = ob["matters"]["M-101"]["org_id"], ob["users"]
    d = deletion.request(org, ob["mid"], "document", ob["msa"]["id"], "Wrong matter", users["alice"]["id"])
    assert d["report"]["obligations"] >= 10
    assert deletion.purge(d["id"], users["mp"]["id"])["verification"]["ok"]
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT COUNT(*) FROM legal_obligations").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM legal_obligation_events").fetchone()[0] == 0


def test_a_lease_break_date_term_expiry_and_duties_are_read_with_dates_only_from_the_document():
    """Golden tuning (obligations-s3.6b): the break notice date from an anniversary, the term's end, duties with
    a time limit, a notification without the word "breach", survival, and a heading line kept out of the quote."""
    import json
    from app.legal.documents import extract as rd, structure
    from app.legal.obligations import extract as ex
    g = json.loads((Path(__file__).resolve().parents[3] / "eval" / "law" / "obligations" / "golden.json").read_text())

    def read(doc_id):
        text = next(d["text"] for d in g["documents"] if d["id"] == doc_id)
        x = rd.extract("d.txt", text.encode())
        rows = [{"number": c.number, "heading": c.heading, "level": c.level, "parent": c.parent, "start": c.start, "end": c.end}
                for c in structure.clauses(x)]
        return {(o["clause"], o["kind"]): o for o in ex.read(text, rows)}
    lease = read("G2-office-lease")
    brk = lease[("3.1", "termination")]
    assert brk["due_date"] == "2030-09-25" and "25 March 2031" in brk["due_note"] and brk["evidence"]["quote"].startswith("3.1 The Tenant")
    assert lease[("1.2", "expiry")]["due_date"] == "2036-03-24"
    assert lease[("4.1", "duty")]["due_date"] is None and "ten (10) Business Days" in lease[("4.1", "duty")]["when"]
    assert lease[("2.1", "payment")]["when"].startswith("quarterly in advance")
    nda = read("G3-mutual-nda")
    assert nda[("1.1", "expiry")]["due_date"] == "2029-05-03"
    assert ("2.2", "notification") in nda and ("2.3", "survival") in nda and ("2.1", "duty") in nda
    assert read("G4-board-minutes-control") == {}
