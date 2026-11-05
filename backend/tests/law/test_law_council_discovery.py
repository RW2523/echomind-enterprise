"""Council Discovery (EchoMind Counsel Intelligence, iteration 2; owner's request 2026-11-05).

Questions worth putting to the Council are found by rules in records the matter already holds — an unresolved potential
inconsistency, a sentence that amends another agreement's clause, a deadline no lawyer has confirmed — never written by the
model. Each is suggested once (fingerprint), shown only to people who may read every document it quotes (at most three), and
a person convenes it (a Council run asked in its words) or dismisses it with a reason. A finding that goes away makes its
question stale. Discovery runs after a document is read, after the register or the obligations change, and daily.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date, timedelta

import pytest

from conftest import as_user, csrf_headers, index_all, upload

AGREEMENT = ("SERVICES AGREEMENT (fictional)\n\nThis Services Agreement is dated 1 April 2026.\n\n"
             "4. Delivery\n4.2 The Supplier shall deliver the goods by 15 May 2026.\n")
AMENDMENT = ("AMENDMENT No. 1 (fictional)\n\nThis amendment to the Services Agreement dated 1 April 2026 takes effect on 2 May 2026.\n\n"
             "1. Change\n1.1 Clause 4.2 of the Services Agreement is deleted and replaced with: \"The Supplier shall deliver the goods by "
             "20 May 2026.\"\n")


class Model:
    """Discovery must never call it; the Council run it starts does."""
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, temperature, max_tokens):
        self.calls += 1
        return "NO OBJECTION"
    chat_raw = chat


@pytest.fixture
def dv(docs, monkeypatch, fresh_db):
    from app.core.config import settings
    from app.legal.matters import repo
    from app.legal.research import llm
    monkeypatch.setattr(settings, "LEGAL_PANEL", True)
    model = Model()
    monkeypatch.setattr(llm, "_override", model)
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    repo.add_membership(mid, docs["users"]["dave"]["id"], "attorney", granted_by=docs["users"]["alice"]["id"])
    agreement = upload(alice, mid, "services.txt", AGREEMENT, status="executed", title="Services Agreement").json()["document"]
    amendment = upload(alice, mid, "amendment.txt", AMENDMENT, status="executed", title="Amendment No. 1").json()["document"]
    index_all()
    with sqlite3.connect(fresh_db) as db:
        vid = {d: db.execute("SELECT current_version_id FROM legal_documents WHERE id=?", (d,)).fetchone()[0] for d in (agreement["id"], amendment["id"])}
        org = db.execute("SELECT org_id FROM matters WHERE id=?", (mid,)).fetchone()[0]
        side = lambda d, t, v: {"document_id": d, "document_title": t, "version_id": vid[d], "value": v, "value_start": 10, "value_end": 20}
        db.execute("INSERT INTO legal_discrepancies (id, org_id, matter_id, fingerprint, kind, verdict, method, status, document_a_id, document_b_id, "
                   "version_a_id, version_b_id, detail_json, found_by, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   ("dsc1", org, mid, "fp-dates", "date", "conflict", "model", "potential", agreement["id"], amendment["id"], vid[agreement["id"]],
                    vid[amendment["id"]], json.dumps({"a": side(agreement["id"], "Services Agreement", "15 May 2026"),
                                                      "b": side(amendment["id"], "Amendment No. 1", "20 May 2026")}), "test", "2026-11-05", "2026-11-05"))
        due = (date.today() + timedelta(days=10)).isoformat()
        db.execute("INSERT INTO legal_obligations (id, matter_id, document_id, version_id, clause, kind, party, action_text, when_text, due_date, "
                   "timezone, status, evidence_json, source, created_by, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   ("obl1", mid, agreement["id"], vid[agreement["id"]], "4.2", "payment", "Supplier", "deliver the goods", "by 15 May 2026", due,
                    "UTC", "proposed", json.dumps({"start": 60, "end": 100}), "rules", "test", "2026-11-05", "2026-11-05"))
    docs.update(mid=mid, model=model, org=org, agreement=agreement, amendment=amendment, due=due, base=f"/api/legal/matters/{mid}", db=fresh_db)
    return docs


def run(w):
    from app.legal.council import discovery
    return discovery.run_for_matter(w["org"], w["mid"])


def listing(w, who="alice"):
    return as_user(w, who).get(f"{w['base']}/panel").json()


def test_questions_come_from_the_records_by_rule_and_the_model_is_never_called(dv):
    assert run(dv)["found"] == 3 and dv["model"].calls == 0
    v = listing(dv)
    assert v["discovery"] is True and v["more_questions"] == 0
    titles = {q["detector"]: q for q in v["questions"]}
    assert titles["inconsistency"]["title"] == "Are “15 May 2026” in Services Agreement and “20 May 2026” in Amendment No. 1 genuinely inconsistent?"
    assert titles["amended"]["title"] == "How does Amendment No. 1 change clause 4.2 of the Services Agreement?"
    assert titles["deadline"]["title"] == f"What must be ready before {dv['due']} under clause 4.2 of Services Agreement?"
    assert {q["priority"] for q in v["questions"]} == {"high"}
    src = titles["amended"]["sources"][0]
    assert src["document_title"] == "Amendment No. 1" and src["href"].startswith(f"/m/M-101/documents/{dv['amendment']['id']}/read?v=")
    assert "&at=" in src["href"]


def test_a_question_is_suggested_once_and_a_dismissal_holds_until_its_sources_change(dv):
    run(dv); run(dv)
    with sqlite3.connect(dv["db"]) as db:
        assert db.execute("SELECT COUNT(*) FROM legal_council_questions WHERE matter_id=?", (dv["mid"],)).fetchone()[0] == 3
    deadline = next(q for q in listing(dv)["questions"] if q["detector"] == "deadline")
    alice = as_user(dv, "alice")
    url = f"{dv['base']}/council/questions/{deadline['id']}/dismiss"
    assert alice.post(url, headers=csrf_headers(alice), json={"reason": "  "}).status_code == 422
    assert alice.post(url, headers=csrf_headers(alice), json={"reason": "Already in the delivery plan."}).status_code == 200
    assert alice.post(url, headers=csrf_headers(alice), json={"reason": "Again."}).status_code == 409
    run(dv)
    assert "deadline" not in {q["detector"] for q in listing(dv)["questions"]}          # dismissed stays dismissed
    with sqlite3.connect(dv["db"]) as db:
        assert db.execute("SELECT decided_reason FROM legal_council_questions WHERE id=?", (deadline["id"],)).fetchone() == ("Already in the delivery plan.",)
        db.execute("UPDATE legal_obligations SET due_date=? WHERE id='obl1'", ((date.today() + timedelta(days=5)).isoformat(),))
    run(dv)
    assert "deadline" in {q["detector"] for q in listing(dv)["questions"]}              # its source changed: a new question


def test_a_question_whose_finding_is_gone_becomes_stale(dv):
    run(dv)
    with sqlite3.connect(dv["db"]) as db:
        db.execute("UPDATE legal_discrepancies SET status='explained' WHERE id='dsc1'")
        db.execute("UPDATE legal_obligations SET status='confirmed' WHERE id='obl1'")
    assert run(dv)["stale"] == 2
    assert [q["detector"] for q in listing(dv)["questions"]] == ["amended"]


def test_only_people_who_may_read_every_source_see_a_question(dv):
    from app.legal.documents import repo
    run(dv)
    assert len(listing(dv, "dave")["questions"]) == 3
    repo.set_acl(dv["amendment"]["id"], {dv["users"]["alice"]["id"]})                   # the amendment names alice only
    assert [q["detector"] for q in listing(dv, "dave")["questions"]] == ["deadline"]
    assert len(listing(dv, "alice")["questions"]) == 3
    hidden = next(q for q in listing(dv, "alice")["questions"] if q["detector"] == "amended")
    dave = as_user(dv, "dave")
    assert dave.post(f"{dv['base']}/council/questions/{hidden['id']}/dismiss", headers=csrf_headers(dave), json={"reason": "x"}).status_code == 404
    assert as_user(dv, "bob").get(f"{dv['base']}/panel").status_code == 404           # not on the matter


def test_convening_puts_the_question_to_the_council_in_its_own_words(dv):
    run(dv)
    q = next(q for q in listing(dv)["questions"] if q["detector"] == "inconsistency")
    url = f"{dv['base']}/council/questions/{q['id']}/convene"
    carol = as_user(dv, "carol")
    assert carol.post(url, headers=csrf_headers(carol), json={}).status_code == 403    # read-only on the matter
    alice = as_user(dv, "alice")
    r = alice.post(url, headers=csrf_headers(alice), json={"specialist": "commercial", "depth": "deep"})
    assert r.status_code == 200, r.text
    rid = r.json()["review"]["id"]
    assert alice.post(url, headers=csrf_headers(alice), json={}).status_code == 409
    with sqlite3.connect(dv["db"]) as db:
        row = db.execute("SELECT question, question_id, depth, roles_json FROM legal_panel_reviews WHERE id=?", (rid,)).fetchone()
        assert row[:3] == (q["title"], q["id"], "deep") and json.loads(row[3]) == ["lead", "opposing", "evidence", "contract"]
        assert db.execute("SELECT status, review_id FROM legal_council_questions WHERE id=?", (q["id"],)).fetchone() == ("convened", rid)
        audit = " ".join(r[0] or "" for r in db.execute("SELECT details_json FROM audit_events WHERE action LIKE 'council.%'"))
    assert '"detector":{"chars":13}' in audit and "15 May 2026" not in audit and "inconsistent" not in audit   # lengths, never words
    assert q["id"] not in {x["id"] for x in listing(dv)["questions"]}


def test_discovery_runs_after_a_document_is_read(dv):
    with sqlite3.connect(dv["db"]) as db:
        before = db.execute("SELECT COUNT(*) FROM legal_jobs WHERE kind='discovery.run' AND status='done'").fetchone()[0]
    assert before >= 1                                                                 # the fixture's uploads queued it
    upload(as_user(dv, "alice"), dv["mid"], "note.txt", "A short fictional note.", status="executed")
    index_all()
    with sqlite3.connect(dv["db"]) as db:
        assert db.execute("SELECT COUNT(*) FROM legal_jobs WHERE kind='discovery.run' AND status='done'").fetchone()[0] > before
        assert db.execute("SELECT COUNT(*) FROM legal_council_questions WHERE matter_id=?", (dv["mid"],)).fetchone()[0] == 3


def test_the_briefing_offers_the_questions(dv):
    from app.legal.counsel import briefing
    from app.legal.counsel.schemas import PriorityListProps
    run(dv)
    props, _ = briefing.build({"sub": dv["users"]["alice"]["id"], "org_id": dv["org"], "username": "alice"},
                              {"id": dv["mid"], "number": "M-101"})
    item = next(i for i in props["items"] if i["kind"] == "council")
    assert item["title"] == "3 questions worth putting to the Council" and item["action"]["path"] == "/m/M-101/panel"
    PriorityListProps(**props)


def test_discovery_is_off_with_the_panel_or_its_own_switch(dv, monkeypatch):
    from app.core.config import settings
    from app.legal.council import discovery
    run(dv)
    q = listing(dv)["questions"][0]
    monkeypatch.setattr(settings, "LEGAL_COUNCIL_DISCOVERY", False)
    alice = as_user(dv, "alice")
    v = listing(dv)
    assert v["discovery"] is False and v["questions"] == []
    assert alice.post(f"{dv['base']}/council/questions/{q['id']}/convene", headers=csrf_headers(alice), json={}).status_code == 404
    assert alice.post(f"{dv['base']}/council/discover", headers=csrf_headers(alice)).status_code == 404
    with sqlite3.connect(dv["db"]) as db:
        assert discovery.enqueue(db, dv["org"], dv["mid"], "x") is False
