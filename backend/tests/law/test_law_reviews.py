"""S2.7 — the review and approval framework (D-043).

Every AI output a lawyer relies on carries its sources, its status and a reviewer:
- **Research notes.** The person who asked can send an answer for review. The answer, its evidence
  and its status become a fixed, hashed note. A second lawyer approves exactly the version they read,
  and only an approved note is filed as a matter document. A note is shown only to people who can see
  every document it cites.
- **One queue.** "Reviews" lists what waits for you (drafts you may approve and did not write,
  exports, deletions, holds, walls) and what of yours waits for others. It never discloses a matter
  you cannot open.
- **Hash binding.** An export is bound to the text its redactions were planned on, and is refused if
  that text changed before approval. A document filed from an approved draft carries the approval
  into its export, and a changed copy of it is refused.
"""
from __future__ import annotations

import json
import re
import sqlite3

import pytest

from conftest import as_user, csrf_headers, index_all, upload

MSA = ("MASTER SERVICES AGREEMENT (fictional)\n\n"
       "4.2 Fees. The Customer shall pay the fees within thirty days of each invoice.\n\n"
       "12.1 Liability cap. Each party’s total liability is capped at USD 250,000.")
LETTER = "Side letter (fictional). The Supplier grants a service credit of 5% for each outage."


class Scripted:
    def __init__(self):
        self.answer = ""

    async def chat_stream(self, messages, temperature, max_tokens):
        for piece in re.findall(r"\S+\s*|\n+", self.answer):
            yield piece


@pytest.fixture
def rv(docs, monkeypatch):
    from app.core.config import settings
    from app.legal.matters import repo
    from app.legal.research import llm
    monkeypatch.setattr(settings, "LEGAL_STREAM_RECHECK_SEC", 0.0)
    fake = Scripted()
    monkeypatch.setattr(llm, "_override", fake)
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    lead = [m for m in repo.list_live_memberships(mid) if m["username"] == "alice"][0]
    repo.change_membership(mid, lead["id"], by=docs["users"]["alice"]["id"], can_export=True)
    repo.add_membership(mid, docs["users"]["dave"]["id"], "attorney", granted_by=docs["users"]["alice"]["id"])
    msa = upload(alice, mid, "msa.txt", MSA, status="executed").json()["document"]
    letter = upload(alice, mid, "side-letter.txt", LETTER, status="executed").json()["document"]
    index_all()
    docs.update(mid=mid, llm=fake, msa=msa, letter=letter, base=f"/api/legal/matters/{mid}")
    return docs


def ask(c, w, question, answer_with=None):
    """Ask, then (if a quote is wanted) ask again with the model citing the source holding it."""
    def post():
        r = c.post(f"{w['base']}/research/query", headers=csrf_headers(c), json={"question": question})
        assert r.status_code == 200, r.text
        return [json.loads(line) for line in r.text.splitlines() if line.strip()]
    w["llm"].answer = "x"
    events = post()
    if answer_with:
        n = next(s["n"] for s in events[0]["citations"] if answer_with in s["text"])
        w["llm"].answer = f'Liability is capped at USD 250,000 [{n}: "{answer_with}"].'
        events = post()
    return c.get(f"{w['base']}/research").json()["turns"][0]


def send(c, w, turn):
    return c.post(f"{w['base']}/research/{turn['id']}/review", headers=csrf_headers(c))


def note_of(c, w):
    turn = ask(c, w, "What is the liability cap?", answer_with="capped at USD 250,000")
    r = send(c, w, turn)
    assert r.status_code == 200, r.text
    return r.json()["artifact"]


def review(c, w, a, decision="approved", sha=None, reason=None):
    return c.post(f"{w['base']}/artifacts/{a['id']}/review", headers=csrf_headers(c),
                  json={"decision": decision, "content_sha256": sha or a["content_sha256"], "reason": reason})


def q(db, sql, *args):
    with sqlite3.connect(db) as c:
        return c.execute(sql, args).fetchall()


# ── Research notes ───────────────────────────────────────────────────────────────
def test_a_research_answer_is_sent_for_review_as_a_fixed_note(rv, fresh_db):
    alice = as_user(rv, "alice")
    turn = ask(alice, rv, "What is the liability cap?", answer_with="capped at USD 250,000")
    r = send(alice, rv, turn)
    assert r.status_code == 200, r.text
    a = r.json()["artifact"]
    assert a["kind"] == "research_note" and a["version"] == 1 and a["status"] == "draft" and a["origin"] == "model"
    c = a["content"]
    assert c["question"] == "What is the liability cap?" and c["status"] == "answered" and "USD 250,000" in c["answer"]
    assert [e["quote"] for e in c["evidence"]] == ["capped at USD 250,000"] and c["evidence"][0]["verification"] == "verbatim"
    assert c["asked_by"] == "alice" and a["title"] == "What is the liability cap?"
    assert re.fullmatch(r"[0-9a-f]{64}", a["content_sha256"])
    again = send(alice, rv, turn)                                   # sending again opens the same note
    assert again.status_code == 200 and again.json()["artifact"]["id"] == a["id"] and again.json()["changed"] is False
    with pytest.raises(sqlite3.IntegrityError):                     # the content is fixed
        with sqlite3.connect(fresh_db) as db:
            db.execute("UPDATE legal_ai_artifacts SET content_json='{}' WHERE id=?", (a["id"],))
    assert alice.get(f"{rv['base']}/artifacts/{a['id']}").json()["artifact"]["content"] == c


def test_only_the_person_who_asked_sends_an_answer_and_not_a_failed_one(rv, fresh_db):
    alice, dave, bob = as_user(rv, "alice"), as_user(rv, "dave"), as_user(rv, "bob")
    turn = ask(alice, rv, "What is the liability cap?", answer_with="capped at USD 250,000")
    assert send(dave, rv, turn).status_code == 404                  # turns are private to the person who asked
    assert send(bob, rv, turn).status_code == 404                   # not on the matter
    with sqlite3.connect(fresh_db) as db:
        db.execute("UPDATE legal_research_turns SET status='error' WHERE id=?", (turn["id"],))
        db.commit()
    r = send(alice, rv, turn)
    assert r.status_code == 409 and "could not be answered" in r.json()["detail"]


def test_a_second_lawyer_approves_the_note_they_read_and_it_is_filed_with_the_approval(rv, fresh_db):
    alice, dave, carol = as_user(rv, "alice"), as_user(rv, "dave"), as_user(rv, "carol")
    a = note_of(alice, rv)
    assert review(alice, rv, a).status_code == 409                  # not by its author
    assert review(carol, rv, a).status_code == 403                  # read-only members do not approve
    assert review(dave, rv, a, sha="0" * 64).status_code == 409     # not a version dave did not read
    r = review(dave, rv, a, "changes_requested", reason="")
    assert r.status_code == 422
    assert alice.post(f"{rv['base']}/artifacts/{a['id']}/file", headers=csrf_headers(alice)).status_code == 409   # not approved yet
    r = review(dave, rv, a)
    assert r.status_code == 200 and r.json()["artifact"]["status"] == "approved"
    f = alice.post(f"{rv['base']}/artifacts/{a['id']}/file", headers=csrf_headers(alice))
    assert f.status_code == 200 and f.json()["changed"] is True
    doc_id = f.json()["document_id"]
    index_all()
    doc = alice.get(f"{rv['base']}/documents/{doc_id}").json()["document"]
    assert doc["title"] == "Research note — What is the liability cap?" and doc["status"] == "final"
    text = alice.get(f"{rv['base']}/documents/{doc_id}/versions/{doc['current_version']['id']}/reader").json()["reader"]["text"]
    assert "Approved by dave" in text and a["content_sha256"] in text and "capped at USD 250,000" in text
    filing = q(fresh_db, "SELECT artifact_id, document_id, version_id, content_sha256, kind, artifact_version, approved_by "
                         "FROM legal_artifact_filings")
    assert filing == [(a["id"], doc_id, doc["current_version"]["id"], a["content_sha256"], "research_note", 1, rv["users"]["dave"]["id"])]
    assert alice.post(f"{rv['base']}/artifacts/{a['id']}/file", headers=csrf_headers(alice)).json()["changed"] is False


def test_a_newer_approved_version_supersedes_the_earlier_approval(rv):
    from app.legal import artifacts
    alice = as_user(rv, "alice")
    a = note_of(alice, rv)
    users = rv["users"]
    artifacts.decide(rv["mid"], a["id"], "approved", users["dave"]["id"], a["content_sha256"], None)
    b = artifacts.create(rv["mid"], "research_note", "research_turn", a["subject_id"], {**a["content"], "answer": "Changed."},
                         a["inputs"], users["alice"]["id"], "person")
    assert artifacts.get(rv["mid"], a["id"])["status"] == "approved"      # stays approved until a later version is
    artifacts.decide(rv["mid"], b["id"], "approved", users["dave"]["id"], b["content_sha256"], None)
    assert artifacts.get(rv["mid"], a["id"])["status"] == "superseded"
    assert artifacts.approved_hash(artifacts.get(rv["mid"], a["id"])) is None


def test_a_note_citing_a_document_you_cannot_see_is_not_shown_to_you(rv):
    from app.legal.documents import repo
    alice, dave = as_user(rv, "alice"), as_user(rv, "dave")
    a = note_of(alice, rv)
    repo.set_acl(rv["msa"]["id"], {rv["users"]["alice"]["id"]})          # the cited contract names alice only
    assert dave.get(f"{rv['base']}/artifacts/{a['id']}").status_code == 404
    assert review(dave, rv, a).status_code == 404
    assert dave.get("/api/legal/reviews").json()["waiting"] == []
    assert alice.get(f"{rv['base']}/artifacts/{a['id']}").status_code == 200


def test_artifacts_record_the_model_and_the_prompt_version(rv):
    from app.core.config import settings
    a = note_of(as_user(rv, "alice"), rv)
    assert a["inputs"]["model"] == settings.LLM_MODEL and a["inputs"]["prompt_version"].startswith("research-")
    assert a["inputs"]["turn_id"] == a["subject_id"]


# ── The queue ────────────────────────────────────────────────────────────────────
def test_the_queue_lists_what_waits_for_you_and_what_of_yours_waits_for_others(rv):
    from app.legal.records import deletion, holds
    alice, dave, mp, carol = as_user(rv, "alice"), as_user(rv, "dave"), as_user(rv, "mp"), as_user(rv, "carol")
    a = note_of(alice, rv)
    e = alice.post(f"{rv['base']}/exports", headers=csrf_headers(alice), json={"object_type": "document", "object_id": rv["letter"]["id"],
                                                                               "format": "pdf", "purpose": "To the client"}).json()["export"]
    assert alice.post(f"{rv['base']}/exports/{e['id']}/submit", headers=csrf_headers(alice)).status_code == 200
    org, users = rv["matters"]["M-101"]["org_id"], rv["users"]
    old = upload(alice, rv["mid"], "old-draft.txt", "An old fictional draft.").json()["document"]
    deletion.request(org, rv["mid"], "document", old["id"], "Duplicate copy", users["alice"]["id"])
    holds.create(org, "Harrowgate dispute", "Anticipated claim", users["mp2"]["id"], [rv["matters"]["M-114"]["id"]])

    waiting = {i["kind"]: i for i in dave.get("/api/legal/reviews").json()["waiting"]}
    assert set(waiting) == {"research_note", "export"}
    note = waiting["research_note"]
    assert note["title"] == "What is the liability cap?" and note["matter"]["number"] == "M-101" and note["by"] == "alice"
    assert note["path"] == f"/m/M-101/notes/{a['id']}" and note["version"] == 1
    assert waiting["export"]["path"] == f"/m/M-101/exports/{e['id']}" and waiting["export"]["by"] == "alice"

    mine = {i["kind"]: i for i in alice.get("/api/legal/reviews").json()["mine"]}
    assert mine["research_note"]["state"] == "waiting" and mine["export"]["state"] == "waiting"
    assert alice.get("/api/legal/reviews").json()["waiting"] == []     # never your own

    firm = {i["kind"] for i in mp.get("/api/legal/reviews").json()["waiting"]}
    assert firm == {"deletion", "hold"}                                  # records decisions for the records authority
    assert carol.get("/api/legal/reviews").json()["waiting"] == []     # a read-only member approves nothing

    review(dave, rv, a, "changes_requested", reason="Cite the side letter too.")
    mine = {i["kind"]: i for i in alice.get("/api/legal/reviews").json()["mine"]}
    assert mine["research_note"]["state"] == "changes_requested" and mine["research_note"]["reason"] == "Cite the side letter too."
    assert all(i["kind"] != "research_note" for i in dave.get("/api/legal/reviews").json()["waiting"])


def test_walls_and_emergency_access_wait_for_a_managing_partner_who_did_not_ask(rv):
    from app.legal.walls import repo as walls
    org, users = rv["matters"]["M-101"]["org_id"], rv["users"]
    walls.create_wall(org, "Beta screen", "Conflict with Beta", users["mp2"]["id"], [users["carol"]["id"]], [rv["matters"]["M-202"]["id"]], [])
    walls.create_break_glass(org, users["admin"]["id"], "M-114", "Partner unreachable; filing due", 60)
    kinds = {i["kind"]: i for i in as_user(rv, "mp").get("/api/legal/reviews").json()["waiting"]}
    assert kinds["wall"]["title"] == "Approve wall: Beta screen" and kinds["wall"]["path"] == "/admin/walls"
    assert kinds["emergency_access"]["title"] == "Emergency access to M-114" and kinds["emergency_access"]["by"] == "admin2"
    assert "wall" not in {i["kind"] for i in as_user(rv, "mp2").get("/api/legal/reviews").json()["waiting"]}   # not their own
    assert as_user(rv, "alice").get("/api/legal/reviews").json()["waiting"] == []


def test_the_queue_never_discloses_a_matter_you_cannot_open(rv):
    from app.legal.matters import repo
    alice, dave, bob = as_user(rv, "alice"), as_user(rv, "dave"), as_user(rv, "bob")
    note_of(alice, rv)
    assert bob.get("/api/legal/reviews").json() == {"waiting": [], "mine": [], "count": 0}
    m = [x for x in repo.list_live_memberships(rv["mid"]) if x["username"] == "dave"][0]
    repo.revoke_membership(rv["mid"], m["id"], by=rv["users"]["alice"]["id"], reason="moved team")
    assert dave.get("/api/legal/reviews").json()["waiting"] == []


# ── Hash binding ─────────────────────────────────────────────────────────────────
def start_export(c, w, doc_id):
    r = c.post(f"{w['base']}/exports", headers=csrf_headers(c), json={"object_type": "document", "object_id": doc_id,
                                                                      "format": "pdf", "purpose": "To the client"})
    return r


def test_an_export_is_bound_to_the_text_its_redactions_were_planned_on(rv, fresh_db):
    alice, dave = as_user(rv, "alice"), as_user(rv, "dave")
    e = start_export(alice, rv, rv["letter"]["id"]).json()["export"]
    assert re.fullmatch(r"[0-9a-f]{64}", e["source_sha256"])
    alice.post(f"{rv['base']}/exports/{e['id']}/submit", headers=csrf_headers(alice))
    upload(alice, rv["mid"], "side-letter-v2.txt", LETTER + " Account 4410098812 for refunds.", version_of=rv["letter"]["id"])
    index_all()
    r = dave.post(f"{rv['base']}/exports/{e['id']}/approve", headers=csrf_headers(dave), json={})
    assert r.status_code == 409 and "changed after this export was requested" in r.json()["detail"]
    assert q(fresh_db, "SELECT status, output_sha256 FROM legal_exports WHERE id=?", e["id"]) == [("failed", None)]


def test_a_change_to_the_text_alone_is_caught_too(rv, monkeypatch):
    """A transcript corrected after the request keeps its id but not its text."""
    from app.legal.exports import source
    alice, dave = as_user(rv, "alice"), as_user(rv, "dave")
    e = start_export(alice, rv, rv["letter"]["id"]).json()["export"]
    alice.post(f"{rv['base']}/exports/{e['id']}/submit", headers=csrf_headers(alice))
    real = source.load
    monkeypatch.setattr(source, "load", lambda *a: (lambda s: (setattr(s, "text", s.text + " Corrected."), s)[1])(real(*a)))
    r = dave.post(f"{rv['base']}/exports/{e['id']}/approve", headers=csrf_headers(dave), json={})
    assert r.status_code == 409


def test_an_export_requested_before_binding_is_asked_for_again(rv, fresh_db):
    alice, dave = as_user(rv, "alice"), as_user(rv, "dave")
    e = start_export(alice, rv, rv["letter"]["id"]).json()["export"]
    alice.post(f"{rv['base']}/exports/{e['id']}/submit", headers=csrf_headers(alice))
    with sqlite3.connect(fresh_db) as db:
        db.execute("UPDATE legal_exports SET source_sha256=NULL WHERE id=?", (e["id"],))
        db.commit()
    r = dave.post(f"{rv['base']}/exports/{e['id']}/approve", headers=csrf_headers(dave), json={})
    assert r.status_code == 409 and "requested before exports were bound" in r.json()["detail"]


def test_an_export_of_a_filed_note_carries_its_approval_and_a_changed_copy_is_refused(rv):
    alice, dave = as_user(rv, "alice"), as_user(rv, "dave")
    a = note_of(alice, rv)
    review(dave, rv, a)
    doc_id = alice.post(f"{rv['base']}/artifacts/{a['id']}/file", headers=csrf_headers(alice)).json()["document_id"]
    index_all()
    e = start_export(alice, rv, doc_id)
    assert e.status_code == 200, e.text
    approval = e.json()["export"]["plan"]["approval"]
    assert approval == {"kind": "research_note", "version": 1, "approved_by": "dave", "approved_at": approval["approved_at"],
                        "content_sha256": a["content_sha256"]}
    upload(alice, rv["mid"], "note-edited.md", "Research note, edited by hand. Liability is unlimited.", version_of=doc_id)
    index_all()
    r = start_export(alice, rv, doc_id)
    assert r.status_code == 409 and "changed since it was approved" in r.json()["detail"]


# ── Records ──────────────────────────────────────────────────────────────────────
def test_a_purge_of_a_cited_document_takes_unfiled_notes(rv, fresh_db):
    from app.legal.records import deletion
    alice = as_user(rv, "alice")
    a = note_of(alice, rv)
    org, users = rv["matters"]["M-101"]["org_id"], rv["users"]
    d = deletion.request(org, rv["mid"], "document", rv["msa"]["id"], "Duplicate", users["alice"]["id"])
    assert d["report"]["research_notes"] == 1
    assert deletion.purge(d["id"], users["mp"]["id"])["verification"]["ok"]
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_ai_artifacts WHERE kind='research_note'") == [(0,)]
    stored = " ".join(r[0] for r in q(fresh_db, "SELECT content_json FROM legal_ai_artifacts"))
    assert "capped at USD 250,000" not in stored



def test_a_filed_note_stays_when_its_source_is_purged(rv, fresh_db):
    from app.legal.records import deletion
    alice, dave = as_user(rv, "alice"), as_user(rv, "dave")
    a = note_of(alice, rv)
    review(dave, rv, a)
    doc_id = alice.post(f"{rv['base']}/artifacts/{a['id']}/file", headers=csrf_headers(alice)).json()["document_id"]
    org, users = rv["matters"]["M-101"]["org_id"], rv["users"]
    d = deletion.request(org, rv["mid"], "document", rv["msa"]["id"], "Duplicate", users["alice"]["id"])
    assert "Research notes and drafts already filed as matter documents" in d["report"]["kept"]
    assert deletion.purge(d["id"], users["mp"]["id"])["verification"]["ok"]
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_documents WHERE id=? AND deleted_at IS NULL", doc_id) == [(1,)]
    d2 = deletion.request(org, rv["mid"], "document", doc_id, "Superseded note", users["alice"]["id"])
    assert deletion.purge(d2["id"], users["mp"]["id"])["verification"]["ok"]
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_artifact_filings") == [(0,)]


def test_a_rollback_below_0021_refuses_to_delete_research_notes(rv):
    from app.core.config import settings
    from app.legal.migrations import runner
    note_of(as_user(rv, "alice"), rv)
    with sqlite3.connect(settings.DB_PATH) as conn, pytest.raises(Exception, match="research note"):
        runner.migrate_down(conn, 20)
