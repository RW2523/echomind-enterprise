"""EchoMind Legal chat: the cards that work on the matter's records (docs/echomind-legal/03 registry; owner's request
2026-11-02, "build all of it").

checklist (the obligations register: tick done, reopen, confirm a date, all through the register's own change),
approval_request (the review queue: decide through the review route's own rules, never filing anything),
guided_questions + document_preview (three questions, then a draft from a firm template, open beside the chat),
council_summary (the review panel queued from the chat), numeric_chart (stated amounts, by rule), graph_excerpt (parties
and the documents naming them, by rule), live_transcript (a meeting being recorded). Each acts only for a person with
the right, on what the card showed, at the version it showed; each is rebuilt from its records when read again.
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic import ValidationError

from conftest import as_user, csrf_headers
from test_law_counsel import cs, events_of, final, new_conversation, turn  # noqa: F401  (cs is a fixture)
from test_law_drafting import dr, template  # noqa: F401  (dr is a fixture)


def card(env, type_):
    return next(c for c in env["components"] if c["type"] == type_)


def act(c, comp, matter_id, action_id, version, parameters, key=None):
    import secrets
    return c.post("/api/legal/counsel/actions", headers=csrf_headers(c), json={
        "component_id": comp["id"], "action_id": action_id, "matter_id": matter_id, "expected_version": version,
        "parameters": parameters, "idempotency_key": key or secrets.token_hex(12)})


def read_card(c, comp_id):
    r = c.get(f"/api/legal/counsel/components/{comp_id}")
    assert r.status_code == 200, r.text
    return r.json()["component"]


# ── checklist ────────────────────────────────────────────────────────────────────
def _obligation(c, mid, doc_id, words, due=None):
    vid = c.get(f"/api/legal/matters/{mid}/documents/{doc_id}/versions").json()["versions"][0]["id"]
    text = c.get(f"/api/legal/matters/{mid}/documents/{doc_id}/versions/{vid}/reader").json()["reader"]["text"]
    start = text.index(words)
    r = c.post(f"/api/legal/matters/{mid}/obligations", headers=csrf_headers(c), json={
        "document_id": doc_id, "start": start, "end": start + len(words), "kind": "payment", "action": words[:120],
        "party": "Apex Industries, Inc.", **({"due_date": due} if due else {})})
    assert r.status_code == 200, r.text
    return r.json()["obligation"]


def test_the_checklist_ticks_reopens_and_confirms_through_the_register(cs, fresh_db):
    alice, carol = as_user(cs, "alice"), as_user(cs, "carol")
    mid, doc = cs["mid"], cs["doc_ids"]["services-agreement.txt"]
    o1 = _obligation(alice, mid, doc, "complete delivery by May 15, 2026", due="2026-05-15")
    o2 = _obligation(alice, mid, doc, "pay USD 120,000 within thirty days of delivery")
    calls = len(cs["model"].stream_calls) + len(cs["model"].chat_calls)
    conv = new_conversation(alice, mid)
    env = final(turn(alice, conv, "What are the open obligations?"))
    assert len(cs["model"].stream_calls) + len(cs["model"].chat_calls) == calls            # from the register, no model
    comp = card(env, "checklist")
    items = {i["id"]: i for i in comp["props"]["items"]}
    assert set(items) == {o1["id"], o2["id"]} and comp["props"]["can_done"] and comp["props"]["can_confirm"]
    assert items[o1["id"]]["status"] == "proposed" and items[o1["id"]]["path"].startswith(f"/m/M-101/documents/{doc}/read?v=")
    assert comp["props"]["summary"] == "2 open obligations in M-101, 1 overdue, 1 with a date not yet confirmed."   # 15 May 2026 has passed

    r = act(alice, comp, mid, "obligation", 1, {"obligation_id": o2["id"], "action": "done"})
    assert r.status_code == 200, r.text
    now = {i["id"]: i for i in r.json()["component"]["props"]["items"]}
    assert now[o2["id"]]["status"] == "done" and now[o2["id"]]["version"] == 2
    stale = act(alice, comp, mid, "obligation", 1, {"obligation_id": o2["id"], "action": "reopen"})
    assert stale.status_code == 409 and stale.json()["code"] == "VERSION_CONFLICT" and stale.json()["component"]["type"] == "checklist"
    assert act(alice, comp, mid, "obligation", 2, {"obligation_id": o2["id"], "action": "reopen"}).status_code == 200
    r = act(alice, comp, mid, "obligation", 1, {"obligation_id": o1["id"], "action": "confirm"})
    assert {i["id"]: i for i in r.json()["component"]["props"]["items"]}[o1["id"]]["status"] == "confirmed"
    assert alice.get(f"/api/legal/matters/{mid}/obligations/{o1['id']}").json()["obligation"]["confirmed_by"] == "alice"

    # Read again: the card shows the register as it stands, not as it was when asked.
    again = alice.get(f"/api/legal/counsel/conversations/{conv}").json()["turns"][0]["envelope"]["components"][0]
    assert {i["id"]: i["status"] for i in again["props"]["items"]} == {o1["id"]: "confirmed", o2["id"]: "proposed"}
    assert [e["details"] for e in events_of(fresh_db, "obligation.update") if '"mode":"chat"' in (e["details"] or "").replace(" ", "")]

    # Only what the card showed, and only with the right.
    o3 = _obligation(alice, mid, doc, "Morgan Company LLC shall pay USD 120,000")
    assert act(alice, comp, mid, "obligation", 1, {"obligation_id": o3["id"], "action": "done"}).status_code == 404
    assert act(alice, comp, mid, "obligation", 1, {"obligation_id": o1["id"], "action": "dismiss"}).status_code == 422
    cconv = new_conversation(carol, mid)
    ccomp = card(final(turn(carol, cconv, "List upcoming deadlines")), "checklist")
    assert not ccomp["props"]["can_done"] and not ccomp["props"]["can_confirm"]
    assert act(carol, ccomp, mid, "obligation", 3, {"obligation_id": o2["id"], "action": "done"}).status_code == 403


def test_a_question_about_what_a_document_says_about_deadlines_is_still_research(cs):
    from app.legal.counsel import intent
    for q in ("What deadlines does the lease set?", "What is the deadline to give notice of non-renewal?",
              "What does the agreement say about payment deadlines?"):
        assert intent.classify(q) == "ask_fact", q
    for q, kind in (("What are the open obligations?", "tasks"), ("Which obligations are overdue?", "tasks"),
                    ("What is waiting for my approval?", "approvals"), ("Show the live transcript", "live"),
                    ("Chart the amounts in the documents", "amounts"), ("How are the parties connected?", "parties"),
                    ("Does clause 5 require approval of the landlord?", "ask_fact"), ("What is the monthly payment?", "ask_fact")):
        assert intent.classify(q) == kind, q


# ── drafting from the chat ───────────────────────────────────────────────────────
def test_a_draft_request_asks_three_questions_then_writes_from_the_template(dr, fresh_db):
    t = template(dr)
    alice = as_user(dr, "alice")
    conv = new_conversation(alice, dr["mid"])
    env = final(turn(alice, conv, "Draft an engagement letter for Harrowgate"))
    form = card(env, "guided_questions")
    qs = {q["id"]: q for q in form["props"]["questions"]}
    assert list(qs) == ["template", "title", "note"] and qs["template"]["default"] == t["id"]
    assert qs["title"]["default"] == "Engagement letter for Harrowgate"
    assert "_form" not in json.dumps(env)                                 # the server's own notes stay on the server
    ev = turn(alice, conv, "Write the draft “Engagement letter — Harrowgate”",
              form={"component_id": form["id"], "values": {"title": "Engagement letter — Harrowgate"}})
    assert [e["event"] for e in ev][-1] == "turn.completed", ev
    prev = card(final(ev), "document_preview")
    p = prev["props"]
    assert p["title"] == "Engagement letter — Harrowgate" and p["status"] == "draft" and p["version"] == 1 and p["can_edit"]
    assert p["gaps"] == 1 and {f["label"]: f["status"] for f in p["fields"]}["Signatory"] == "needs_input"
    assert {f["label"]: f["value"] for f in p["fields"]}["Client"] == "Harrowgate Systems Inc." and p["sources"] >= 2
    assert p["path"] == f"/m/M-101/drafting/{p['artifact_id']}"
    assert any(e["details"] and '"mode":"chat"' in e["details"].replace(" ", "") for e in events_of(fresh_db, "draft.create"))
    # A second lawyer sees it waiting; nothing was filed.
    dave = as_user(dr, "dave")
    assert any(w["id"] == p["artifact_id"] for w in dave.get("/api/legal/reviews").json()["waiting"])
    # Corrected elsewhere: the card shows the newest version.
    r = alice.post(f"{dr['base']}/drafts/{p['artifact_id']}/edit", headers=csrf_headers(alice), json={
        "content_sha256": alice.get(f"{dr['base']}/artifacts/{p['artifact_id']}").json()["artifact"]["content_sha256"],
        "fields": [{"key": "signatory", "value": "A. Partner"}], "blocks": []})
    assert r.status_code == 200, r.text
    now = read_card(alice, prev["id"])["props"]
    assert now["version"] == 2 and now["gaps"] == 0

    # Answers are checked against the questions asked: a choice from elsewhere fails; another conversation's form is refused.
    bad = turn(alice, conv, "Write it", form={"component_id": form["id"], "values": {"template": "not-a-template"}})
    assert bad[-1]["event"] == "turn.failed" and "Choose one of the options" in bad[-1]["data"]["message"]
    other = new_conversation(alice, dr["mid"])
    r = alice.post(f"/api/legal/counsel/conversations/{other}/turns", headers=csrf_headers(alice),
                   json={"text": "Write it", "form": {"component_id": form["id"], "values": {}}})
    assert r.status_code == 404
    r = alice.post(f"/api/legal/counsel/conversations/{conv}/turns", headers=csrf_headers(alice),
                   json={"text": "Write it", "form": {"component_id": prev["id"], "values": {}}})
    assert r.status_code == 404                                         # only a form can be answered


def test_a_read_only_member_is_not_offered_drafting_and_a_firm_without_templates_is_told(dr):
    carol = as_user(dr, "carol")
    env = final(turn(carol, new_conversation(carol, dr["mid"]), "Draft an engagement letter"))
    assert card(env, "notice")["props"]["text"] == "Your role in this matter does not include drafting."
    alice = as_user(dr, "alice")
    env = final(turn(alice, new_conversation(alice, dr["mid"]), "Draft an engagement letter"))
    assert "no approved drafting template" in card(env, "notice")["props"]["text"]


# ── approvals ────────────────────────────────────────────────────────────────────
def _note(dr, by="dave"):
    from app.legal import artifacts
    import uuid
    content = {"question": "When is delivery due?", "answer": "Delivery is due by May 15, 2026.", "status": "answered", "evidence": [],
               "sources": [], "conflicts": [], "unsupported": []}
    return artifacts.create(dr["mid"], "research_note", "research_turn", str(uuid.uuid4()), content, {}, dr["users"][by]["id"], "model")


def test_the_approval_card_decides_through_the_review_rules_and_never_files(dr, fresh_db):
    t = template(dr)
    alice = as_user(dr, "alice")
    note = _note(dr)
    dave = as_user(dr, "dave")
    gapped = dave.post(f"{dr['base']}/drafts", headers=csrf_headers(dave), json={"template_id": t["id"], "title": "Letter by Dave"}).json()["draft"]
    conv = new_conversation(alice, dr["mid"])
    comp = card(final(turn(alice, conv, "What is waiting for my approval?")), "approval_request")
    items = {i["artifact_id"]: i for i in comp["props"]["items"]}
    assert set(items) == {note["id"], gapped["id"]} and all(i["can_decide"] for i in items.values())
    assert items[gapped["id"]]["gaps"] == 1 and "Nothing is filed or sent" in items[gapped["id"]]["consequence"]
    assert items[note["id"]]["preview"] == ["Delivery is due by May 15, 2026."] and items[note["id"]]["path"] == f"/m/M-101/notes/{note['id']}"

    assert act(alice, comp, dr["mid"], "decide", 1, {"artifact_id": note["id"], "decision": "approved"}).status_code == 422   # read it first
    key = "approve-note-0001"
    r = act(alice, comp, dr["mid"], "decide", 1, {"artifact_id": note["id"], "decision": "approved", "read_it": True}, key=key)
    assert r.status_code == 200, r.text
    after = {i["artifact_id"]: i for i in r.json()["component"]["props"]["items"]}[note["id"]]
    assert after["state"] == "approved" and after["decided_by"] == "alice" and not after["can_decide"]
    again = act(alice, comp, dr["mid"], "decide", 1, {"artifact_id": note["id"], "decision": "approved", "read_it": True}, key=key)
    assert again.status_code == 200 and again.json()["replayed"]
    gap = act(alice, comp, dr["mid"], "decide", 1, {"artifact_id": gapped["id"], "decision": "approved", "read_it": True})
    assert gap.status_code == 409 and "gap" in gap.json()["detail"]
    assert act(alice, comp, dr["mid"], "decide", 1, {"artifact_id": gapped["id"], "decision": "changes_requested"}).status_code == 422
    r = act(alice, comp, dr["mid"], "decide", 1, {"artifact_id": gapped["id"], "decision": "changes_requested", "reason": "Name the signatory."})
    assert {i["artifact_id"]: i["state"] for i in r.json()["component"]["props"]["items"]}[gapped["id"]] == "changes_requested"
    with sqlite3.connect(fresh_db) as db:                              # deciding is not filing
        assert db.execute("SELECT COUNT(*) FROM legal_artifact_filings").fetchone() == (0,)
    assert [e for e in events_of(fresh_db, "artifact.review") if '"mode":"chat"' in (e["details"] or "").replace(" ", "")]
    # Only what the card showed; and not the author's own, not a reader's.
    other = _note(dr)
    assert act(alice, comp, dr["mid"], "decide", 1, {"artifact_id": other["id"], "decision": "approved", "read_it": True}).status_code == 404
    carol = as_user(dr, "carol")
    ccomp = card(final(turn(carol, new_conversation(carol, dr["mid"]), "What is waiting for my approval?")), "approval_request")
    assert ccomp["props"]["items"] == [] and ccomp["props"]["summary"] == "Nothing in M-101 waits for your decision."
    dcomp = card(final(turn(dave, new_conversation(dave, dr["mid"]), "Anything for me to approve?")), "approval_request")
    assert other["id"] not in {i["artifact_id"] for i in dcomp["props"]["items"]}       # his own


# ── the review panel ─────────────────────────────────────────────────────────────
def test_the_review_panel_is_queued_from_the_chat_when_the_firm_has_it(dr, monkeypatch, fresh_db):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_PANEL", True)
    from app.legal.jobs import worker
    monkeypatch.setattr(worker, "wake", lambda: None)
    alice = as_user(dr, "alice")
    conv = new_conversation(alice, dr["mid"])
    form = card(final(turn(alice, conv, "Challenge our position on the fee")), "guided_questions")
    assert form["props"]["purpose"] == "panel" and [q["id"] for q in form["props"]["questions"]] == ["question", "specialist", "depth"]
    assert "Lead counsel, Opposing reading, Evidence investigator" in form["props"]["intro"]      # chosen by rule for a challenge
    ev = turn(alice, conv, "Ask the review panel: is the fixed fee binding?",
              form={"component_id": form["id"], "values": {"question": "Is the fixed fee of USD 18,000 binding on the client?",
                                                           "specialist": "commercial", "depth": "deep"}})
    summary = card(final(ev), "council_summary")
    assert summary["props"]["status"] == "queued" and summary["props"]["area"] == "commercial contracts"
    assert summary["props"]["path"] == f"/m/M-101/panel/{summary['props']['review_id']}"
    with sqlite3.connect(fresh_db) as db:
        assert db.execute("SELECT kind FROM legal_jobs WHERE object_id=?", (summary["props"]["review_id"],)).fetchone() == ("panel.run",)
        assert db.execute("SELECT depth, roles_json FROM legal_panel_reviews WHERE id=?", (summary["props"]["review_id"],)).fetchone() == \
            ("deep", '["deal", "risk", "litigation", "specialist"]')
    assert read_card(alice, summary["id"])["props"]["status"] == "queued"
    carol = as_user(dr, "carol")
    env = final(turn(carol, new_conversation(carol, dr["mid"]), "Challenge our position on the fee"))
    assert "for the matter's lawyers and paralegals" in card(env, "notice")["props"]["text"]


# ── amounts and parties ──────────────────────────────────────────────────────────
def test_amounts_and_parties_come_from_the_documents_by_rule(cs):
    alice = as_user(cs, "alice")
    calls = len(cs["model"].stream_calls) + len(cs["model"].chat_calls)
    conv = new_conversation(alice, cs["mid"])
    chart = card(final(turn(alice, conv, "Chart the amounts in the documents")), "numeric_chart")["props"]
    assert chart["unit"] == "USD" and [b["shown"] for b in chart["bars"]] == ["USD 120,000"]
    bar = chart["bars"][0]
    assert bar["value"] == 120000 and bar["document_title"] and bar["source"]["value"] == "USD 120,000"
    assert bar["source"]["reader_path"].startswith(f"/m/M-101/documents/{cs['doc_ids']['services-agreement.txt']}/read?v=")
    assert "40 million" not in json.dumps(chart)                         # M-114's term sheet is another matter
    graph = card(final(turn(alice, conv, "How are the parties connected?")), "graph_excerpt")["props"]
    parties = {n["label"]: n for n in graph["nodes"] if n["kind"] == "party"}
    assert {"Morgan Company LLC", "Apex Industries, Inc."} <= set(parties)
    agreement = next(n["id"] for n in graph["nodes"] if n["kind"] == "document" and "services" in n["label"].lower())
    edges = {(e["source"], e["target"]): e for e in graph["edges"]}
    assert edges[(parties["Morgan Company LLC"]["id"], agreement)]["label"] == "party to"
    assert "ZEPHYR" not in json.dumps(graph) and "PWNED" not in json.dumps(graph)
    assert len(cs["model"].stream_calls) + len(cs["model"].chat_calls) == calls


# ── a meeting being recorded ─────────────────────────────────────────────────────
def _meeting(db_path, mid, by):
    with sqlite3.connect(db_path) as db:
        db.execute("INSERT INTO legal_meetings (id, org_id, matter_id, title, kind, status, consent_policy, created_by, created_at) "
                   "SELECT 'mtg1', org_id, id, 'Client call (fictional)', 'client_call', 'planned', 'all_parties', ?, '2026-11-02T10:00:00.000Z' "
                   "FROM matters WHERE id=?", (by, mid))
        db.execute("INSERT INTO legal_captures (id, meeting_id, matter_id, started_by, started_at, consent_snapshot, transcription) "
                   "VALUES ('cap1','mtg1',?,?,'2026-11-02T10:01:00.000Z','[]','available')", (mid, by))
        for i, text in enumerate(("We expect delivery by the fifteenth.", "Apex says the twelfth was dispatch."), start=1):
            db.execute("INSERT INTO legal_transcript_segments (id, capture_id, meeting_id, matter_id, seq, text, created_at) "
                       "VALUES (?,?,?,?,?,?,?)", (f"seg{i}", "cap1", "mtg1", mid, i, text, f"2026-11-02T10:0{i + 1}:00.000Z"))


def test_the_live_transcript_card_follows_the_meeting_and_never_records(cs, fresh_db):
    import asyncio
    from app.legal.meetings import capture
    alice, carol = as_user(cs, "alice"), as_user(cs, "carol")
    _meeting(fresh_db, cs["mid"], cs["users"]["alice"]["id"])
    conv = new_conversation(alice, cs["mid"])
    comp = card(final(turn(alice, conv, "Show the live transcript")), "live_transcript")
    p = comp["props"]
    assert p["state"] == "not_recording" and [s["text"] for s in p["segments"]][-1] == "Apex says the twelfth was dispatch."
    assert p["path"] == "/m/M-101/meetings/mtg1"
    h = capture.Handle("mtg1", asyncio.new_event_loop())
    capture._register(h)
    try:
        assert read_card(alice, comp["id"])["props"]["state"] == "recording"
    finally:
        capture._unregister(h)
    assert card(final(turn(carol, new_conversation(carol, cs["mid"]), "Show the live transcript")), "live_transcript")
    with sqlite3.connect(fresh_db) as db:
        db.execute("UPDATE legal_meetings SET transcript_deleted_at='2026-11-02T11:00:00.000Z' WHERE id='mtg1'")
    assert read_card(alice, comp["id"])["type"] == "unavailable"           # a deleted transcript leaves the card too
    env = final(turn(alice, conv, "Show the live transcript"))
    assert "No meeting of M-101 is being recorded" in card(env, "notice")["props"]["text"]


# ── reading a card, and the contract ─────────────────────────────────────────────
def test_a_card_is_read_only_by_its_owner_in_a_conversation_they_can_still_open(cs):
    alice, carol = as_user(cs, "alice"), as_user(cs, "carol")
    conv = new_conversation(alice, cs["mid"])
    comp = card(final(turn(alice, conv, "Chart the amounts in the documents")), "numeric_chart")
    assert carol.get(f"/api/legal/counsel/components/{comp['id']}").status_code == 404
    assert alice.delete(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(alice)).status_code == 204
    assert alice.get(f"/api/legal/counsel/components/{comp['id']}").status_code == 404


def test_the_new_cards_hold_to_the_contract():
    from app.legal.counsel import schemas
    side = {"document_id": "d", "document_title": "D", "version_id": "v", "value": "USD 1", "context": "pay USD 1", "value_start": 4, "value_end": 9}
    with pytest.raises(ValidationError):
        schemas.ChecklistProps.model_validate({"summary": "s", "items": [{"id": "o1", "title": "t", "status": "proposed", "document_title": "D",
                                                                            "path": "https://evil.test/x", "version": 1}],
                                               "can_done": True, "can_confirm": True, "caution": "c", "register_path": "/m/M-101/obligations"})
    with pytest.raises(ValidationError):
        schemas.GuidedQuestion.model_validate({"id": "template", "label": "T", "kind": "choice", "options": [{"value": "a", "label": "A"}], "default": "b"})
    with pytest.raises(ValidationError):
        schemas.GuidedQuestionsProps.model_validate({"purpose": "draft", "title": "t", "intro": "i", "submit_label": "s", "note": "n",
                                                     "questions": [{"id": f"q{i}", "label": "Q", "kind": "text"} for i in range(4)]})
    with pytest.raises(ValidationError):
        schemas.GraphExcerptProps.model_validate({"summary": "s", "caution": "c", "nodes": [{"id": "p1", "label": "A", "kind": "party"}],
                                                  "edges": [{"id": "e1", "source": "p1", "target": "d9", "label": "named in", "evidence": side}]})
    with pytest.raises(ValidationError):
        schemas.NumericChartProps.model_validate({"summary": "s", "unit": "USD", "caution": "c",
                                                  "bars": [{"id": "b", "shown": "x", "value": -1, "document_title": "D", "source": side}]})
    with pytest.raises(ValidationError):
        schemas.validate_component({"id": "cmp_" + "0" * 32, "type": "checklist", "props": {}})          # a live card names its data
    with pytest.raises(ValidationError):
        schemas.GuidedOption.model_validate({"value": "<script>", "label": "x"})
