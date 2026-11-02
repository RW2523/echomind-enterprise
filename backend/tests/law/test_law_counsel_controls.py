"""EchoMind Legal chat controls (owner's request 2026-11-02, "like ChatGPT"; migration 0039).

Asking again or with the question edited adds another version of that turn (a sibling in the conversation's tree), so
every answer that was shown stays on record, and a follow-up only ever builds on its own path. A person can rename a
conversation, take it off their list (kept as work product), search their own conversations, and rate an answer. Each
works only on the caller's own conversation in matters they can still open; anything else is the same 404.
"""
from __future__ import annotations

import sqlite3

import pytest

from conftest import as_user, csrf_headers
from test_law_counsel import cs, events_of, final, new_conversation, turn  # noqa: F401  (cs is a fixture)

FEES = ("Morgan Company LLC shall pay USD 120,000", "The fees are USD 120,000")
DELIVERY = ("Apex Industries, Inc. shall complete delivery by May 15, 2026", "Delivery is due by May 15, 2026")


def answer(cs, quote_claim):
    cs["model"].quote, cs["model"].claim = quote_claim


def started(events):
    return next(e["data"] for e in events if e["event"] == "turn.started")


def read(c, conv):
    r = c.get(f"/api/legal/counsel/conversations/{conv}")
    assert r.status_code == 200, r.text
    return r.json()["turns"]


def test_asking_again_adds_a_version_and_keeps_the_first(cs):
    alice = as_user(cs, "alice")
    conv = new_conversation(alice, cs["mid"])
    answer(cs, DELIVERY)
    first = started(turn(alice, conv, "When is delivery due?"))
    again = turn(alice, conv, "When is delivery due?", retry_of=first["turn_id"])
    assert started(again)["parent_turn_id"] is None and final(again)["components"][0]["props"]["state"] == "answered"
    turns = read(alice, conv)
    assert [(t["question"], t["parent_turn_id"]) for t in turns] == [("When is delivery due?", None)] * 2
    assert all(t["status"] == "completed" and t["envelope"] for t in turns)        # both answers stay readable

    # A question asked after the second version follows it; asked after the first, it starts another path there.
    nxt = started(turn(alice, conv, "Who pays the fees?", after_turn_id=turns[1]["id"]))
    assert nxt["parent_turn_id"] == turns[1]["id"]
    side = started(turn(alice, conv, "Who pays the fees?", after_turn_id=turns[0]["id"]))
    assert side["parent_turn_id"] == turns[0]["id"]
    # Without either, a question follows the newest turn, as before versions existed.
    assert started(turn(alice, conv, "And the date?"))["parent_turn_id"] == side["turn_id"]


def test_an_edited_question_is_a_version_and_a_follow_up_uses_only_its_own_path(cs):
    alice = as_user(cs, "alice")
    conv = new_conversation(alice, cs["mid"])
    answer(cs, DELIVERY)
    first = started(turn(alice, conv, "When is delivery due?"))
    answer(cs, FEES)
    edited = started(turn(alice, conv, "What are the fees?", retry_of=first["turn_id"]))
    assert edited["parent_turn_id"] is None
    cs["model"].standalone = "What are the fees and when are they paid?"
    cs["model"].chat_calls.clear()
    turn(alice, conv, "And when are they paid?", after_turn_id=edited["turn_id"])
    rewrite = cs["model"].chat_calls[0][-1]["content"]                  # the follow-up rewriting saw only this path
    assert "What are the fees?" in rewrite and "delivery" not in rewrite.lower()


def test_versions_and_follow_ups_only_point_inside_the_callers_own_conversation(cs):
    alice, bob = as_user(cs, "alice"), as_user(cs, "carol")      # carol: another member of M-101
    answer(cs, DELIVERY)
    mine = new_conversation(alice, cs["mid"])
    t = started(turn(alice, mine, "When is delivery due?"))
    other = new_conversation(alice, cs["mid"])
    for field in ("retry_of", "after_turn_id"):
        r = alice.post(f"/api/legal/counsel/conversations/{other}/turns", headers=csrf_headers(alice),
                       json={"text": "When is delivery due?", field: t["turn_id"]})
        assert r.status_code == 404, (field, r.text)
        r = alice.post(f"/api/legal/counsel/conversations/{mine}/turns", headers=csrf_headers(alice),
                       json={"text": "When?", field: "turn_" + "0" * 32})
        assert r.status_code == 404
    r = alice.post(f"/api/legal/counsel/conversations/{mine}/turns", headers=csrf_headers(alice),
                   json={"text": "When?", "retry_of": t["turn_id"], "after_turn_id": t["turn_id"]})
    assert r.status_code == 422
    bobs = new_conversation(bob, cs["mid"])
    r = bob.post(f"/api/legal/counsel/conversations/{bobs}/turns", headers=csrf_headers(bob),
                 json={"text": "When is delivery due?", "retry_of": t["turn_id"]})
    assert r.status_code == 404


def test_rename(cs, fresh_db):
    alice, bob = as_user(cs, "alice"), as_user(cs, "carol")      # carol: another member of M-101
    conv = new_conversation(alice, cs["mid"])
    answer(cs, DELIVERY)
    turn(alice, conv, "When is delivery due?")
    r = alice.patch(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(alice), json={"title": "  Delivery   date  "})
    assert r.status_code == 200 and r.json()["conversation"]["title"] == "Delivery date"
    assert alice.get("/api/legal/counsel/conversations").json()["conversations"][0]["title"] == "Delivery date"
    turn(alice, conv, "Who pays?")                                       # a later question does not rename it back
    assert alice.get("/api/legal/counsel/conversations").json()["conversations"][0]["title"] == "Delivery date"
    assert alice.patch(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(alice), json={"title": "   "}).status_code == 422
    assert alice.patch(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(alice), json={"title": "x" * 81}).status_code == 422
    assert alice.patch(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(alice),
                       json={"title": "A", "owner_id": "bob"}).status_code == 422
    assert bob.patch(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(bob), json={"title": "Mine"}).status_code == 404
    assert events_of(fresh_db, "counsel.conversation.rename")[0]["outcome"] == "allow"


def test_delete_takes_it_off_the_list_and_closes_it_but_keeps_the_record(cs, fresh_db):
    alice, bob = as_user(cs, "alice"), as_user(cs, "carol")      # carol: another member of M-101
    conv = new_conversation(alice, cs["mid"])
    answer(cs, DELIVERY)
    t = started(turn(alice, conv, "When is delivery due?"))
    assert bob.delete(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(bob)).status_code == 404
    assert alice.delete(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(alice)).status_code == 204
    assert alice.get("/api/legal/counsel/conversations").json()["conversations"] == []
    assert alice.get(f"/api/legal/counsel/conversations/{conv}").status_code == 404
    assert alice.post(f"/api/legal/counsel/conversations/{conv}/turns", headers=csrf_headers(alice), json={"text": "Again?"}).status_code == 404
    assert alice.put(f"/api/legal/counsel/turns/{t['turn_id']}/feedback", headers=csrf_headers(alice),
                     json={"rating": "up"}).status_code == 404
    assert alice.delete(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(alice)).status_code == 404
    with sqlite3.connect(fresh_db) as db:                               # work product: the rows stay
        assert db.execute("SELECT deleted_at IS NOT NULL FROM legal_conversations WHERE id=?", (conv,)).fetchone() == (1,)
        assert db.execute("SELECT COUNT(*) FROM legal_conversation_turns WHERE conversation_id=?", (conv,)).fetchone() == (1,)
    assert [e["outcome"] for e in events_of(fresh_db, "counsel.conversation.delete")] == ["deny", "allow", "deny"]


def test_search_finds_own_conversations_by_title_or_question_only(cs):
    alice, bob = as_user(cs, "alice"), as_user(cs, "carol")      # carol: another member of M-101
    answer(cs, DELIVERY)
    a = new_conversation(alice, cs["mid"])
    turn(alice, a, "When is delivery due under the services agreement?")
    turn(alice, a, "Is there a 100% late-delivery_fee?")
    b = new_conversation(alice, cs["mid"])
    turn(alice, b, "Who pays the fees?")
    alice.patch(f"/api/legal/counsel/conversations/{b}", headers=csrf_headers(alice), json={"title": "Payment terms"})
    bobs = new_conversation(bob, cs["mid"])
    turn(bob, bobs, "When is delivery due?")

    def found(c, q):
        r = c.get("/api/legal/counsel/conversations", params={"q": q})
        assert r.status_code == 200, r.text
        return r.json()["conversations"]

    hits = found(alice, "DELIVERY")
    assert [h["id"] for h in hits] == [a] and hits[0]["match"].startswith("When is delivery due")
    assert [h["id"] for h in found(alice, "payment")] == [b] and "match" not in found(alice, "payment")[0]
    assert [h["id"] for h in found(alice, "100%")] == [a] and found(alice, "1%0") == []       # % and _ are literal
    assert [h["id"] for h in found(alice, "delivery_fee")] == [a] and found(alice, "late_d") == []
    assert [h["id"] for h in found(bob, "delivery")] == [bobs]
    assert found(alice, "zz-not-there") == []
    assert alice.get("/api/legal/counsel/conversations", params={"q": "x" * 101}).status_code == 422


def test_feedback_is_the_askers_own_and_says_nothing_of_the_note_in_the_audit(cs, fresh_db):
    alice, bob = as_user(cs, "alice"), as_user(cs, "carol")      # carol: another member of M-101
    conv = new_conversation(alice, cs["mid"])
    answer(cs, DELIVERY)
    t = started(turn(alice, conv, "When is delivery due?"))["turn_id"]
    url = f"/api/legal/counsel/turns/{t}/feedback"
    assert alice.put(url, headers=csrf_headers(alice), json={"rating": "up"}).json() == {"feedback": {"rating": "up"}}
    assert read(alice, conv)[0]["feedback"] == {"rating": "up"}
    r = alice.put(url, headers=csrf_headers(alice), json={"rating": "down", "reason": "wrong_source", "note": "Clause 4.2 is the wrong one"})
    assert r.json() == {"feedback": {"rating": "down", "reason": "wrong_source"}}
    assert read(alice, conv)[0]["feedback"] == {"rating": "down", "reason": "wrong_source"}
    assert alice.put(url, headers=csrf_headers(alice), json={"rating": "up", "reason": "missed"}).status_code == 422
    assert alice.put(url, headers=csrf_headers(alice), json={"rating": "down", "reason": "rude"}).status_code == 422
    assert alice.put(url, headers=csrf_headers(alice), json={"rating": "down", "note": "x" * 501}).status_code == 422
    assert bob.put(url, headers=csrf_headers(bob), json={"rating": "up"}).status_code == 404
    assert alice.put(url, headers=csrf_headers(alice), json={"rating": None}).json() == {"feedback": None}
    assert "feedback" not in read(alice, conv)[0]
    with sqlite3.connect(fresh_db) as db:
        assert db.execute("SELECT COUNT(*) FROM legal_counsel_feedback").fetchone() == (0,)
    audit = events_of(fresh_db, "counsel.feedback")
    assert len(audit) >= 3 and all("Clause 4.2" not in (e["details"] or "") for e in audit)
    assert '"note_chars":27' in audit[1]["details"].replace(" ", "")


def test_a_running_answer_cannot_be_rated(cs, fresh_db):
    alice = as_user(cs, "alice")
    conv = new_conversation(alice, cs["mid"])
    answer(cs, DELIVERY)
    t = started(turn(alice, conv, "When is delivery due?"))["turn_id"]
    with sqlite3.connect(fresh_db) as db:
        db.execute("UPDATE legal_conversation_turns SET status='failed' WHERE id=?", (t,))
    assert alice.put(f"/api/legal/counsel/turns/{t}/feedback", headers=csrf_headers(alice), json={"rating": "up"}).status_code == 409


def test_a_member_who_lost_the_matter_can_no_longer_rename_delete_or_rate(cs):
    from app.legal.matters import repo
    carol = as_user(cs, "carol")
    conv = new_conversation(carol, cs["mid"])
    answer(cs, DELIVERY)
    t = started(turn(carol, conv, "When is delivery due?"))["turn_id"]
    m = next(x for x in repo.list_live_memberships(cs["mid"]) if x["user_id"] == cs["users"]["carol"]["id"])
    repo.revoke_membership(cs["mid"], m["id"], by=cs["users"]["alice"]["id"], reason="left the team")
    assert carol.patch(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(carol), json={"title": "X"}).status_code == 404
    assert carol.delete(f"/api/legal/counsel/conversations/{conv}", headers=csrf_headers(carol)).status_code == 404
    assert carol.put(f"/api/legal/counsel/turns/{t}/feedback", headers=csrf_headers(carol), json={"rating": "up"}).status_code == 404
    assert carol.get("/api/legal/counsel/conversations", params={"q": "delivery"}).json()["conversations"] == []


def test_migration_0039_makes_existing_conversations_one_line_and_refuses_to_drop_versions(tmp_path):
    from app.legal.migrations import runner
    conn = sqlite3.connect(tmp_path / "m.sqlite")
    runner.migrate_up(conn, target=38)
    conn.execute("INSERT INTO legal_conversations (id, org_id, matter_id, owner_id, title, created_at, updated_at) "
                 "VALUES ('c','o','m','u','t','t','t')")
    for i, tid in enumerate(("t1", "t2", "t3"), start=1):
        conn.execute("INSERT INTO legal_conversation_turns (id, conversation_id, matter_id, owner_id, seq, question, intent, status, "
                     "created_at) VALUES (?,?,?,?,?,?,?,?,?)", (tid, "c", "m", "u", i, "q", "ask_fact", "completed", "t"))
    conn.commit()
    runner.migrate_up(conn)
    assert conn.execute("SELECT id, parent_turn_id FROM legal_conversation_turns ORDER BY seq").fetchall() == [
        ("t1", None), ("t2", "t1"), ("t3", "t2")]
    conn.execute("INSERT INTO legal_conversation_turns (id, conversation_id, matter_id, owner_id, seq, question, intent, status, "
                 "created_at, parent_turn_id) VALUES ('t4','c','m','u',4,'q','ask_fact','completed','t','t1')")   # a version of t2
    conn.commit()
    with pytest.raises(runner.MigrationError, match="version"):
        runner.migrate_down(conn, 38)
    conn.execute("DELETE FROM legal_conversation_turns WHERE id='t4'")
    conn.commit()
    runner.migrate_down(conn, 38)
    assert "parent_turn_id" not in [r[1] for r in conn.execute("PRAGMA table_info(legal_conversation_turns)")]
    conn.close()
