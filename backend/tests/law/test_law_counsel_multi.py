"""EchoMind Legal chat about several matters at once (owner's request, 2026-10-28).

A conversation may cover up to five of the person's matters. Each one is authorized when the conversation starts, on every
read and on every turn; losing any one closes the conversation. Answers search each matter under its own scope and name the
matter of every source; the conflict scan runs inside each matter only; the briefing and the chronology label every item;
a table shows the matters side by side from their records. A highly restricted matter is only ever discussed on its own.
A conversation about one matter is unchanged.
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic import ValidationError

from conftest import as_user, csrf_headers, make_user
from test_law_counsel import cs, component, final, names, new_conversation, turn  # noqa: F401  (cs is a fixture)


def several(c, matter_ids, status=201):
    r = c.post("/api/legal/counsel/conversations", headers=csrf_headers(c), json={"matter_ids": matter_ids})
    assert r.status_code == status, r.text
    return r.json()["conversation"]["id"] if status == 201 else r


@pytest.fixture
def two(cs):
    return {**cs, "m101": cs["matters"]["M-101"]["id"], "m114": cs["matters"]["M-114"]["id"]}


def test_one_answer_from_two_matters_names_the_matter_of_every_source_and_stays_out_of_each_matter_s_history(two, fresh_db):
    alice = as_user(two, "alice")
    two["model"].claim, two["model"].quote = "The pre-money valuation is USD 40 million", "pre-money valuation is USD 40 million"
    conv = several(alice, [two["m101"], two["m114"]])
    read = alice.get(f"/api/legal/counsel/conversations/{conv}").json()["conversation"]
    assert read["matter_ids"] == [two["m101"], two["m114"]]
    events = turn(alice, conv, "What is the pre-money valuation?")
    assert next(e for e in events if e["event"] == "turn.started")["data"]["matter_ids"] == [two["m101"], two["m114"]]
    env = final(events)
    assert [m["number"] for m in env["matters"]] == ["M-101", "M-114"]
    cite = component(env, "source_answer")["props"]["citations"][0]
    assert cite["matter"] == "M-114" and cite["verified"] is True
    opened = alice.get(f"/api/legal/counsel/citations/{cite['id']}").json()["citation"]
    assert opened["matter_id"] == two["m114"] and "pre-money valuation is USD 40 million" in opened["quote"]
    prompt = two["model"].stream_calls[-1][-1]["content"]
    assert 'matter="M-114"' in prompt                                     # every source tells the model its matter
    with sqlite3.connect(fresh_db) as c:                                  # nothing from M-114 lands in M-101's research history
        assert c.execute("SELECT COUNT(*) FROM legal_research_turns WHERE question LIKE '%pre-money%'").fetchone()[0] == 0


def test_a_conversation_about_one_matter_is_unchanged(two):
    alice = as_user(two, "alice")
    two["model"].claim, two["model"].quote = "Delivery is due by May 15, 2026", "complete delivery by May 15, 2026"
    conv = new_conversation(alice, two["m101"])
    env = final(turn(alice, conv, "When must Apex complete delivery?"))
    assert "matters" not in env and "matter" not in component(env, "source_answer")["props"]["citations"][0]
    assert 'matter="' not in two["model"].stream_calls[-1][-1]["content"]


def test_every_matter_is_the_person_s_own_and_a_highly_restricted_one_stays_alone(two, fresh_db):
    alice = as_user(two, "alice")
    walled = two["matters"]["M-202"]["id"]
    assert several(alice, [two["m101"], walled], 404).status_code == 404          # not hers: as if it did not exist
    r = alice.post("/api/legal/counsel/conversations", headers=csrf_headers(alice), json={"matter_ids": [two["m101"]] * 6})
    assert r.status_code == 422                                                    # at most five
    with sqlite3.connect(fresh_db) as c:
        c.execute("UPDATE matters SET sensitivity='highly_restricted' WHERE id=?", (two["m114"],))
    r = several(alice, [two["m101"], two["m114"]], 409)
    assert "only be discussed on its own" in r.json()["detail"]
    several(alice, [two["m114"]])                                                  # on its own: as before


def test_losing_any_one_matter_closes_the_whole_conversation(two, fresh_db):
    alice = as_user(two, "alice")
    conv = several(alice, [two["m101"], two["m114"]])
    assert [m["number"] for m in alice.get("/api/legal/counsel/conversations").json()["conversations"][0]["matters"]] == ["M-101", "M-114"]
    with sqlite3.connect(fresh_db) as c:                                  # her membership of M-114 ends
        c.execute("UPDATE matter_memberships SET valid_to='2000-01-01T00:00:00.000Z' WHERE matter_id=? AND user_id=?",
                  (two["m114"], two["users"]["alice"]["id"]))
    assert alice.get(f"/api/legal/counsel/conversations/{conv}").status_code == 404
    assert conv not in [c["id"] for c in alice.get("/api/legal/counsel/conversations").json()["conversations"]]
    r = alice.post(f"/api/legal/counsel/conversations/{conv}/turns", headers=csrf_headers(alice), json={"text": "Anything new?"})
    assert r.status_code == 404


def test_conflicts_are_found_inside_each_matter_never_across_and_resolved_under_their_own_matter(two):
    alice = as_user(two, "alice")
    conv = several(alice, [two["m114"], two["m101"]])                    # the witnesses' matter is not the conversation's own
    events = turn(alice, conv, "Find conflicting delivery dates")
    env = final(events)
    card = component(env, "evidence_comparison")
    assert card["matter_id"] == two["m101"] and {card["props"]["a"]["value"], card["props"]["b"]["value"]} >= {"May 12, 2026"}
    assert any(e["event"] == "task.progress" and e["data"]["label"].startswith("M-101 (2 of 2)") for e in events)
    for call in two["model"].chat_calls:                                  # no check ever puts two matters' documents together
        text = call[-1]["content"]
        assert not ("ZEPHYR-CANARY-114" in text and "Witness" in text)
    body = {"component_id": card["id"], "action_id": "review", "expected_version": card["version"], "idempotency_key": "multi-review-1",
            "parameters": {"resolution": "explained", "note": "Dispatch and arrival."}}
    wrong = alice.post("/api/legal/counsel/actions", headers=csrf_headers(alice), json={**body, "matter_id": two["m114"]})
    assert wrong.status_code == 404                                        # an action names the card's own matter
    ok = alice.post("/api/legal/counsel/actions", headers=csrf_headers(alice), json={**body, "matter_id": two["m101"]})
    assert ok.status_code == 200 and ok.json()["component"]["props"]["status"] == "explained"
    assert ok.json()["component"]["matter_id"] == two["m101"]
    again = alice.get(f"/api/legal/counsel/conversations/{conv}").json()["turns"][-1]["envelope"]
    assert component(again, "evidence_comparison")["matter_id"] == two["m101"]


def test_the_chronology_and_the_briefing_label_every_item_with_its_matter(two):
    alice = as_user(two, "alice")
    conv = several(alice, [two["m101"], two["m114"]])
    events = turn(alice, conv, "Build a chronology of events")
    seq = names(events)
    assert seq.index("ui.partial") < seq.index("ui.final")                 # the card is shown as soon as it is ready
    tl = component(final(events), "timeline")["props"]
    assert tl["events"] and all(e["matter"] in ("M-101", "M-114") for e in tl["events"])
    assert {e["matter"] for e in tl["events"] if "May 15, 2026" in e["date_text"]} == {"M-101"}
    assert "in 2 matters" in tl["summary"]
    brief = component(final(turn(alice, conv, "What matters today?")), "priority_list")["props"]
    assert "2 matters" in brief["summary"] and all(i["matter"] in ("M-101", "M-114") for i in brief["items"])
    assert len(brief["items"]) <= 3


def test_the_matters_side_by_side_from_their_records(two):
    alice = as_user(two, "alice")
    conv = several(alice, [two["m101"], two["m114"]])
    events = turn(alice, conv, "Compare these cases")
    table = component(final(events), "case_table")["props"]
    rows = {r["number"]: r for r in table["rows"]}
    assert list(rows) == ["M-101", "M-114"] and rows["M-101"]["documents"] == 4 and rows["M-114"]["documents"] == 1
    assert rows["M-101"]["path"] == "/m/M-101/overview" and rows["M-101"]["client"]
    assert "ui.partial" in names(events)


def test_other_matters_says_what_this_conversation_covers(two):
    alice = as_user(two, "alice")
    single = new_conversation(alice, two["m101"])
    text = component(final(turn(alice, single, "Let other case materials inform this")), "notice")["props"]["text"]
    assert "uses only the documents in M-101" in text and "choose up to five" in text
    both = several(alice, [two["m101"], two["m114"]])
    text = component(final(turn(alice, both, "Let other case materials inform this")), "notice")["props"]["text"]
    assert "covers M-101, M-114" in text


def test_the_contract_for_several_matters():
    from app.legal.counsel import schemas
    row = {"matter_id": "m1", "number": "M-1", "title": "t", "client": "c", "status": "open", "sensitivity": "standard",
           "documents": 1, "path": "/m/M-1/overview"}
    ok = {"id": "cmp_" + "a" * 32, "type": "case_table", "props": {"summary": "s", "caution": "c", "rows": [row]}}
    assert schemas.validate_component(ok)["type"] == "case_table"
    with pytest.raises(ValidationError):                                   # a row opens only the workspace's own page
        schemas.validate_component({**ok, "props": {**ok["props"], "rows": [{**row, "path": "https://evil.example/x"}]}})
    with pytest.raises(ValidationError):
        schemas.validate_component({**ok, "props": {**ok["props"], "rows": [{**row, "script": "x"}]}})
    env = {"message_id": "t1", "matter_id": "m1", "components": [],
           "matters": [{"id": f"m{i}", "number": f"M-{i}", "title": "t"} for i in range(6)]}
    with pytest.raises(ValidationError):                                   # at most five matters
        schemas.validate_envelope(env)
    assert len(schemas.validate_envelope({**env, "matters": env["matters"][:5]})["matters"]) == 5


def test_purging_a_document_reaches_the_copies_kept_in_a_conversation_about_several_matters(two, fresh_db):
    """Commercial-readiness review F2 (2026-10-28): an answer quoting M-114 is stored under the conversation's first matter
    (M-101). Purging the M-114 document removes its title and quote from that answer too, and the purge's own check says so."""
    two["users"]["rhea"] = make_user("rhea", mfa=True, firm_roles=["records_admin"])
    alice, rhea = as_user(two, "alice"), as_user(two, "rhea")
    two["model"].claim, two["model"].quote = "The pre-money valuation is USD 40 million", "pre-money valuation is USD 40 million"
    conv = several(alice, [two["m101"], two["m114"]])
    cite = component(final(turn(alice, conv, "What is the pre-money valuation?")), "source_answer")["props"]["citations"][0]
    did = cite["document_id"]
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT matter_id FROM legal_documents WHERE id=?", (did,)).fetchone()[0] == two["m114"]
        assert c.execute("SELECT matter_id FROM legal_counsel_components WHERE type='source_answer'").fetchone()[0] == two["m101"]
    d = alice.post(f"/api/legal/matters/{two['m114']}/documents/{did}/delete", headers=csrf_headers(alice),
                   json={"reason": "Sent to us by mistake"})
    assert d.status_code == 200, d.text
    r = rhea.post(f"/api/legal/deletions/{d.json()['deletion']['id']}/approve", headers=csrf_headers(rhea))
    assert r.status_code == 200, r.text
    assert r.json()["deletion"]["verification"]["ok"] is True
    chips = component(alice.get(f"/api/legal/counsel/conversations/{conv}").json()["turns"][0]["envelope"], "source_answer")["props"]["citations"]
    assert chips[0]["removed"] is True and chips[0]["document_title"] == ""
    assert alice.get(f"/api/legal/counsel/citations/{cite['id']}").status_code == 404
    with sqlite3.connect(fresh_db) as c:
        from app.legal.counsel import purge
        assert purge.remaining(c, two["m114"], did) == 0
