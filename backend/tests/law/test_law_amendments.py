"""Amendment-aware answers (docs/echomind-legal/IMPLEMENTATION_STATUS.md, live finding 1).

On the DGX Spark, "Which contractual deadline applied?" was answered "the amendment did not change the delivery deadline":
every quote was real, but the amendment's clause replacing clause 4.2 never reached the model. Now, when a cited clause is
replaced or changed by another document of the matter, the amending clause is added to the sources (found in the documents'
own words, never by the model), marked for the model, and the answer says whether the amendment is signed, when it says it
applies, and whether the answer used it. An answer that relies on wording a signed amendment replaced, without the amendment,
is never a plain "answered". Only documents the person may read, in the same matter, are used.
"""
from __future__ import annotations

import json
import re
import sqlite3

import pytest

from conftest import as_user, csrf_headers, index_all, make_user, upload

AGREEMENT = ("SERVICES AGREEMENT (fictional, DEMO / SYNTHETIC)\n\n"
             "This Services Agreement is dated April 1, 2026 and is made between Morgan Company LLC and Apex Industries, Inc.\n\n"
             "4.2 Delivery. Apex Industries, Inc. shall complete delivery by May 15, 2026.\n\n"
             "9.1 Fees. Morgan Company LLC shall pay USD 240,000 within thirty days of complete delivery.\n\n"
             "12.1 Amendments. No amendment to this Agreement is effective unless it is in writing and signed by both parties.")
AMENDMENT = ("AMENDMENT No. 1 TO THE SERVICES AGREEMENT (fictional, DEMO / SYNTHETIC)\n\n"
             "This Amendment No. 1 is made on May 14, 2026 between Morgan Company LLC and Apex Industries, Inc. "
             "It amends the Services Agreement dated April 1, 2026 between them.\n\n"
             "1.1 Clause 4.2 of the Services Agreement is deleted and replaced with: Apex Industries, Inc. shall complete "
             "delivery by May 20, 2026.\n\n"
             "2.1 This Amendment takes effect when both parties have signed it. Both parties signed it on May 14, 2026.\n\n"
             "2.2 All other terms of the Services Agreement remain in force.")
QUESTION = "When must Apex complete delivery?"
OLD = ("Apex must complete delivery by May 15, 2026", "complete delivery by May 15, 2026")
NEW = ("The amendment replaced that date with May 20, 2026", "complete delivery by May 20, 2026")


class Model:
    """Cites, for each (claim, quote), the source whose text holds the quote; records every prompt."""

    def __init__(self):
        self.parts, self.calls = [], []

    async def chat_stream(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        user = messages[-1]["content"]
        blocks = re.findall(r'<source n="(\d+)"[^>]*>(.*?)</source>', user, re.S)
        out = []
        for claim, quote in self.parts:
            n = next((k for k, text in blocks if quote in text), None)
            if n:
                out.append(f'{claim} [{n}: "{quote}"].')
        text = " ".join(out) or "The documents in this matter do not answer that."
        for piece in re.findall(r"\S+\s*|\n+", text):
            yield piece

    async def chat(self, messages, temperature, max_tokens):
        return "?"


@pytest.fixture
def am(docs, monkeypatch):
    from app.core.config import settings
    from app.legal.research import llm
    monkeypatch.setattr(settings, "LEGAL_STREAM_RECHECK_SEC", 0.0)
    model = Model()
    monkeypatch.setattr(llm, "_override", model)
    docs.update(mid=docs["matters"]["M-101"]["id"], model=model)
    return docs


def load(w, amendment_status="executed", amendment_matter=None):
    alice = as_user(w, "alice")
    agreement = upload(alice, w["mid"], "services-agreement.txt", AGREEMENT, status="executed").json()["document"]
    amendment = upload(alice, amendment_matter or w["mid"], "amendment-1.txt", AMENDMENT, status=amendment_status).json()["document"]
    index_all()
    return agreement, amendment


def ask(c, mid, question=QUESTION):
    r = c.post(f"/api/legal/matters/{mid}/research/query", headers=csrf_headers(c), json={"question": question})
    assert r.status_code == 200, r.text
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def done(events):
    return events[-1]


# ── Reading the documents' own words ─────────────────────────────────────────────
def test_changes_and_when_they_apply_are_read_from_the_documents_words():
    from app.legal.research import amendments
    [c] = amendments.changes(AMENDMENT)
    assert (c.clause, c.title, c.how) == ("4.2", "Services Agreement", "deleted and replaced")
    eff = amendments.effective(AMENDMENT)
    assert eff["kind"] == "signed" and eff["date"] == "2026-05-14" and eff["words"] == "Both parties signed it on May 14, 2026."
    within = "In clause 9.1 of the Services Agreement, “USD 240,000” is replaced by “USD 250,000”."
    assert [(c.clause, c.how) for c in amendments.changes(within)] == [("9.1", "replaced")]
    assert amendments.effective("This variation takes effect on 1 June 2026, as agreed.")["date"] == "2026-06-01"
    for not_an_amendment in (AGREEMENT,                                          # "no amendment … is effective unless …"
                             "Clause 4 of this Agreement may be amended only in writing.",
                             "The parties may agree that clause 4.2 of the Services Agreement is to be replaced later."):
        assert amendments.changes(not_an_amendment) == []
    assert amendments.effective("Nothing here gives a date.") is None


# ── Answers ──────────────────────────────────────────────────────────────────────
def test_the_amending_clause_reaches_the_model_and_an_answer_without_it_is_not_a_plain_answer(am, fresh_db):
    """The live failure, replayed: the model cites only the replaced wording. The amendment is now in its sources, and the
    answer comes back partial, saying what changed, from the documents."""
    agreement, amendment = load(am)
    alice = as_user(am, "alice")
    am["model"].parts = [OLD]
    events = ask(alice, am["mid"])
    prompt = am["model"].calls[-1][-1]["content"]
    assert re.search(r'amends="[^"]+ clause 4\.2" amendment="signed"', prompt)
    assert "complete delivery by May 20, 2026" in prompt
    sources = events[0]["citations"]
    marked = [s for s in sources if s.get("amends")]
    assert marked and marked[0]["document_id"] == amendment["id"] and marked[0]["amends"]["clause"] == "4.2"
    d = done(events)
    assert d["status"] == "partial"
    [a] = d["amendments"]
    assert (a["document_id"], a["clause"], a["cited"], a["amendment_cited"]) == (agreement["id"], "4.2", True, False)
    assert a["by"]["document_id"] == amendment["id"] and a["by"]["clause"] == "1.1" and a["by"]["signed"] is True
    assert a["by"]["effective"] == {"kind": "signed", "date": "2026-05-14", "words": "Both parties signed it on May 14, 2026."}
    with sqlite3.connect(fresh_db) as c:
        text = c.execute("SELECT text FROM legal_version_text WHERE version_id=?", (a["by"]["version_id"],)).fetchone()[0]
    assert "Clause 4.2 of the Services Agreement is deleted and replaced" in text[a["by"]["char_start"]:a["by"]["char_end"]]


def test_an_answer_citing_the_amendment_too_is_a_full_answer(am):
    load(am)
    alice = as_user(am, "alice")
    am["model"].parts = [OLD, NEW]
    d = done(ask(alice, am["mid"]))
    assert d["status"] == "answered"
    [a] = d["amendments"]
    assert a["cited"] and a["amendment_cited"]


def test_an_unsigned_amendment_is_shown_as_unsigned_and_does_not_mark_the_answer_down(am):
    load(am, amendment_status="draft")
    alice = as_user(am, "alice")
    am["model"].parts = [OLD]
    events = ask(alice, am["mid"])
    assert 'amendment="unsigned"' in am["model"].calls[-1][-1]["content"]
    d = done(events)
    assert d["status"] == "answered" and d["amendments"][0]["by"]["signed"] is False


def test_an_amendment_the_person_may_not_read_is_never_used(am):
    from app.legal.documents import repo
    _, amendment = load(am)
    repo.set_acl(amendment["id"], {am["users"]["alice"]["id"]})                # carol is on M-101 but may not read it
    carol = as_user(am, "carol")
    am["model"].parts = [OLD]
    events = ask(carol, am["mid"])
    assert "May 20, 2026" not in am["model"].calls[-1][-1]["content"]
    assert done(events)["amendments"] == [] and done(events)["status"] == "answered"
    assert amendment["id"] not in json.dumps(events)


def test_an_amendment_in_another_matter_is_never_linked(am):
    load(am, amendment_matter=am["matters"]["M-114"]["id"])
    alice = as_user(am, "alice")
    am["model"].parts = [OLD]
    events = ask(alice, am["mid"])
    assert "May 20, 2026" not in am["model"].calls[-1][-1]["content"] and done(events)["amendments"] == []


def test_history_keeps_the_amendment_and_a_purge_of_the_amendment_removes_it(am, fresh_db):
    _, amendment = load(am)
    am["users"]["rhea"] = make_user("rhea", mfa=True, firm_roles=["records_admin"])
    alice, rhea = as_user(am, "alice"), as_user(am, "rhea")
    am["model"].parts = [OLD]
    ask(alice, am["mid"])
    [turn] = alice.get(f"/api/legal/matters/{am['mid']}/research").json()["turns"]
    assert turn["status"] == "partial" and turn["amendments"][0]["by"]["document_id"] == amendment["id"]
    d = alice.post(f"/api/legal/matters/{am['mid']}/documents/{amendment['id']}/delete", headers=csrf_headers(alice),
                   json={"reason": "Duplicate of the signed copy"}).json()["deletion"]
    r = rhea.post(f"/api/legal/deletions/{d['id']}/approve", headers=csrf_headers(rhea))
    assert r.status_code == 200 and r.json()["deletion"]["verification"]["ok"] is True, r.text
    with sqlite3.connect(fresh_db) as c:
        stored = c.execute("SELECT amendments_json FROM legal_research_turns WHERE matter_id=?", (am["mid"],)).fetchone()[0]
    assert amendment["id"] not in stored


def test_the_chat_says_what_changed_and_offers_the_amending_clause(am):
    agreement, amendment = load(am)
    alice = as_user(am, "alice")
    am["model"].parts = [OLD]
    conv = alice.post("/api/legal/counsel/conversations", headers=csrf_headers(alice), json={"matter_id": am["mid"]}).json()["conversation"]["id"]
    r = alice.post(f"/api/legal/counsel/conversations/{conv}/turns", headers=csrf_headers(alice), json={"text": QUESTION})
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    env = next(e["data"] for e in events if e["event"] == "ui.final")
    ans = next(c for c in env["components"] if c["type"] == "source_answer")["props"]
    assert ans["state"] == "partial"
    note = ans["notes"][0]
    assert note.startswith("Clause 4.2 of ") and "was deleted and replaced by clause 1.1 of" in note and "(signed)" in note
    assert "It says: “Both parties signed it on May 14, 2026.”" in note and "relies on the earlier wording" in note
    [action] = env["suggested_actions"]
    assert action["id"] == "open_amendment" and action["path"].startswith(f"/m/M-101/documents/{amendment['id']}/read?v=")


def test_the_clause_saying_when_the_amendment_applies_is_a_source_too(am):
    load(am)
    alice = as_user(am, "alice")
    am["model"].parts = [OLD, NEW, ("It applies once both parties signed it", "Both parties signed it on May 14, 2026")]
    events = ask(alice, am["mid"])
    texts = [s["text"] for s in events[0]["citations"]]
    assert any("Both parties signed it on May 14, 2026." in t for t in texts)
    d = done(events)
    assert d["status"] == "answered" and len(d["evidence"]) == 3 and all(e["verification"] == "verbatim" for e in d["evidence"])
