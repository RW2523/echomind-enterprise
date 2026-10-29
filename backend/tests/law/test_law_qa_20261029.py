"""The 2026-10-29 functional QA findings, as regression tests (EchoMind Legal fix brief: RAG-001, RAG-002, RAG-003, UX-001,
CHAT-001, and the isolation check).

The testers imported a fictional sample agreement (sample_msa_test.txt, kept in eval/law/qa_20261029/) into M-101 and
asked through the chat. Here the same file goes into M-101 and the model is scripted with the answers the testers saw
(their displayed text, as saved by the demo, with the quotes in the model's own [n: "…"] form), so each finding is checked
end to end through the chat endpoint: citations, verification, the unsupported list, the decline, the calculated dates,
the document list, and that nothing of M-101 reaches M-114 or a person without access.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from conftest import as_user, csrf_headers, index_all, upload

FIXTURE = (Path(__file__).resolve().parents[3] / "eval" / "law" / "qa_20261029" / "sample_msa_test.txt").read_text()
TERMINATE = "Either party may terminate this agreement by providing 60 days' written notice"


class Model:
    """Answers with the text a test sets, in pieces."""

    def __init__(self):
        self.answer, self.calls = "", 0

    async def chat_stream(self, messages, temperature, max_tokens):
        self.calls += 1
        for piece in re.findall(r"\S+\s*|\n+", self.answer):
            yield piece

    async def chat(self, messages, temperature, max_tokens):
        return "?"


@pytest.fixture
def qa(docs, monkeypatch):
    from app.core.config import settings
    from app.legal.research import llm
    monkeypatch.setattr(settings, "LEGAL_STREAM_RECHECK_SEC", 0.0)
    monkeypatch.setattr(settings, "LEGAL_RERANK_MIN_SCORE", -1.0)        # the stand-in reranker scores a short file low
    model = Model()
    monkeypatch.setattr(llm, "_override", model)
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    r = upload(alice, mid, "sample_msa_test.txt", FIXTURE, status="draft", title="sample_msa_test")
    assert r.status_code == 202, r.text
    index_all()
    docs.update(mid=mid, model=model, doc=r.json()["document"])
    return docs


def chat(c, matter_id, text):
    r = c.post("/api/legal/counsel/conversations", headers=csrf_headers(c), json={"matter_id": matter_id})
    assert r.status_code == 201, r.text
    t = c.post(f"/api/legal/counsel/conversations/{r.json()['conversation']['id']}/turns", headers=csrf_headers(c), json={"text": text})
    assert t.status_code == 200, t.text
    events = [json.loads(line) for line in t.text.splitlines() if line.strip()]
    return next(e["data"] for e in events if e["event"] == "ui.final")


def card(env, type_="source_answer"):
    return next(c["props"] for c in env["components"] if c["type"] == type_)


def test_the_fixture_is_read_with_its_six_numbered_clauses(qa):
    alice = as_user(qa, "alice")
    docs = alice.get(f"/api/legal/matters/{qa['mid']}/documents").json()["documents"]
    assert [d["title"] for d in docs] == ["sample_msa_test"] and docs[0]["status"] == "draft"
    vid = alice.get(f"/api/legal/matters/{qa['mid']}/documents/{qa['doc']['id']}/versions").json()["versions"][0]["id"]
    reader = alice.get(f"/api/legal/matters/{qa['mid']}/documents/{qa['doc']['id']}/versions/{vid}/reader").json()["reader"]
    assert [(c["number"], c["heading"]) for c in reader["clauses"]] == [
        ("1", "TERM"), ("2", "PAYMENT"), ("3", "TERMINATION"), ("4", "CONFIDENTIALITY"), ("5", "GOVERNING LAW"), ("6", "RENEWAL")]
    assert reader["version"]["index_status"] == "indexed"


def test_rag001_payment_is_supported_and_the_pointer_to_section_2_is_not_flagged(qa):
    qa["model"].answer = ('The monthly payment amount specified in the Master Services Agreement is $12,000 [1: "The Client will pay $12,000 '
                          'per month"]. The payment deadline is 30 days from the invoice date [1: "Invoices must be paid within 30 days"]. '
                          "These details are found in Section 2 of the agreement.")
    p = card(chat(as_user(qa, "alice"), qa["mid"], "What is the monthly payment and when must invoices be paid?"))
    assert p["state"] == "answered" and p["unsupported"] == []
    assert [(c["clause"], c["verified"]) for c in p["citations"]] == [("2", True)]


def test_rag002_the_termination_quote_under_3_is_verified_against_clause_3_and_opens_there(qa):
    qa["model"].answer = (f'The termination notice period in the Master Services Agreement is 60 days. This is stated in the source as: '
                          f'"{TERMINATE}" [3: "{TERMINATE}"].\n\nThe documents also indicate that either party can terminate the agreement, '
                          f'as the clause states: "{TERMINATE}" [3: "{TERMINATE}"].')
    alice = as_user(qa, "alice")
    p = card(chat(alice, qa["mid"], "What is the termination notice period, and can either party terminate? Cite the supporting clause."))
    assert p["state"] == "answered" and p["unsupported"] == []
    assert "[3]" not in p["answer"] and p["answer"].count("[1]") == 2
    assert [(c["n"], c["clause"], c["verified"]) for c in p["citations"]] == [(1, "3", True)]
    opened = alice.get(f"/api/legal/counsel/citations/{p['citations'][0]['id']}").json()
    assert TERMINATE in json.dumps(opened)


def test_rag003_term_end_and_notice_deadline_are_calculated_from_clauses_1_and_6(qa):
    qa["model"].answer = ('The initial 12-month term end date is not specified in the provided text [1: "GOVERNING LAW"].\n\n'
                          "The deadline for providing notice of non-renewal is at least 45 days before the current term ends "
                          '[1: "at least 45 days before the current term ends"].\n\nIf neither party provides notice, the agreement '
                          'automatically renews for an additional 12 months [1: "automatically renews for an additional 12 months"].')
    p = card(chat(as_user(qa, "alice"), qa["mid"], "When does the term end, what is the non-renewal deadline, what happens on renewal, "
                                                     "and what assumptions apply?"))
    assert all(c["clause"] != "5" or not c["verified"] for c in p["citations"])          # "GOVERNING LAW" supports nothing here
    calc = p["calculation"]
    assert {d["label"]: d["display"] for d in calc["derived"]} == {
        "Current term ends": "31 January 2027", "Last day to give notice of non-renewal": "17 December 2026",
        "If no notice is given, the next term": "1 February 2027 to 31 January 2028"}
    assert {f["label"]: f["clause"] for f in calc["facts"]} == {"Start": "1", "Term": "1", "Notice before the term ends": "6", "Renewal": "6"}
    assert all(f["path"].startswith(f"/m/M-101/documents/{qa['doc']['id']}/read?v=") for f in calc["facts"])
    said = " ".join(calc["assumptions"])
    assert "12 calendar months from and including 1 February 2026" in said and "45 calendar days counted back from 31 January 2027" in said


def test_ux001_the_absent_penalty_is_said_plainly_with_no_invented_fee_or_citation(qa):
    qa["model"].answer = ("The early termination penalty amount specified in the Master Services Agreement is not mentioned in the provided "
                          "sources.\n\nThe documents in this matter do not answer that.")
    env = chat(as_user(qa, "alice"), qa["mid"], "What is the early termination penalty?")
    p = card(env)
    assert p["state"] == "abstained" and p["citations"] == [] and p["unsupported"] == [] and "calculation" not in p
    assert p["absence"] == "The early termination penalty amount specified in the Master Services Agreement is not mentioned in the provided sources."
    assert not re.search(r"\$|\d", p["absence"])


def test_chat001_the_document_list_comes_from_the_records(qa):
    alice = as_user(qa, "alice")
    calls = qa["model"].calls
    assert card(chat(alice, qa["mid"], "Which documents are available in this matter?"), "notice")["text"] == (
        "M-101 has 1 document that you can open: sample_msa_test (contract, draft, version 1).")
    assert card(chat(alice, qa["matters"]["M-114"]["id"], "Which documents are available?"), "notice")["text"] == (
        "M-114 has no documents that you can open.")
    assert qa["model"].calls == calls


def test_isolation_m114_and_people_without_m101_see_nothing_of_the_fixture(qa):
    alice, dave = as_user(qa, "alice"), as_user(qa, "dave")
    other = qa["matters"]["M-114"]["id"]
    qa["model"].answer = 'The Client will pay $12,000 per month [1: "The Client will pay $12,000 per month"].'
    env = chat(alice, other, "What is the monthly payment and when must invoices be paid?")
    assert "12,000" not in json.dumps(env) and "sample_msa_test" not in json.dumps(env)
    r = alice.post(f"/api/legal/matters/{other}/research/query", headers=csrf_headers(alice), json={"question": "monthly payment $12,000"})
    assert "sample_msa_test" not in r.text and "12,000 per month" not in r.text
    assert "sample_msa_test" not in alice.get(f"/api/legal/matters/{other}/documents").text
    assert dave.post(f"/api/legal/matters/{qa['mid']}/research/query", headers=csrf_headers(dave), json={"question": "monthly payment"}).status_code == 404
    assert dave.get(f"/api/legal/matters/{qa['mid']}/documents").status_code in (403, 404)
    assert dave.post("/api/legal/counsel/conversations", headers=csrf_headers(dave), json={"matter_id": qa["mid"]}).status_code in (403, 404)
    p = card(chat(alice, qa["mid"], "What is the monthly payment?"))
    assert dave.get(f"/api/legal/counsel/citations/{p['citations'][0]['id']}").status_code == 404
    carol = as_user(qa, "carol")                                          # read-only on M-101: may read and research it
    assert "sample_msa_test" in carol.get(f"/api/legal/matters/{qa['mid']}/documents").text
