"""A question that could mean more than one document or party (2026-11-07, external review point 10). The answer is still
given; the person is offered asks narrowed to one document or one party, and told why. Fictional documents only."""
from __future__ import annotations

import json
import re

import pytest

from conftest import as_user, csrf_headers, index_all, upload

NORTHWIND = ("SUPPLY AGREEMENT (fictional)\n\nThis Supply Agreement is made between Northwind Foods Ltd (the \"Supplier\") and "
             "Orrin Retail Ltd (the \"Customer\").\n\n5.1 Payment. The Customer shall pay each invoice within 30 days.")
TIDEWATER = ("SUPPLY AGREEMENT (fictional)\n\nThis Supply Agreement is made between Tidewater Produce Ltd (the \"Supplier\") and "
             "Orrin Retail Ltd (the \"Customer\").\n\n5.1 Payment. The Customer shall pay each invoice within 45 days.")
SERVICES = ("SERVICES AGREEMENT (fictional)\n\nThis Services Agreement is made between Apex Industries Ltd (the \"Provider\") and "
            "Orrin Retail Ltd (the \"Client\").\n\n3.1 Delivery. The Provider shall deliver the reports monthly.")
GUARANTEE = ("PARENT COMPANY GUARANTEE (fictional)\n\nThis Guarantee is made between Apex Holdings Ltd (the \"Guarantor\") and "
             "Orrin Retail Ltd (the \"Beneficiary\").\n\n2.1 Guarantee. The Guarantor guarantees the Provider's obligations.")


class Model:
    async def chat_stream(self, messages, temperature, max_tokens):
        user = messages[-1]["content"]
        m = re.search(r'<source n="(\d+)"[^>]*>(.*?)</source>', user, re.S)
        quote = re.search(r"(shall [a-z ]+)", m.group(2)).group(1) if m else ""
        text = f'The document provides that the party {quote} [{m.group(1)}: "{quote}"].' if quote else "The documents in this matter do not answer that."
        for piece in re.findall(r"\S+\s*|\n+", text):
            yield piece

    async def chat(self, messages, temperature=0.0, max_tokens=200):
        return "?"


@pytest.fixture
def amb(docs, monkeypatch):
    from app.core.config import settings
    from app.legal.research import llm
    monkeypatch.setattr(settings, "LEGAL_STREAM_RECHECK_SEC", 0.0)
    monkeypatch.setattr(settings, "LEGAL_RERANK_MIN_SCORE", -100.0)
    monkeypatch.setattr(llm, "_override", Model())
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    ids = {}
    for name, title, text in (("nw.txt", "Supply Agreement - Northwind (DEMO)", NORTHWIND), ("tw.txt", "Supply Agreement - Tidewater (DEMO)", TIDEWATER),
                              ("sa.txt", "Services Agreement (DEMO)", SERVICES), ("pg.txt", "Parent Company Guarantee (DEMO)", GUARANTEE)):
        ids[title] = upload(alice, mid, name, text, status="executed", title=title).json()["document"]["id"]
    index_all()
    docs.update(mid=mid, ids=ids)
    return docs


def turn(c, mid, text):
    conv = c.post("/api/legal/counsel/conversations", headers=csrf_headers(c), json={"matter_id": mid}).json()["conversation"]["id"]
    r = c.post(f"/api/legal/counsel/conversations/{conv}/turns", headers=csrf_headers(c), json={"text": text})
    assert r.status_code == 200, r.text
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    return next(e["data"] for e in events if e["event"] == "ui.final")


def test_two_documents_of_the_kind_asked_about_are_offered_one_at_a_time(amb):
    from app.legal.counsel import schemas
    alice = as_user(amb, "alice")
    env = turn(alice, amb["mid"], "What are the payment terms in the supply agreement?")
    schemas.validate_envelope(env)
    asks = [a for a in env["suggested_actions"] if a["id"].startswith("only_document_")]
    assert {tuple(a["document_ids"]) for a in asks} == {(amb["ids"]["Supply Agreement - Northwind (DEMO)"],),
                                                        (amb["ids"]["Supply Agreement - Tidewater (DEMO)"],)}
    assert all(a["prompt"] == "What are the payment terms in the supply agreement?" and a["label"].startswith("Only Supply Agreement")
               for a in asks)
    notes = next(c for c in env["components"] if c["type"] == "source_answer")["props"]["notes"]
    assert any("2 documents that are a “supply agreement”" in n for n in notes)


def test_a_name_two_parties_share_is_offered_as_each_party(amb):
    from app.legal.counsel import schemas
    alice = as_user(amb, "alice")
    env = turn(alice, amb["mid"], "What must Apex deliver to Orrin?")
    schemas.validate_envelope(env)
    asks = {a["label"]: a["prompt"] for a in env["suggested_actions"] if a["id"].startswith("only_party_")}
    assert asks == {"Only Apex Holdings Ltd": "What must Apex Holdings Ltd deliver to Orrin?",
                    "Only Apex Industries Ltd": "What must Apex Industries Ltd deliver to Orrin?"}


def test_a_question_that_names_one_document_or_party_in_full_is_offered_nothing(amb):
    alice = as_user(amb, "alice")
    for q in ("What are the payment terms in the Northwind supply agreement?", "What must Apex Industries Ltd deliver?"):
        env = turn(alice, amb["mid"], q)
        assert not [a for a in env["suggested_actions"] if a["id"].startswith("only_")], q


def test_more_documents_of_a_kind_than_are_answered_for_each_are_counted():
    from app.legal.research import kinds
    titles = {f"d{i}": f"Side Letter {i} (DEMO)" for i in range(6)}
    assert kinds.same_kind("What did the side letter say about fees?", titles) is None
    kind, ids = kinds.many_of_kind("What did the side letter say about fees?", titles)
    assert kind == "side letter" and len(ids) == 6
    assert kinds.many_of_kind("What did the side letter say?", dict(list(titles.items())[:3])) is None


def test_a_shorter_name_of_the_same_party_is_not_an_ambiguity():
    from app.legal.research import attribution
    assert attribution.ambiguous_party("What does Harrowgate charge?", {"harrowgate", "harrowgate technologies"}) is None
    assert attribution.ambiguous_party("What does Apex owe?", {"apex industries", "apex holdings"}) == ("Apex", ["apex holdings", "apex industries"])
    assert attribution.ambiguous_party("What does Apex Holdings Ltd owe?", {"apex industries", "apex holdings"}) is None
    assert attribution.party_names("", "Made between Apex Industries Ltd and Apex Holdings Ltd.") == {"apex industries", "apex holdings"}
    assert attribution.parties_of("", "Made between Apex Industries Ltd and Apex Holdings Ltd.") == {"apex"}
