"""S2.2a — answers grounded in the right version, with conflicts shown (RAG-01; spec test T05; D-036).

A matter often holds a signed contract and a later draft that says something else. By default
search and research use each document's latest signed (executed) version; a document that was
never signed is used in its current version, marked unsigned. "Latest drafts" uses current versions
instead. Either way, where another version of the same document words a cited clause differently,
the answer says so.
"""
from __future__ import annotations

import json
import re

import pytest

from conftest import as_user, csrf_headers, index_all, search_in, upload

SIGNED = ("MASTER SERVICES AGREEMENT (fictional)\n\n"
          "12.1 Liability cap. Each party's total liability is capped at USD 250,000.\n\n"
          "12.2 Notices. Notices must be given in writing to the addresses in Schedule 2.")
DRAFT = ("MASTER SERVICES AGREEMENT (fictional) — client redraft\n\n"
         "12.1 Liability cap. Each party's total liability is capped at USD 500,000.\n\n"
         "12.2 Notices. Notices must be given in writing to the addresses in Schedule 2.")
CAP_Q = "What is each party's total liability cap?"


class Quoting:
    """A stand-in model that answers from source 1 with an exact quote."""
    def __init__(self):
        self.calls = []

    async def chat_stream(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        user = messages[-1]["content"]
        cap = re.search(r"capped at USD [\d,]+", user)
        yield f'The cap is {cap.group(0)[len("capped at "):] if cap else "unknown"} [1: "{cap.group(0) if cap else "x"}"].'


@pytest.fixture
def two_versions(docs, monkeypatch):
    from app.core.config import settings
    from app.legal.research import llm
    monkeypatch.setattr(settings, "LEGAL_STREAM_RECHECK_SEC", 0.0)
    fake = Quoting()
    monkeypatch.setattr(llm, "_override", fake)
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    v1 = upload(alice, mid, "msa.txt", SIGNED, status="executed").json()["document"]
    index_all()
    v2 = upload(alice, mid, "msa-redraft.txt", DRAFT, status="draft", version_of=v1["id"]).json()["document"]
    index_all()
    docs.update(mid=mid, doc=v2, llm=fake)
    return docs


def ask(c, mid, question, **extra):
    r = c.post(f"/api/legal/matters/{mid}/research/query", headers=csrf_headers(c), json={"question": question, **extra})
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def test_search_uses_the_signed_version_by_default(two_versions):
    alice = as_user(two_versions, "alice")
    hits = search_in(alice, two_versions["mid"], CAP_Q).json()["passages"]
    assert hits and all("500,000" not in h["text"] for h in hits)
    top = hits[0]["citation"]
    assert "250,000" in hits[0]["text"] and top["version_number"] == 1 and top["version_status"] == "executed"


def test_latest_drafts_can_be_chosen_instead(two_versions):
    alice = as_user(two_versions, "alice")
    hits = search_in(alice, two_versions["mid"], CAP_Q, versions="current").json()["passages"]
    assert hits and all("250,000" not in h["text"] for h in hits)
    assert hits[0]["citation"]["version_number"] == 2 and hits[0]["citation"]["version_status"] == "draft"


def test_t05_the_answer_follows_the_signed_version_and_warns_of_the_draft(two_versions):
    alice = as_user(two_versions, "alice")
    ev = ask(alice, two_versions["mid"], CAP_Q)
    done = ev[-1]
    assert done["type"] == "done" and "250,000" in done["answer"] and "500,000" not in done["answer"]
    conflicts = done["conflicts"]
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c["clause"] == "12.1" and c["version"]["number"] == 1 and c["version"]["status"] == "executed"
    assert c["other"]["number"] == 2 and c["other"]["status"] == "draft" and "500,000" in c["other"]["text"]
    assert done["versions"] == "signed"


def test_t05_with_latest_drafts_the_answer_follows_the_draft_and_warns_of_the_signed_text(two_versions):
    alice = as_user(two_versions, "alice")
    done = ask(alice, two_versions["mid"], CAP_Q, versions="current")[-1]
    assert "500,000" in done["answer"]
    c = done["conflicts"][0]
    assert c["version"]["number"] == 2 and c["other"]["number"] == 1 and c["other"]["status"] == "executed"
    assert "250,000" in c["other"]["text"]


def test_an_unchanged_clause_is_not_a_conflict(two_versions):
    alice = as_user(two_versions, "alice")
    done = ask(alice, two_versions["mid"], "How must notices be given?")[-1]
    assert done["conflicts"] == []


def test_a_document_never_signed_is_used_in_its_current_version_marked_unsigned(docs):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    upload(alice, mid, "side-letter.txt", "Side letter (fictional). The parties agree a service credit of 5% for each outage.")
    index_all()
    hit = search_in(alice, mid, "service credit for each outage").json()["passages"][0]
    assert hit["citation"]["version_status"] == "draft" and hit["citation"]["signed"] is False


def test_signing_the_draft_makes_it_the_version_answers_come_from(two_versions):
    alice = as_user(two_versions, "alice")
    mid, doc = two_versions["mid"], two_versions["doc"]
    r = alice.patch(f"/api/legal/matters/{mid}/documents/{doc['id']}", headers=csrf_headers(alice),
                    json={"status": "executed", "version": doc["version"]})
    assert r.status_code == 200
    hits = search_in(alice, mid, CAP_Q).json()["passages"]
    assert "500,000" in hits[0]["text"] and hits[0]["citation"]["version_number"] == 2
    assert all("250,000" not in h["text"] for h in hits)


def test_an_unsigned_older_version_leaves_the_index_when_replaced(docs, fresh_db):
    import sqlite3
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    v1 = upload(alice, mid, "memo.txt", "Draft memo one (fictional). The HERON-11 figure is provisional.").json()["document"]
    index_all()
    upload(alice, mid, "memo2.txt", "Draft memo two (fictional). The HERON-22 figure is final.", version_of=v1["id"])
    index_all()
    with sqlite3.connect(fresh_db) as c:
        rows = dict(c.execute("SELECT v.number, MAX(ch.active) FROM legal_chunks ch JOIN legal_document_versions v ON v.id = ch.version_id "
                              "GROUP BY v.number").fetchall())
        status = dict(c.execute("SELECT number, index_status FROM legal_document_versions").fetchall())
    assert rows == {1: 0, 2: 1} and status[1] == "superseded"
    assert search_in(alice, mid, "HERON-11 provisional figure").json()["passages"] == [] or \
        all("HERON-11" not in h["text"] for h in search_in(alice, mid, "HERON-11 provisional figure").json()["passages"])
