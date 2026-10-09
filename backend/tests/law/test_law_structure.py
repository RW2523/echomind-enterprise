"""S2.1b — clauses and defined terms (RAG-01; D-035).

A lawyer cites "clause 12.3(b)", not "characters 1,204 to 1,262". Clauses come from Word's own
numbering when there is one, and otherwise from the clause markers legal drafting uses (12.3, (b),
Article IV, Section 5.1, § 7, Schedule 2, Exhibit A), without mistaking a year or an amount at the
start of a line for a clause. Defined terms are found where they are defined. Search citations
and research sources carry the clause they quote from.
"""
from __future__ import annotations

import json

import pytest

from conftest import as_user, csrf_headers, index_all, search_in, upload
from fixtures_s21 import contract_docx

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TEXT_CONTRACT = """SUPPLY AGREEMENT (fictional)

ARTICLE IV — PAYMENT

4.1 Fees. The Customer shall pay the fees in Schedule 2.

(a) Fees are invoiced monthly.

(b) Late payments bear interest at 4% a year.

4.2 Taxes. Fees exclude VAT.

2024 was the first year of supply under this Agreement.

12,000 units were delivered in the first quarter.

Section 5.1 Warranties. The Supplier warrants that the Goods conform to the Specification.

§ 7 Governing law. This Agreement is governed by English law.

Schedule 2 — Fees

Exhibit A — Specification
"""
DEFINITIONS = ('In this Agreement, "Services" means the platform services described in Schedule 2. '
               'Harrowgate Ltd (the "Supplier") provides them. “Confidential Information” shall mean any information '
               'marked confidential. "Business Day" has the meaning given in clause 1.2.')


def clauses_of(x):
    from app.legal.documents import structure
    return structure.clauses(x)


def test_clauses_come_from_word_numbering():
    from app.legal.documents import extract
    x = extract.extract("msa.docx", contract_docx())
    cl = clauses_of(x)
    assert [c.number for c in cl] == ["Article I", "12", "12.1", "12.2", "12.3", "12.3(a)", "12.3(b)", "13", "13.1"]
    by = {c.number: c for c in cl}
    assert by["12"].heading == "Limitation of Liability" and by["Article I"].heading == "Definitions"
    assert x.text[by["12.3"].start:by["12.3"].end].startswith("12.3 Neither party is liable for:")
    assert "(b) indirect or consequential loss." in x.text[by["12.3"].start:by["12.3"].end]     # children inside
    assert "13. Term" not in x.text[by["12"].start:by["12"].end]
    assert by["12.3(b)"].parent == "12.3" and by["12.3"].parent == "12" and by["12"].parent is None


def test_clauses_are_found_in_plain_text_and_pdfs_without_mistaking_years_or_amounts():
    from app.legal.documents import extract
    x = extract.extract("supply.txt", TEXT_CONTRACT.encode())
    numbers = [c.number for c in clauses_of(x)]
    assert numbers == ["Article IV", "4.1", "4.1(a)", "4.1(b)", "4.2", "5.1", "§ 7", "Schedule 2", "Exhibit A"]
    by = {c.number: c for c in clauses_of(x)}
    assert by["4.1"].heading == "Fees" and by["4.1"].parent == "Article IV" and by["4.1(b)"].parent == "4.1"
    assert by["Schedule 2"].heading == "Fees"
    assert "2024 was the first year" not in [c.heading for c in clauses_of(x)]


def test_clause_at_finds_the_deepest_clause():
    from app.legal.documents import extract, structure
    x = extract.extract("supply.txt", TEXT_CONTRACT.encode())
    cl = clauses_of(x)
    assert structure.clause_at(cl, x.text.index("interest at 4%")).number == "4.1(b)"
    assert structure.clause_at(cl, x.text.index("exclude VAT")).number == "4.2"
    assert structure.clause_at(cl, 0) is None                       # the title is before any clause


def test_defined_terms_are_found_where_they_are_defined():
    from app.legal.documents import extract, structure
    x = extract.extract("defs.txt", DEFINITIONS.encode())
    terms = {t.term: x.text[t.start:t.end] for t in structure.defined_terms(x)}
    assert set(terms) == {"Services", "Supplier", "Confidential Information", "Business Day"}
    assert terms["Services"].startswith('"Services" means the platform services')
    assert terms["Supplier"] == 'Harrowgate Ltd (the "Supplier") provides them.'
    assert terms["Business Day"].endswith("in clause 1.2.")


def test_structure_is_stored_and_search_citations_carry_the_clause(docs, fresh_db):
    import sqlite3
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "supply.txt", TEXT_CONTRACT).json()["document"]
    index_all()
    vid = doc["current_version"]["id"]
    with sqlite3.connect(fresh_db) as c:
        stored = [r[0] for r in c.execute("SELECT number FROM legal_clauses WHERE version_id=? ORDER BY char_start", (vid,))]
        terms = c.execute("SELECT COUNT(*) FROM legal_defined_terms WHERE version_id=?", (vid,)).fetchone()[0]
    assert stored[:3] == ["Article IV", "4.1", "4.1(a)"] and terms == 0
    hits = search_in(alice, mid, "late payments bear interest 4% a year").json()["passages"]
    assert hits and hits[0]["citation"]["clause"] in ("4.1", "4.1(a)", "4.1(b)")


def test_research_sources_name_their_clause(docs, monkeypatch):
    from app.legal.research import llm, prompt
    from app.core.config import settings

    class Echo:
        calls = []

        async def chat_stream(self, messages, temperature, max_tokens):
            Echo.calls.append(messages)
            yield "Late payments bear interest at 4% a year [1]."
    monkeypatch.setattr(llm, "_override", Echo())
    monkeypatch.setattr(settings, "LEGAL_STREAM_RECHECK_SEC", 0.0)
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    upload(alice, mid, "supply.txt", TEXT_CONTRACT)
    index_all()
    r = alice.post(f"/api/legal/matters/{mid}/research/query", headers=csrf_headers(alice),
                   json={"question": "What interest do late payments bear a year?"})
    ev = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert ev[0]["citations"][0]["clause"] in ("4.1", "4.1(a)", "4.1(b)")
    assert 'clause="4.1' in Echo.calls[-1][-1]["content"]


def test_a_purge_removes_clauses_and_terms(docs, fresh_db):
    import sqlite3
    from app.legal.records import deletion
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "defs.txt", DEFINITIONS + "\n\n" + TEXT_CONTRACT).json()["document"]
    index_all()
    assert deletion.preview(mid, "document", doc["id"])["clauses"] >= 9
    req = deletion.request(docs["matters"]["M-101"]["org_id"], mid, "document", doc["id"], "Duplicate", docs["users"]["alice"]["id"])
    assert deletion.purge(req["id"], docs["users"]["mp"]["id"])["verification"]["ok"] is True
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT COUNT(*) FROM legal_clauses").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM legal_defined_terms").fetchone()[0] == 0


def test_a_sentence_wrapped_across_visual_lines_is_one_sentence():
    """PDF and OCR text keeps its visual lines; a citation's focus must still be the whole sentence."""
    from app.legal.retrieval.search import _focus
    text = ("NOTICE OF TERMINATION\n\nHarrowgate Ltd gives notice that the Master Services Agreement dated 3 March\n"
            "2024 will terminate on 30 June 2026 under clause 13.1. All fees up to that date remain payable.")
    a, b = _focus(text, "On what date will the agreement terminate?")
    assert text[a:b].startswith("Harrowgate Ltd") and text[a:b].endswith("under clause 13.1.")
    from app.legal.documents import extract, structure
    x = extract.extract("defs.txt", b'The "Supplier" means Harrowgate Ltd, a company\nregistered in England.')
    t = structure.defined_terms(x)[0]
    assert x.text[t.start:t.end].endswith("registered in England.")
