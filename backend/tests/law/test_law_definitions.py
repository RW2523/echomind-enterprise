"""S3.5 — defined terms and cross-references (DOC-04; decision D-049).

A document is read for its own consistency, by rules (no model):
- **definitions:** every definition, a term defined twice (and whether the meanings differ), a
  definition never used, a capitalised term used but never defined, a defined term used in another
  form ("fees" for "Fees"), and a term used before it is defined;
- **numbering:** gaps and repeats at each level, sub-items included ("(a)", "(c)");
- **references:** a clause that does not exist, and a reference whose heading no longer matches.

Each finding carries exact offsets. A finding may carry proposed fixes; nothing changes until a
lawyer picks fixes, they become a redline (S3.3), and a second lawyer approves it before it is filed.
"""
from __future__ import annotations

import sqlite3

import pytest

from conftest import as_user, csrf_headers, index_all, upload

CONTRACT = ("SERVICES AGREEMENT (fictional)\n\n"
            "This Services Agreement (the \"Agreement\") is made between Northwind Hosting Ltd (the \"Supplier\") and Alder Retail Ltd "
            "(the \"Customer\").\n\n"
            "1. Definitions\n"
            "1.1 \"Services\" means the services described in Schedule 1.\n"
            "1.2 \"Fees\" means the fees set out in Schedule 2.\n"
            "1.3 \"Confidential Information\" means all information disclosed by a party that is marked as confidential.\n"
            "1.4 \"Business Day\" means a day other than a Saturday, Sunday or public holiday in England.\n"
            "1.5 \"Fees\" means the charges set out in the Order Form.\n"
            "1.6 \"Territory\" means the United Kingdom.\n\n"
            "2. Services\n"
            "2.1 The Supplier shall provide the Services from the Commencement Date.\n"
            "2.2 The Customer shall pay the fees within thirty Business Days of invoice.\n"
            "2.3 Each party shall protect the other's confidential information as required by clause 4 (Liability).\n\n"
            "4. Confidentiality\n"
            "4.1 Each party shall keep the other party's Confidential Information confidential.\n"
            "4.3 This clause 4.3 survives termination of this Agreement.\n\n"
            "9. Liability\n"
            "9.1 Liability is limited as set out in clause 12.\n"
            "9.2 Neither party excludes liability for:\n"
            "(a) death or personal injury caused by negligence;\n"
            "(c) fraud.\n\n"
            "Schedule 1 Services\nManaged hosting (fictional).\n\n"
            "Schedule 2 Fees\nEUR 10,000 per month (fictional).")


@pytest.fixture
def dc(docs):
    from app.legal.matters import repo
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    repo.add_membership(mid, docs["users"]["dave"]["id"], "attorney", granted_by=docs["users"]["alice"]["id"])
    doc = upload(alice, mid, "services-agreement.txt", CONTRACT, status="draft").json()["document"]
    index_all()
    docs.update(mid=mid, doc=doc, base=f"/api/legal/matters/{mid}")
    return docs


def run(w, by="alice"):
    c = as_user(w, by)
    r = c.post(f"{w['base']}/documents/{w['doc']['id']}/definitions-check", headers=csrf_headers(c))
    assert r.status_code == 200, r.text
    return r.json()["check"]


def found(ch, type_, contains=""):
    return [f for f in ch["findings"] if f["type"] == type_ and contains in f["text"]]


def test_definitions_duplicated_unused_and_undefined(dc):
    ch = run(dc)
    dup = found(ch, "duplicate", "Fees")
    assert len(dup) == 1 and dup[0]["conflicting"] is True and len(dup[0]["places"]) == 2
    assert [CONTRACT[p["start"]:p["end"]] for p in dup[0]["places"]] == ["Fees", "Fees"]
    assert [p["where"] for p in dup[0]["places"]] == ["clause 1.2", "clause 1.5"]
    assert found(ch, "undefined")[0]["where"] == "clause 2.1"
    assert [f["text"] for f in found(ch, "unused")] == ["Territory"]
    assert [f["text"] for f in found(ch, "undefined")] == ["Commencement Date"]
    f = found(ch, "undefined")[0]
    assert CONTRACT[f["start"]:f["end"]] == "Commencement Date"


def test_a_defined_term_used_in_another_form_comes_with_a_fix_at_its_offsets(dc):
    ch = run(dc)
    inc = {f["text"]: f for f in found(ch, "inconsistent")}
    assert set(inc) == {"fees", "confidential information"}
    fix = inc["fees"]["fixes"][0]
    assert CONTRACT[fix["start"]:fix["end"]] == "fees" and fix["revised"] == "Fees"
    assert CONTRACT[fix["start"] - 4:fix["end"] + 7] == "the fees within"            # 2.2, not the definition in 1.2
    assert inc["confidential information"]["fixes"][0]["revised"] == "Confidential Information"


def test_numbering_gaps_at_each_level_and_in_sub_items(dc):
    ch = run(dc)
    gaps = [f["message"] for f in found(ch, "gap")]
    assert "Clause 3 is missing: the numbering goes from 2 to 4." in gaps
    assert "Clauses 5 to 8 are missing: the numbering goes from 4 to 9." in gaps
    assert "Clause 4.2 is missing: the numbering goes from 4.1 to 4.3." in gaps
    assert "Item (b) of clause 9.2 is missing: the items go from (a) to (c)." in gaps
    assert len(gaps) == 4 and all(f["fixes"] == [] for f in found(ch, "gap"))   # never renumbered automatically


def test_broken_and_outdated_references(dc):
    ch = run(dc)
    broken = found(ch, "broken")
    assert [f["text"] for f in broken] == ["clause 12"] and broken[0]["message"] == "This document has no clause 12."
    old = found(ch, "outdated")
    assert len(old) == 1 and old[0]["text"] == "clause 4 (Liability)"
    assert old[0]["message"] == "Clause 4 is headed Confidentiality, not Liability."
    assert [x["revised"] for x in old[0]["fixes"]] == ["clause 4 (Confidentiality)", "clause 9 (Liability)"]
    assert CONTRACT[old[0]["fixes"][0]["start"]:old[0]["fixes"][0]["end"]] == "clause 4 (Liability)"


def test_nothing_is_flagged_that_is_not_a_problem(dc):
    ch = run(dc)
    texts = " | ".join(f"{f['type']}:{f['text']}" for f in ch["findings"])
    for fine in ("Agreement", "Supplier", "Customer", "Business Day", "Services", "Confidential Information:", "Northwind", "Alder",
                 "Order Form", "Saturday", "England", "United Kingdom", "Schedule 1", "Schedule 2", "clause 4.3"):
        assert f":{fine}" not in texts, (fine, texts)
    assert ch["summary"] == {"definitions": 8, "duplicate": 1, "unused": 1, "undefined": 1, "inconsistent": 2, "before_definition": 0,
                             "gap": 4, "repeat": 0, "broken": 1, "outdated": 1}


def test_a_term_used_before_it_is_defined_and_a_repeated_number():
    from app.legal.definitions import check as checkmod
    text = ("1. Scope\n1.1 The Supplier provides the Deliverables.\n1.1 The Customer pays the Price.\n\n"
            "2. Terms\n2.1 \"Deliverables\" means the reports. \"Price\" means EUR 5 (fictional).\n")
    clauses = [{"number": "1", "heading": "Scope", "level": 1, "parent": None, "start": 0, "end": 85},
               {"number": "1.1", "heading": None, "level": 2, "parent": "1", "start": 9, "end": 47},
               {"number": "1.1", "heading": None, "level": 2, "parent": "1", "start": 48, "end": 85},
               {"number": "2", "heading": "Terms", "level": 1, "parent": None, "start": 87, "end": len(text)},
               {"number": "2.1", "heading": None, "level": 2, "parent": "2", "start": 96, "end": len(text)}]
    out = checkmod.analyse(text, clauses)
    before = {f["text"] for f in out if f["type"] == "before_definition"}
    assert before == {"Deliverables", "Price"}
    assert [f["message"] for f in out if f["type"] == "repeat"] == ["Clause 1.1 appears twice."]


def test_fixes_become_a_redline_that_a_second_lawyer_approves_before_it_is_filed(dc, fresh_db):
    from app.legal.redlines import word
    ch = run(dc)
    alice, dave = as_user(dc, "alice"), as_user(dc, "dave")
    inc = {f["text"]: f for f in found(ch, "inconsistent")}
    old = found(ch, "outdated")[0]
    pick = [{"finding": inc["fees"]["id"], "fix": 0}, {"finding": old["id"], "fix": 0}]
    r = alice.post(f"{dc['base']}/definitions-checks/{ch['id']}/redline", headers=csrf_headers(alice), json={"fixes": pick})
    assert r.status_code == 200, r.text
    red = r.json()["redline"]
    assert red["kind"] == "redline" and red["content"]["origin"] == "definitions" and red["content"]["playbook"] is None
    revised = {c["clause"]["number"]: c["revised"] for c in red["content"]["changes"]}
    assert revised == {"2.2": "2.2 The Customer shall pay the Fees within thirty Business Days of invoice.",
                       "2.3": "2.3 Each party shall protect the other's confidential information as required by clause 4 (Confidentiality)."}
    # nothing has changed in the document yet
    assert alice.get(f"{dc['base']}/documents/{dc['doc']['id']}").json()["document"]["current_version"]["number"] == 1
    out = alice.get(f"{dc['base']}/redlines/{red['id']}/word")
    assert out.status_code == 200
    after = [word.fold(p) for p in word.accept_all(out.content)]
    assert "2.2 The Customer shall pay the Fees within thirty Business Days of invoice." in after
    assert alice.post(f"{dc['base']}/artifacts/{red['id']}/file", headers=csrf_headers(alice)).status_code == 409    # not approved
    assert dave.post(f"{dc['base']}/artifacts/{red['id']}/review", headers=csrf_headers(dave),
                     json={"decision": "approved", "content_sha256": red["content_sha256"]}).status_code == 200
    assert alice.post(f"{dc['base']}/artifacts/{red['id']}/file", headers=csrf_headers(alice)).status_code == 200
    assert alice.get(f"{dc['base']}/documents/{dc['doc']['id']}").json()["document"]["current_version"]["number"] == 2
    # a fix that does not exist, or two fixes for one finding, are refused
    bad = alice.post(f"{dc['base']}/definitions-checks/{ch['id']}/redline", headers=csrf_headers(alice),
                     json={"fixes": [{"finding": old["id"], "fix": 7}]})
    assert bad.status_code == 422
    both = alice.post(f"{dc['base']}/definitions-checks/{ch['id']}/redline", headers=csrf_headers(alice),
                      json={"fixes": [{"finding": old["id"], "fix": 0}, {"finding": old["id"], "fix": 1}]})
    assert both.status_code == 422


def test_an_unused_definition_can_be_deleted_as_a_tracked_change(dc):
    from app.legal.redlines import word
    ch = run(dc)
    unused = found(ch, "unused")[0]
    assert [x["label"] for x in unused["fixes"]] == ["Delete the definition"]
    alice = as_user(dc, "alice")
    red = alice.post(f"{dc['base']}/definitions-checks/{ch['id']}/redline", headers=csrf_headers(alice),
                     json={"fixes": [{"finding": unused["id"], "fix": 0}]}).json()["redline"]
    out = alice.get(f"{dc['base']}/redlines/{red['id']}/word").content
    after, before = [word.fold(p) for p in word.accept_all(out)], [word.fold(p) for p in word.reject_all(out)]
    assert not any("Territory" in p for p in after) and any("Territory" in p for p in before)


def test_any_member_who_can_see_the_document_checks_it_lawyers_note_and_propose(dc):
    ch = run(dc, by="carol")                                     # read-only: research.query
    carol, alice = as_user(dc, "carol"), as_user(dc, "alice")
    f = found(ch, "undefined")[0]
    url = f"{dc['base']}/definitions-checks/{ch['id']}"
    assert carol.post(f"{url}/notes", headers=csrf_headers(carol), json={"finding": f["id"], "note": "x"}).status_code == 403
    assert carol.post(f"{url}/redline", headers=csrf_headers(carol), json={"fixes": []}).status_code == 403
    r = alice.post(f"{url}/notes", headers=csrf_headers(alice), json={"finding": f["id"], "note": "Define it in 1.7."})
    assert r.status_code == 200
    assert next(x for x in r.json()["check"]["findings"] if x["id"] == f["id"])["notes"][0]["note"] == "Define it in 1.7."


def test_a_check_is_shown_only_with_its_document_and_purged_with_it(dc, fresh_db):
    from app.legal.documents import repo
    from app.legal.records import deletion
    ch = run(dc)
    assert [c["id"] for c in as_user(dc, "carol").get(f"{dc['base']}/documents/{dc['doc']['id']}/definitions-checks").json()["checks"]] == [ch["id"]]
    repo.set_acl(dc["doc"]["id"], {dc["users"]["alice"]["id"]})
    assert as_user(dc, "carol").get(f"{dc['base']}/definitions-checks/{ch['id']}").status_code == 404
    assert as_user(dc, "bob").get(f"{dc['base']}/documents/{dc['doc']['id']}/definitions-checks").status_code == 404
    org, users = dc["matters"]["M-101"]["org_id"], dc["users"]
    d = deletion.request(org, dc["mid"], "document", dc["doc"]["id"], "Superseded", users["alice"]["id"])
    assert d["report"]["definition_checks"] == 1
    assert deletion.purge(d["id"], users["mp"]["id"])["verification"]["ok"]
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT COUNT(*) FROM legal_definition_checks").fetchone()[0] == 0


def test_the_demo_agreement(docs):
    """The fixture journey 26 uses: the demo MSA cites Schedules 1, 2 and 4, which it does not contain,
    and uses "Confidential Information" without defining it."""
    from pathlib import Path
    fx = Path(__file__).resolve().parents[3] / "eval" / "fixtures" / "law" / "Master Services Agreement (DEMO).md"
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    msa = upload(alice, mid, fx.name, fx.read_text(), mime="text/markdown", status="executed").json()["document"]
    index_all()
    ch = alice.post(f"/api/legal/matters/{mid}/documents/{msa['id']}/definitions-check", headers=csrf_headers(alice)).json()["check"]
    assert sorted({f["text"] for f in found(ch, "broken")}) == ["Schedule 1", "Schedule 2", "Schedule 4"]
    assert [f["text"] for f in found(ch, "undefined")] == ["Confidential Information"]
    assert not found(ch, "duplicate") and not found(ch, "outdated")
    assert "Clauses 10 and 11 are missing: the numbering goes from 9 to 12." in [f["message"] for f in found(ch, "gap")]


def test_the_demo_supply_agreement_in_markdown(docs):
    """Journey 26's fixture: Markdown headings, so clauses 1, 2, 4 and 9 and both schedules are known only by
    their headings and sub-clauses (documents.numbering)."""
    from pathlib import Path
    fx = Path(__file__).resolve().parents[3] / "eval" / "fixtures" / "law" / "Supply Agreement (DEMO).md"
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, fx.name, fx.read_text(), mime="text/markdown", status="draft").json()["document"]
    index_all()
    ch = alice.post(f"/api/legal/matters/{mid}/documents/{doc['id']}/definitions-check", headers=csrf_headers(alice)).json()["check"]
    got = sorted((f["type"], f["text"]) for f in ch["findings"] if f["kind"] != "numbering")
    assert got == sorted([("duplicate", "Price"), ("unused", "Territory"), ("undefined", "Commencement Date"), ("inconsistent", "price"),
                          ("inconsistent", "confidential information"), ("outdated", "clause 4 (Liability)"), ("broken", "clause 12")])
    assert sorted(f["message"] for f in found(ch, "gap")) == sorted([
        "Clause 3 is missing: the numbering goes from 2 to 4.", "Clauses 5 to 8 are missing: the numbering goes from 4 to 9.",
        "Clause 4.2 is missing: the numbering goes from 4.1 to 4.3.", "Item (b) of clause 9.2 is missing: the items go from (a) to (c)."])


def test_a_clean_agreement_has_no_findings():
    """Golden control G4: party short names left unused, 'each a "Party"', possessives ("Party's Confidential
    Information") and (i) under (c) are all as they should be."""
    import json
    from pathlib import Path
    from app.legal.definitions import check as checkmod
    from app.legal.documents import extract, structure
    g = json.loads((Path(__file__).resolve().parents[3] / "eval" / "law" / "definitions" / "golden.json").read_text())
    text = next(d["text"] for d in g["documents"] if d["id"] == "G4-clean-nda-control")
    x = extract.extract("nda.txt", text.encode())
    rows = [{"number": c.number, "heading": c.heading, "level": c.level, "parent": c.parent, "start": c.start, "end": c.end}
            for c in structure.clauses(x)]
    assert "2.2(c)(i)" in [r["number"] for r in rows]
    assert checkmod.analyse(text, rows) == []
