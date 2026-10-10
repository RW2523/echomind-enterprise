"""S3.4 — the citation and quote checker (DOC-03; spec test T07; decision D-048).

A document is read for what it cites: references to clauses (in itself or in another matter
document it names), quotations, and external authorities. References are resolved to the exact
passage, or flagged when the clause does not exist. A quotation is compared with the passage it
quotes, and changed words are shown. External authorities are listed and always marked UNVERIFIED:
no licensed library is available (P-009) and nothing is looked up. A citation that cannot be real
(an unknown reporter or court, a future year) is flagged. No model is involved.
"""
from __future__ import annotations

import sqlite3

import pytest

from conftest import as_user, csrf_headers, index_all, upload

MSA = ("MASTER SERVICES AGREEMENT (fictional)\n\n"
       "4. Fees\n4.2 The Customer shall pay each invoice within thirty days of receipt.\n\n"
       "8. Liability\n8.1 Each party's total liability shall not exceed USD 250,000.\n\n"
       "12. Termination\n12.1 Either party may terminate on ninety days' notice.")
MEMO = ("ADVICE MEMORANDUM (fictional)\n\n"
        "1. Liability\n1.1 Under clause 8.1 of the Master Services Agreement, liability is capped. Clause 8.1 provides that "
        "\"Each party's total liability shall not exceed USD 250,000.\"\n\n"
        "2. Payment\n2.1 Clause 4.2 of the Master Services Agreement states that \"The Customer shall pay each invoice within sixty days of receipt.\"\n\n"
        "3. Termination\n3.1 Clause 15.4 of the Master Services Agreement allows termination for convenience. See also paragraph 1.1 above.\n\n"
        "4. Authorities\n4.1 Disclosure duties arise under 15 U.S.C. § 78j(b) and 17 C.F.R. § 240.10b-5, and see Regulation (EU) 2016/679.\n\n"
        "4.2 The point was decided in Henderson v. Bramwell, 999 F.5th 1234 (14th Cir. 2031) and in [2019] UKSC 12.")


@pytest.fixture
def ct(docs):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    msa = upload(alice, mid, "Master Services Agreement.txt", MSA, status="executed", title="Master Services Agreement").json()["document"]
    memo = upload(alice, mid, "advice-memo.txt", MEMO, status="draft").json()["document"]
    index_all()
    docs.update(mid=mid, msa=msa, memo=memo, base=f"/api/legal/matters/{mid}")
    return docs


def run(w, by="alice"):
    c = as_user(w, by)
    r = c.post(f"{w['base']}/documents/{w['memo']['id']}/citation-check", headers=csrf_headers(c))
    assert r.status_code == 200, r.text
    return r.json()["check"]


def item(ch, kind, contains):
    return next(i for i in ch["items"] if i["kind"] == kind and contains in i["text"])


def test_references_are_resolved_to_the_exact_clause_or_flagged_as_missing(ct, fresh_db):
    ch = run(ct)
    ref = item(ch, "reference", "clause 8.1 of the Master Services Agreement")
    assert ref["status"] == "resolved" and ref["target"]["document_id"] == ct["msa"]["id"] and ref["target"]["clause"] == "8.1"
    with sqlite3.connect(fresh_db) as c:
        text = c.execute("SELECT text FROM legal_version_text WHERE version_id=?", (ref["target"]["version_id"],)).fetchone()[0]
    assert text[ref["target"]["start"]:ref["target"]["end"]].startswith("8.1 Each party's total liability")
    missing = item(ch, "reference", "Clause 15.4")
    assert missing["status"] == "missing" and "has no clause 15.4" in missing["note"]
    own = item(ch, "reference", "paragraph 1.1")
    assert own["status"] == "resolved" and own["target"]["document_id"] == ct["memo"]["id"]


def test_a_quotation_is_compared_with_the_passage_it_quotes(ct):
    ch = run(ct)
    good = item(ch, "quote", "Each party's total liability shall not exceed USD 250,000.")
    assert good["status"] == "accurate" and good["target"]["clause"] == "8.1"
    bad = item(ch, "quote", "within sixty days of receipt")
    assert bad["status"] == "altered" and bad["target"]["clause"] == "4.2"
    assert ["del", "thirty"] in bad["diff"] and ["ins", "sixty"] in bad["diff"]


def test_external_authorities_are_listed_and_never_called_verified(ct):
    ch = run(ct)
    auths = [i for i in ch["items"] if i["kind"] == "authority"]
    found = {i["text"] for i in auths}
    assert {"15 U.S.C. § 78j(b)", "17 C.F.R. § 240.10b-5", "Regulation (EU) 2016/679", "[2019] UKSC 12"} <= found
    assert all(i["status"] == "unverified" for i in auths)
    assert ch["external_lookup"] is False and "P-009" in ch["notice"]


def test_a_citation_that_cannot_be_real_is_flagged(ct):
    ch = run(ct)
    fake = item(ch, "authority", "Henderson v. Bramwell")
    assert fake["status"] == "unverified" and fake["malformed"]
    reasons = " ".join(fake["malformed"])
    assert "F.5th" in reasons and "14th Cir." in reasons and "2031" in reasons
    real = item(ch, "authority", "15 U.S.C.")
    assert real["malformed"] == []


def test_a_check_is_kept_shown_only_with_its_document_and_purged_with_it(ct, fresh_db):
    from app.legal.documents import repo
    from app.legal.records import deletion
    ch = run(ct)
    assert ch["summary"] == {"references": 3, "resolved": 2, "missing": 1, "quotes": 2, "accurate": 1, "altered": 1, "not_found": 0,
                             "authorities": 5, "malformed": 1}
    listed = as_user(ct, "carol").get(f"{ct['base']}/documents/{ct['memo']['id']}/citation-checks").json()["checks"]
    assert [c["id"] for c in listed] == [ch["id"]]
    repo.set_acl(ct["memo"]["id"], {ct["users"]["alice"]["id"]})
    assert as_user(ct, "carol").get(f"{ct['base']}/documents/{ct['memo']['id']}/citation-checks").status_code == 404
    assert as_user(ct, "bob").get(f"{ct['base']}/documents/{ct['memo']['id']}/citation-checks").status_code == 404
    org, users = ct["matters"]["M-101"]["org_id"], ct["users"]
    d = deletion.request(org, ct["mid"], "document", ct["memo"]["id"], "Draft superseded", users["alice"]["id"])
    assert d["report"]["citation_checks"] == 1
    assert deletion.purge(d["id"], users["mp"]["id"])["verification"]["ok"]
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT COUNT(*) FROM legal_citation_checks").fetchone()[0] == 0


def test_a_target_document_the_person_cannot_see_is_never_quoted_or_named(ct):
    from app.legal.documents import repo
    repo.set_acl(ct["msa"]["id"], {ct["users"]["bob"]["id"]})                  # alice cannot see the MSA
    ch = run(ct)
    ref = item(ch, "reference", "clause 8.1 of the Master Services Agreement")
    assert ref["status"] == "not_found" and ref["target"] is None
    assert all((i.get("target") or {}).get("document_id") != ct["msa"]["id"] for i in ch["items"])
    # words only the MSA has (the memo quotes "sixty days"; the MSA says thirty) and its file name never appear
    assert "thirty days" not in str(ch) and "ninety days' notice" not in str(ch) and "Master Services Agreement.txt" not in str(ch)
    assert item(ch, "quote", "within sixty days")["status"] == "not_found"


def test_any_member_who_can_see_the_document_can_check_it(ct):
    """research.query (IMPLEMENTATION_PLAN §5): a reviewer or read-only member checks a memo before relying on it."""
    from app.legal.documents import repo
    ch = run(ct, by="carol")
    assert ch["created_by"] == "carol" and ch["summary"]["references"] == 3
    repo.set_acl(ct["memo"]["id"], {ct["users"]["alice"]["id"]})
    carol = as_user(ct, "carol")
    assert carol.post(f"{ct['base']}/documents/{ct['memo']['id']}/citation-check", headers=csrf_headers(carol)).status_code == 404
    bob = as_user(ct, "bob")                                   # not a member of M-101
    assert bob.post(f"{ct['base']}/documents/{ct['memo']['id']}/citation-check", headers=csrf_headers(bob)).status_code == 404


def test_a_lawyer_notes_an_item(ct):
    ch = run(ct)
    alice, carol = as_user(ct, "alice"), as_user(ct, "carol")
    fake = item(ch, "authority", "Henderson v. Bramwell")
    url = f"{ct['base']}/citation-checks/{ch['id']}/notes"
    assert carol.post(url, headers=csrf_headers(carol), json={"item": fake["id"], "note": "Checked"}).status_code == 403
    r = alice.post(url, headers=csrf_headers(alice), json={"item": fake["id"], "note": "No such case: removed from the memo."})
    assert r.status_code == 200
    got = item(r.json()["check"], "authority", "Henderson v. Bramwell")["notes"]
    assert got[0]["note"] == "No such case: removed from the memo." and got[0]["by"] == "alice"


def test_a_target_restricted_after_the_check_is_hidden_from_those_who_cannot_open_it(ct):
    """A stored check keeps where an item points, never the target's words; they are read again for each person."""
    from app.legal.documents import repo
    ch = run(ct)
    repo.set_acl(ct["msa"]["id"], {ct["users"]["alice"]["id"]})
    seen = as_user(ct, "carol").get(f"{ct['base']}/citation-checks/{ch['id']}").json()["check"]
    ref = item(seen, "reference", "clause 8.1 of the Master Services Agreement")
    assert ref["status"] == "not_found" and ref["target"] is None
    bad = item(seen, "quote", "within sixty days")
    assert bad["status"] == "not_found" and bad["diff"] == [] and "thirty" not in str(seen)
    mine = as_user(ct, "alice").get(f"{ct['base']}/citation-checks/{ch['id']}").json()["check"]
    assert item(mine, "reference", "clause 8.1 of the Master")["target"]["passage"].startswith("8.1 Each party's")


def test_purging_the_document_a_check_points_into_leaves_none_of_it_behind(ct, fresh_db):
    from app.legal.records import deletion
    ch = run(ct)
    org, users = ct["matters"]["M-101"]["org_id"], ct["users"]
    d = deletion.request(org, ct["mid"], "document", ct["msa"]["id"], "Wrong client", users["alice"]["id"])
    assert deletion.purge(d["id"], users["mp"]["id"])["verification"]["ok"]
    with sqlite3.connect(fresh_db) as c:
        stored = c.execute("SELECT results_json FROM legal_citation_checks WHERE id=?", (ch["id"],)).fetchone()[0]
    assert ct["msa"]["id"] not in stored
    seen = as_user(ct, "alice").get(f"{ct['base']}/citation-checks/{ch['id']}").json()["check"]
    assert item(seen, "reference", "clause 8.1 of the Master")["note"] == "The document it pointed to was deleted."


def test_lawyers_notes_are_kept_through_an_attempted_rollback(ct, fresh_db):
    from app.legal.migrations import runner
    ch = run(ct)
    alice = as_user(ct, "alice")
    alice.post(f"{ct['base']}/citation-checks/{ch['id']}/notes", headers=csrf_headers(alice), json={"item": "a1", "note": "Checked in Westlaw"})
    with sqlite3.connect(fresh_db) as c:
        with pytest.raises(runner.MigrationError, match="citation checks"):
            runner.migrate_down(c, 25)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            c.execute("UPDATE legal_citation_notes SET note='changed'")


@pytest.mark.parametrize("text, sub, bad", [
    ("Smith v. Jones, 410 U.S. 113 (1973)", "us_case", []),
    ("Doe v. Roe Holdings, 12 F.4th 345, 350 (2d Cir. 2021)", "us_case", []),
    ("Acme v. Beta, 77 F.3d 10 (13th Cir. 1999)", "us_case", ["there is no 13th Cir."]),
    ("Acme v. Beta, 77 Fed. Rptr. 10 (2d Cir. 1999)", "us_case", ["Fed. Rptr. is not a known reporter"]),
    ("[2014] UKHL 9", "uk_neutral", ["the House of Lords stopped sitting in 2009"]),
    ("[1998] UKSC 4", "uk_neutral", ["UKSC gave no judgments before 2009"]),
    ("[2020] EWHC 1234 (Comm)", "uk_neutral", []),
    ("[1932] AC 562", "uk_report", []),
    ("section 2 of the Unfair Contract Terms Act 1977", "uk_act", []),
    ("Late Payment of Commercial Debts (Interest) Act 1998", "uk_act", []),
    ("Contracts (Rights of Third Parties) Act 1999", "uk_act", []),
    ("Sale of Goods Act 1979", "uk_act", []),
    ("Third Parties (Rights against Insurers) Act 2010", "uk_act", []),
    ("Directive 93/13/EEC", "eu_act", []),
    ("Case C-131/12", "cjeu_case", []),
    ("60 U.S.C. § 1", "us_code", ["the US Code has no title 60"]),
    ("29 C.F.R. Part 1910", "cfr", []),
    ("Fed. R. Civ. P. 81(c)", "us_rule", []),
    ("Fed. R. Evid. 401", "us_rule", []),
    ("Fed. R. Civ. P. 95", "us_rule", ["there is no Rule 95 of the Federal Rules of Civil Procedure"]),
    ("CPR r. 3.4(2)(a)", "uk_rule", []),
    ("CPR Part 36", "uk_rule", []),
    ("CPR Part 96", "uk_rule", ["the Civil Procedure Rules have no Part 96"]),
])
def test_the_parser_reads_authorities_and_flags_impossible_ones(text, sub, bad):
    from app.legal.citations import parse
    found = [f for f in parse.read(f"See {text} for this.") if f.kind == "authority"]
    assert [f.sub for f in found] == [sub] and found[0].malformed == bad
    if sub.endswith("_rule") or sub == "uk_act":
        assert found[0].text == text      # the whole rule or Act name (citations-s3.4f, s3.4g)


def test_words_before_an_act_name_are_not_part_of_it():
    from app.legal.citations import parse
    text = "See Sale of Goods Act 1979. Under the Companies Act 2006 and The Arbitration Act 1996, nothing changes."
    assert [f.text for f in parse.read(text) if f.kind == "authority"] == ["Sale of Goods Act 1979", "Companies Act 2006", "Arbitration Act 1996"]


def test_the_parser_reads_references_and_quotations():
    from app.legal.citations import parse
    text = ("Under Schedule 2 of the Lease and Article IV, see clause 9.2(b). 15 U.S.C. § 78j is an authority, not a reference.\n"
            "Clause 3 says that \"the Tenant shall [k]eep the premises . . . in good repair\". Short \"quotes\" are not checked.")
    found = parse.read(text)
    refs = [(f.sub, f.data["number"], f.data.get("sub"), f.data.get("document")) for f in found if f.kind == "reference"]
    assert refs == [("schedule", "2", None, "Lease"), ("article", "IV", None, None), ("clause", "9.2", "(b)", None)]
    quotes = [f for f in found if f.kind == "quote"]
    assert len(quotes) == 1 and quotes[0].data["reference"].text == "Clause 3"


def test_a_quotation_with_an_ellipsis_and_a_bracketed_capital_is_accurate(docs):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    lease = ("LEASE (fictional)\n\n3. Repair\n3.1 The Tenant shall keep the premises and all fixtures in good and substantial repair "
             "at all times during the Term.\n\n4. Use\n4.1 The premises may be used only as offices.")
    note = ("NOTE (fictional)\n\n1. Repair\n1.1 Clause 3.1 of the Lease says that \"[t]he Tenant shall keep the premises . . . in good and "
            "substantial repair at all times\". Clause 4.1 of the Lease says \"The premises may be used only as shops.\"")
    upload(alice, mid, "Lease.txt", lease, status="executed", title="Lease")
    n = upload(alice, mid, "note.txt", note, status="draft").json()["document"]
    index_all()
    r = alice.post(f"/api/legal/matters/{mid}/documents/{n['id']}/citation-check", headers=csrf_headers(alice))
    ch = r.json()["check"]
    good = item(ch, "quote", "good and substantial repair")
    assert good["status"] == "accurate" and good["target"]["clause"] == "3.1", good
    bad = item(ch, "quote", "only as shops")
    assert bad["status"] == "altered" and ["del", "offices."] in bad["diff"] and ["ins", "shops."] in bad["diff"]


def test_the_demo_memo_against_the_demo_agreement(docs):
    """The fixtures journey 25 uses: what a lawyer sees in the browser is checked here first."""
    from pathlib import Path
    fx = Path(__file__).resolve().parents[3] / "eval" / "fixtures" / "law"
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    msa = upload(alice, mid, "Master Services Agreement (DEMO).md", (fx / "Master Services Agreement (DEMO).md").read_text(),
                 mime="text/markdown", status="executed").json()["document"]
    memo = upload(alice, mid, "Advice memo on the MSA (DEMO).md", (fx / "Advice memo on the MSA (DEMO).md").read_text(),
                  mime="text/markdown", status="draft").json()["document"]
    index_all()
    ch = alice.post(f"/api/legal/matters/{mid}/documents/{memo['id']}/citation-check", headers=csrf_headers(alice)).json()["check"]

    def got(kind, starts):
        i = next(i for i in ch["items"] if i["kind"] == kind and i["text"].startswith(starts))
        return i["status"], (i["target"] or {}).get("clause"), (i["target"] or {}).get("document_id") == msa["id"]
    assert got("quote", "Invoices are payable net sixty (60) days") == ("altered", "4.2", True)
    assert got("quote", "within forty-eight (48) hours of becoming") == ("accurate", "6.1", True)
    assert got("reference", "Clause 8.1 of the Master Services Agreem") == ("resolved", "8.1", True)
    assert got("reference", "clause 8.3") == ("resolved", "8.3", True)          # the agreement named earlier in the paragraph
    assert got("reference", "Clause 18.3") == ("missing", None, True)
    assert item(ch, "reference", "Clause 18.3")["note"] == "The Master Services Agreement has no clause 18.3."
    assert {i["text"] for i in ch["items"] if i["kind"] == "authority"} >= {"17 C.F.R. § 240.10b-5", "15 U.S.C. § 78j(b)"}
    assert got("reference", "paragraph 1.1") == ("resolved", "1.1", False)
    fake = item(ch, "authority", "Pemberton v. Halloway")
    assert len(fake["malformed"]) == 3 and item(ch, "authority", "Ashby v. Corran")["malformed"] == []
    assert ch["summary"]["authorities"] == 5 and ch["summary"]["malformed"] == 1


def test_a_clause_known_only_by_its_markdown_heading_resolves(docs):
    """citations-s3.4c: the reader does not read "## 12. Termination" as clause 12; its sub-clauses imply it."""
    from pathlib import Path
    fx = Path(__file__).resolve().parents[3] / "eval" / "fixtures" / "law"
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    upload(alice, mid, "Master Services Agreement (DEMO).md", (fx / "Master Services Agreement (DEMO).md").read_text(),
           mime="text/markdown", status="executed")
    memo = upload(alice, mid, "note.txt", "NOTE (fictional)\n\n1. Exit\n1.1 Clause 12 of the Master Services Agreement governs exit; "
                  "clause 13 of the Master Services Agreement does not exist.", status="draft").json()["document"]
    index_all()
    ch = alice.post(f"/api/legal/matters/{mid}/documents/{memo['id']}/citation-check", headers=csrf_headers(alice)).json()["check"]
    twelve = item(ch, "reference", "Clause 12 of")
    assert twelve["status"] == "resolved" and twelve["target"]["clause"] == "12" and "12.1 Either party" in twelve["target"]["passage"]
    assert item(ch, "reference", "clause 13 of")["status"] == "missing"


def test_a_heading_is_not_a_reference_to_itself():
    from app.legal.citations import parse
    text = ("Article 1 Grant\n1.1 The licence is subject to Article 3 and Annex A.\n\n## Schedule 1 — Fees\nEUR 10 (fictional).\n"
            "Clause 3 says that the parties agree.\nSchedule 2\n")
    refs = [f.text for f in parse.read(text) if f.kind == "reference"]
    assert refs == ["Article 3", "Annex A", "Clause 3"]


def test_a_defined_short_name_does_not_pair_quotation_marks_the_wrong_way():
    """Held-out run 1 (citations-s3.4): `(the "SPA")` paired the wrong quotation marks for the rest of the memo."""
    from app.legal.citations import parse
    text = ('1.1 This memo concerns the Share Purchase Agreement (the "SPA").\n'
            '2.1 Under section 6.3 of the Share Purchase Agreement, claims must be notified "within eighteen (18) months after Completion".\n'
            '2.2 By clause 7.2 of the SPA, small claims are excluded; paragraph 1 of Schedule 4 of the SPA is the tax warranty.\n'
            "2.3 Clause 7.1 of the SPA caps liability at \"the Purchase Price\". UK style: clause 3 says ‘the party’s notice must be in writing’.")
    found = parse.read(text)
    quotes = [(f.text, f.data["reference"].text if f.data["reference"] else None) for f in found if f.kind == "quote"]
    assert quotes == [("within eighteen (18) months after Completion", "section 6.3 of the Share Purchase Agreement"),
                      ("the Purchase Price", "Clause 7.1 of the SPA"), ("the party’s notice must be in writing", "clause 3")]
    refs = {f.text: f.data for f in found if f.kind == "reference"}
    assert refs["clause 7.2 of the SPA"]["document"] == "Share Purchase Agreement"
    assert refs["paragraph 1 of Schedule 4"]["within"] == "Schedule 4" and refs["paragraph 1 of Schedule 4"]["document"] == "Share Purchase Agreement"
