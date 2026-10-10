"""S3.8 — governance checks (DOC-07; decision D-052).

Run with a due-diligence checklist that asks for them, governance checks read the company's own documents
against each other, by rules (no model):
- **share capital:** the shares in the register of members against those the board approved and the classes
  the articles create;
- **authority:** each contract's value against the signing limit of the person who signed it, in the
  delegation of authority;
- **constitution:** a policy against the articles (quorum, notice of board meetings, number of directors);
- **IP chain:** each founder named in the company's documents against the IP assignments.
Each finding quotes both documents. Findings are advisory and preliminary until a reviewer decides; nothing is
decided automatically.
"""
from __future__ import annotations

import sqlite3

import pytest

from conftest import as_user, csrf_headers, index_all, upload

ARTICLES = ("ARTICLES OF ASSOCIATION of Tidewater Holdings Ltd (fictional)\n\n"
            "1. Share capital\n1.1 The share capital of the company consists of ordinary shares and A preference shares.\n\n"
            "2. Directors\n2.1 The number of directors shall not be less than two.\n2.2 The quorum for board meetings is two directors.\n"
            "2.3 Not less than seven (7) days' notice shall be given of every board meeting.\n")
REGISTER = ("REGISTER OF MEMBERS of Tidewater Holdings Ltd (fictional)\n\n1. A. Moreau: 600 ordinary shares.\n2. B. Lind: 400 ordinary shares.\n"
            "3. C. Osei: 150 ordinary shares.\n4. Northgate Capital LLP: 200 B ordinary shares.\n")
MINUTES = ("MINUTES OF A BOARD MEETING of Tidewater Holdings Ltd held on 10 January 2026 (fictional)\n\n"
           "1. Founders\n1.1 The founders of the company are A. Moreau and B. Lind.\n\n"
           "2. Allotment\n2.1 The board approved the allotment of 1,000 ordinary shares to the founders.\n")
DELEGATION = ("DELEGATION OF AUTHORITY of Tidewater Holdings Ltd (fictional)\n\n"
              "1. Contracts\n1.1 The Finance Director may sign contracts with a value of up to GBP 50,000.\n"
              "1.2 Contracts with a value above GBP 50,000 must be signed by two directors.\n")
SUPPLY = ("SUPPLY AGREEMENT (fictional)\n\nThis Agreement is made between Northwind Components Ltd (the \"Supplier\") and Tidewater Holdings Ltd (the \"Customer\").\n\n"
          "1. Price\n1.1 The total price payable under this Agreement is GBP 120,000.\n\n"
          "2. Signature\n2.1 Signed for Tidewater Holdings Ltd by J. Park, Finance Director.\n")
SMALL = ("SERVICES AGREEMENT (fictional)\n\nThis Agreement is made between Brightline Data, Inc. (the \"Provider\") and Tidewater Holdings Ltd (the \"Client\").\n\n"
         "1. Fees\n1.1 The total fees payable under this Agreement are GBP 18,000.\n\n2. Signature\n2.1 Signed for Tidewater Holdings Ltd by J. Park, Finance Director.\n")
POLICY = ("BOARD PROCEDURES POLICY of Tidewater Holdings Ltd (fictional)\n\n"
          "1. Meetings\n1.1 A board meeting is quorate when one director is present.\n1.2 Board meetings may be called on two (2) days' notice.\n")
IP = ("IP ASSIGNMENT DEED (fictional)\n\n1. Assignment\n1.1 A. Moreau assigns to Tidewater Holdings Ltd all intellectual property created for the company.\n")
CHECKLIST = {
    "name": "Governance review (test)", "expected_law": None,
    "items": [
        {"id": "articles", "section": "Corporate", "title": "Articles of association", "expect": ["articles of association"]},
        {"id": "register", "section": "Corporate", "title": "Register of members", "expect": ["register of members"]},
        {"id": "minutes", "section": "Corporate", "title": "Board minutes", "expect": ["minutes"]},
        {"id": "authority", "section": "Corporate", "title": "Delegation of authority", "expect": ["delegation of authority"]},
        {"id": "policies", "section": "Corporate", "title": "Board policies", "expect": ["policy"]},
        {"id": "contracts", "section": "Commercial", "title": "Material contracts", "expect": ["supply agreement", "services agreement"]},
        {"id": "ip", "section": "IP", "title": "IP assignments", "expect": ["ip assignment"]},
    ],
    "governance": ["share_capital", "authority", "constitution", "ip_chain"],
}
FILES = (("Articles of Association.txt", ARTICLES), ("Register of members.txt", REGISTER), ("Board minutes January 2026.txt", MINUTES),
         ("Delegation of authority.txt", DELEGATION), ("Supply Agreement - Northwind.txt", SUPPLY), ("Services Agreement - Brightline.txt", SMALL),
         ("Board procedures policy.txt", POLICY), ("IP Assignment - A Moreau.txt", IP))


@pytest.fixture
def gv(docs):
    from app.legal.matters import repo
    alice, mp = as_user(docs, "alice"), as_user(docs, "mp")
    mid = docs["matters"]["M-101"]["id"]
    repo.add_membership(mid, docs["users"]["dave"]["id"], "reviewer", granted_by=docs["users"]["alice"]["id"])
    ids = {name: upload(alice, mid, name, text, status="executed").json()["document"]["id"] for name, text in FILES}
    index_all()
    c = alice.post("/api/legal/diligence/checklists", headers=csrf_headers(alice), json=CHECKLIST).json()["checklist"]
    c = mp.post(f"/api/legal/diligence/checklists/{c['id']}/approve", headers=csrf_headers(mp), json={"content_sha256": c["content_sha256"]}).json()["checklist"]
    docs.update(mid=mid, ids=ids, base=f"/api/legal/matters/{mid}", checklist=c)
    return docs


def run(w):
    from app.legal.jobs import worker
    alice = as_user(w, "alice")
    r = alice.post(f"{w['base']}/diligence/runs", headers=csrf_headers(alice), json={"checklist_id": w["checklist"]["id"]})
    assert r.status_code == 200, r.text
    worker.run_pending()
    return alice.get(f"{w['base']}/diligence/runs/{r.json()['run']['id']}").json()


def found(view, kind):
    return [i for i in view["issues"] if i["kind"] == kind]


def test_a_checklist_asks_for_governance_checks_by_name(gv):
    assert gv["checklist"]["governance"] == ["share_capital", "authority", "constitution", "ip_chain"]
    alice = as_user(gv, "alice")
    bad = alice.post("/api/legal/diligence/checklists", headers=csrf_headers(alice), json={**CHECKLIST, "governance": ["horoscope"]})
    assert bad.status_code == 422


def test_shares_in_the_register_are_checked_against_approvals_and_the_articles(gv):
    view = run(gv)
    caps = found(view, "share_capital")
    titles = sorted(i["title"] for i in caps)
    assert titles == ["150 ordinary shares in the register have no approval", "B ordinary shares are not a class the articles create"]
    over = next(i for i in caps if i["title"].startswith("150"))
    assert over["document_title"] == "Register of members.txt" and "1,000 ordinary shares" in over["related"][0]["quote"]
    assert over["related"][0]["document_title"] == "Board minutes January 2026.txt" and over["severity"] == "high"
    cls = next(i for i in caps if i["title"].startswith("B ordinary"))
    assert "Northgate Capital LLP: 200 B ordinary shares" in cls["evidence"]["quote"]
    assert "ordinary shares and A preference shares" in cls["related"][0]["quote"]
    assert cls["detail"] == "The register shows 200 B ordinary shares; the articles create ordinary and A preference shares only."


def test_a_contract_signed_above_the_signer_s_limit_is_raised_and_one_within_it_is_not(gv, fresh_db):
    view = run(gv)
    auth = found(view, "authority")
    assert len(auth) == 1
    a = auth[0]
    assert a["document_title"] == "Supply Agreement - Northwind.txt" and a["severity"] == "high"
    assert "GBP 120,000" in a["detail"] and "GBP 50,000" in a["detail"] and "Finance Director" in a["detail"]
    assert a["related"][0]["document_title"] == "Delegation of authority.txt" and "up to GBP 50,000" in a["related"][0]["quote"]
    with sqlite3.connect(fresh_db) as c:
        text = c.execute("SELECT text FROM legal_version_text WHERE version_id=?", (a["related"][0]["version_id"],)).fetchone()[0]
    r = a["related"][0]
    assert text[r["start"]:r["end"]] == r["quote"]


def test_a_policy_that_contradicts_the_articles_is_raised(gv):
    view = run(gv)
    con = {i["title"]: i for i in found(view, "constitution")}
    assert set(con) == {"Board quorum: one director in the policy, two in the articles",
                        "Notice of board meetings: 2 days in the policy, 7 days in the articles"}
    q = con["Board quorum: one director in the policy, two in the articles"]
    assert q["document_title"] == "Board procedures policy.txt" and q["related"][0]["document_title"] == "Articles of Association.txt"
    assert q["severity"] == "medium"


def test_a_founder_without_an_ip_assignment_is_raised(gv):
    view = run(gv)
    ip = found(view, "ip_chain")
    assert [i["title"] for i in ip] == ["No IP assignment from B. Lind"]
    assert ip[0]["document_title"] == "Board minutes January 2026.txt" and "A. Moreau and B. Lind" in ip[0]["evidence"]["quote"]


def test_findings_are_advisory_preliminary_and_hidden_with_either_document(gv):
    from app.legal.documents import repo
    view = run(gv)
    gov = [i for i in view["issues"] if i["kind"] in ("share_capital", "authority", "constitution", "ip_chain")]
    assert gov and all(i["status"] == "preliminary" for i in gov)
    repo.set_acl(gv["ids"]["Delegation of authority.txt"], {gv["users"]["alice"]["id"]})          # the related document
    seen = as_user(gv, "carol").get(f"{gv['base']}/diligence/runs/{view['run']['id']}").json()
    assert found(seen, "authority") == [] and "up to GBP 50,000" not in str(seen)


def test_purging_a_related_document_removes_the_finding(gv, fresh_db):
    from app.legal.records import deletion
    view = run(gv)
    doc = gv["ids"]["Delegation of authority.txt"]
    org, users = gv["matters"]["M-101"]["org_id"], gv["users"]
    d = deletion.request(org, gv["mid"], "document", doc, "Superseded", users["alice"]["id"])
    assert d["report"]["diligence_issues"] >= 1
    assert deletion.purge(d["id"], users["mp"]["id"])["verification"]["ok"]
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT COUNT(*) FROM legal_dd_issues WHERE related_json LIKE ?", (f"%{doc}%",)).fetchone()[0] == 0
