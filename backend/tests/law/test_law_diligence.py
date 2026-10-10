"""S3.7 — due diligence (DOC-06; decision D-051).

A firm checklist (versioned, approved by a managing partner who did not write it) is run against a
matter's documents as a background job: each item is matched to the documents that answer it, the
items nothing answers are the missing-document list, and the matched contracts are read for the issues
a reviewer looks for (change of control, assignment, termination, governing law, liability, renewal and
expiry), each quoting its sentence. Every issue is preliminary until a reviewer verifies or dismisses it.
"""
from __future__ import annotations

import sqlite3

import pytest

from conftest import as_user, csrf_headers, index_all, upload

ARTICLES = ("ARTICLES OF ASSOCIATION of Tidewater Holdings Ltd (fictional)\n\n1. Shares\n1.1 The share capital is divided into ordinary shares.\n"
            "2. Directors\n2.1 The number of directors shall not be less than two.\n")
MINUTES = ("MINUTES OF A MEETING OF THE BOARD of Tidewater Holdings Ltd (fictional)\n\n1. Present\n1.1 A. Moreau (chair) and B. Lind.\n"
           "2. Resolutions\n2.1 The board approved the supply agreement with Northwind Components Ltd.\n")
SUPPLY = ("SUPPLY AGREEMENT (fictional)\n\nThis Agreement is made on 1 December 2024 between Northwind Components Ltd (the \"Supplier\") and "
          "Tidewater Holdings Ltd (the \"Customer\").\n\n"
          "1. Term\n1.1 This Agreement continues for an initial term of two (2) years from the date of this Agreement.\n"
          "1.2 It renews automatically for one year unless either party gives notice of non-renewal at least sixty (60) days before the end of the term.\n\n"
          "2. Change of control\n2.1 The Supplier may terminate this Agreement on written notice if the Customer undergoes a change of control.\n\n"
          "3. Assignment\n3.1 The Customer shall not assign this Agreement without the prior written consent of the Supplier.\n\n"
          "4. Termination\n4.1 Either party may terminate this Agreement for convenience on thirty (30) days' written notice.\n\n"
          "5. Liability\n5.1 The Supplier's total liability shall not exceed the fees paid in the preceding twelve (12) months.\n\n"
          "6. Law\n6.1 This Agreement is governed by the laws of the State of New York.\n")
LEASE = ("LEASE (fictional)\n\nThis Lease is made between Harbour Estates Limited (the \"Landlord\") and Tidewater Holdings Ltd (the \"Tenant\").\n\n"
         "1. Term\n1.1 \"Term Commencement Date\" means 25 March 2026.\n"
         "2. Break\n2.1 The Tenant may terminate this Lease on the fifth anniversary of the Term Commencement Date by giving not less than six (6) months' notice.\n"
         "3. Law\n3.1 This Lease is governed by the laws of England and Wales.\n")
CHECKLIST = {
    "name": "Share acquisition — core documents (test)", "expected_law": "England and Wales",
    "items": [
        {"id": "corp-articles", "section": "Corporate", "title": "Articles of association", "expect": ["articles of association", "constitution"]},
        {"id": "corp-sha", "section": "Corporate", "title": "Shareholders' agreement", "expect": ["shareholders agreement", "shareholders' agreement"]},
        {"id": "corp-minutes", "section": "Corporate", "title": "Board minutes", "expect": ["minutes"]},
        {"id": "contracts", "section": "Commercial", "title": "Material contracts", "expect": ["supply agreement", "services agreement", "distribution agreement"],
         "questions": ["change_of_control", "assignment", "termination", "governing_law", "liability", "renewal"]},
        {"id": "property", "section": "Property", "title": "Leases", "expect": ["lease"], "questions": ["termination", "governing_law"]},
        {"id": "people", "section": "Employment", "title": "Key employment contracts", "expect": ["employment contract", "service agreement"]},
        {"id": "insurance", "section": "Insurance", "title": "Insurance policies", "expect": ["insurance policy", "policy schedule"]},
    ],
}


@pytest.fixture
def dd(docs):
    from app.legal.matters import repo
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    repo.add_membership(mid, docs["users"]["dave"]["id"], "reviewer", granted_by=docs["users"]["alice"]["id"])
    ids = {}
    for name, text in (("Articles of Association.txt", ARTICLES), ("Board minutes January 2026.txt", MINUTES),
                       ("Supply Agreement - Northwind.txt", SUPPLY), ("Office Lease.txt", LEASE)):
        ids[name] = upload(alice, mid, name, text, status="executed").json()["document"]["id"]
    index_all()
    docs.update(mid=mid, ids=ids, base=f"/api/legal/matters/{mid}")
    return docs


def checklist(w):
    """A checklist written by an attorney (alice) and approved by a managing partner (mp)."""
    alice, mp = as_user(w, "alice"), as_user(w, "mp")
    r = alice.post("/api/legal/diligence/checklists", headers=csrf_headers(alice), json=CHECKLIST)
    assert r.status_code == 200, r.text
    c = r.json()["checklist"]
    assert c["status"] == "draft"
    a = mp.post(f"/api/legal/diligence/checklists/{c['id']}/approve", headers=csrf_headers(mp), json={"content_sha256": c["content_sha256"]})
    assert a.status_code == 200, a.text
    return a.json()["checklist"]


def start(w, c, by="alice"):
    from app.legal.jobs import worker
    u = as_user(w, by)
    r = u.post(f"{w['base']}/diligence/runs", headers=csrf_headers(u), json={"checklist_id": c["id"]})
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    assert run["status"] == "queued"
    worker.run_pending()
    got = u.get(f"{w['base']}/diligence/runs/{run['id']}")
    assert got.status_code == 200, got.text
    return got.json()


def issue(view, kind, doc_contains=""):
    return [i for i in view["issues"] if i["kind"] == kind and doc_contains in (i["document_title"] or "")]


def test_a_checklist_is_a_version_approved_by_another_managing_partner(dd):
    alice, mp = as_user(dd, "alice"), as_user(dd, "mp")
    r = alice.post("/api/legal/diligence/checklists", headers=csrf_headers(alice), json=CHECKLIST).json()["checklist"]
    assert alice.post(f"/api/legal/diligence/checklists/{r['id']}/approve", headers=csrf_headers(alice),
                      json={"content_sha256": r["content_sha256"]}).status_code == 403          # attorneys do not approve
    assert mp.post(f"/api/legal/diligence/checklists/{r['id']}/approve", headers=csrf_headers(mp),
                   json={"content_sha256": "0" * 64}).status_code == 409                       # not the version read
    c = checklist(dd)
    assert c["status"] == "approved" and c["approved_by"] == "mp" and c["version"] == 1
    listed = as_user(dd, "carol").get("/api/legal/diligence/checklists").json()["checklists"]
    assert [x["status"] for x in listed if x["name"] == CHECKLIST["name"]] == ["approved"]             # drafts only for those who manage


def test_a_run_lists_what_answers_each_item_and_what_is_missing(dd):
    view = start(dd, checklist(dd))
    run = view["run"]
    assert run["status"] == "done" and run["checklist"]["version"] == 1
    items = {i["item_id"]: i for i in view["items"]}
    assert [d["title"] for d in items["corp-articles"]["documents"]] == ["Articles of Association.txt"]
    assert [d["title"] for d in items["contracts"]["documents"]] == ["Supply Agreement - Northwind.txt"]
    assert [d["title"] for d in items["property"]["documents"]] == ["Office Lease.txt"]
    assert sorted(i["title"] for i in view["missing"]) == ["Insurance policies", "Key employment contracts", "Shareholders' agreement"]
    assert view["summary"]["missing"] == 3 and view["summary"]["items"] == 7


def test_issues_are_found_with_their_sentences_and_stay_preliminary(dd, fresh_db):
    view = start(dd, checklist(dd))
    coc = issue(view, "change_of_control", "Supply")
    assert len(coc) == 1 and coc[0]["severity"] == "high" and coc[0]["status"] == "preliminary"
    with sqlite3.connect(fresh_db) as c:
        text = c.execute("SELECT t.text FROM legal_version_text t WHERE t.version_id=?", (coc[0]["version_id"],)).fetchone()[0]
    e = coc[0]["evidence"]
    assert text[e["start"]:e["end"]] == e["quote"] and "change of control" in e["quote"]
    asg = issue(view, "assignment", "Supply")[0]
    assert asg["severity"] == "medium" and asg["detail"].startswith("Customer shall not assign this Agreement without the prior written consent")
    assert issue(view, "termination", "Supply")[0]["evidence"]["quote"].startswith("4.1 Either party may terminate")
    law = issue(view, "governing_law", "Supply")
    assert len(law) == 1 and "New York" in law[0]["title"] and "England and Wales" in law[0]["detail"]
    assert issue(view, "governing_law", "Lease") == []                                       # the law the checklist expects
    assert issue(view, "liability", "Supply")[0]["evidence"]["quote"].startswith("5.1")
    ren = issue(view, "renewal", "Supply")
    assert len(ren) == 1 and "1 October 2026" in ren[0]["detail"]                             # 60 days before 30 November 2026
    assert issue(view, "termination", "Lease")[0]["severity"] == "medium"                      # the tenant's break right
    assert all(i["status"] == "preliminary" for i in view["issues"])
    assert issue(view, "change_of_control", "Articles") == []                                  # only the items' questions are asked


def test_a_reviewer_verifies_or_dismisses_with_a_reason_and_the_trail_is_kept(dd, fresh_db):
    view = start(dd, checklist(dd))
    coc = issue(view, "change_of_control", "Supply")[0]
    law = issue(view, "governing_law", "Supply")[0]
    url = f"{dd['base']}/diligence/issues"
    carol, dave = as_user(dd, "carol"), as_user(dd, "dave")
    assert carol.patch(f"{url}/{coc['id']}", headers=csrf_headers(carol), json={"expected_version": 1, "action": "verify"}).status_code == 403
    v = dave.patch(f"{url}/{coc['id']}", headers=csrf_headers(dave), json={"expected_version": 1, "action": "verify", "reason": "Consent needed"})
    assert v.status_code == 200, v.text
    assert v.json()["issue"]["status"] == "verified" and v.json()["issue"]["reviewer"] == "dave"
    assert dave.patch(f"{url}/{law['id']}", headers=csrf_headers(dave), json={"expected_version": 1, "action": "dismiss"}).status_code == 422
    d = dave.patch(f"{url}/{law['id']}", headers=csrf_headers(dave), json={"expected_version": 1, "action": "dismiss", "reason": "New York law accepted by the client"})
    assert d.json()["issue"]["status"] == "dismissed"
    assert dave.patch(f"{url}/{coc['id']}", headers=csrf_headers(dave), json={"expected_version": 1, "action": "dismiss", "reason": "x x x"}).status_code == 409
    trail = as_user(dd, "alice").get(f"{url}/{coc['id']}").json()["trail"]
    assert [t["action"] for t in trail] == ["found", "verify"] and trail[1]["by"] == "dave" and trail[1]["reason"] == "Consent needed"
    after = as_user(dd, "alice").get(f"{dd['base']}/diligence/runs/{view['run']['id']}").json()
    assert after["summary"]["verified"] == 1 and after["summary"]["dismissed"] == 1
    with sqlite3.connect(fresh_db) as c:
        details = " ".join(r[0] or "" for r in c.execute("SELECT details_json FROM audit_events WHERE action LIKE 'diligence.%'"))
    assert "Consent needed" not in details and "New York" not in details


def test_issues_on_a_document_someone_cannot_see_are_not_shown_to_them(dd):
    from app.legal.documents import repo
    view = start(dd, checklist(dd))
    repo.set_acl(dd["ids"]["Supply Agreement - Northwind.txt"], {dd["users"]["alice"]["id"]})
    seen = as_user(dd, "carol").get(f"{dd['base']}/diligence/runs/{view['run']['id']}").json()
    assert not any("Supply" in (i["document_title"] or "") for i in seen["issues"])
    items = {i["item_id"]: i for i in seen["items"]}
    assert items["contracts"]["documents"] == [] and "Supply Agreement" not in str(seen)
    assert as_user(dd, "bob").get(f"{dd['base']}/diligence/runs/{view['run']['id']}").status_code == 404


def test_purging_a_document_takes_its_issues_and_its_name_from_every_run(dd, fresh_db):
    from app.legal.records import deletion
    view = start(dd, checklist(dd))
    doc = dd["ids"]["Supply Agreement - Northwind.txt"]
    org, users = dd["matters"]["M-101"]["org_id"], dd["users"]
    d = deletion.request(org, dd["mid"], "document", doc, "Wrong data room", users["alice"]["id"])
    assert d["report"]["diligence_issues"] >= 5
    assert deletion.purge(d["id"], users["mp"]["id"])["verification"]["ok"]
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT COUNT(*) FROM legal_dd_issues WHERE document_id=?", (doc,)).fetchone()[0] == 0
        results = c.execute("SELECT results_json FROM legal_dd_runs WHERE id=?", (view["run"]["id"],)).fetchone()[0]
    assert doc not in results and "Northwind" not in results


def test_only_contributors_start_a_run_and_only_with_an_approved_checklist(dd):
    alice, carol = as_user(dd, "alice"), as_user(dd, "carol")
    draft = alice.post("/api/legal/diligence/checklists", headers=csrf_headers(alice), json=CHECKLIST).json()["checklist"]
    assert alice.post(f"{dd['base']}/diligence/runs", headers=csrf_headers(alice), json={"checklist_id": draft["id"]}).status_code == 409
    c = checklist(dd)
    assert carol.post(f"{dd['base']}/diligence/runs", headers=csrf_headers(carol), json={"checklist_id": c["id"]}).status_code == 403


def test_s37c_a_termination_right_without_a_notice_period_is_raised():
    """Found by held-out run 2 (diligence-s3.7b), to be measured on a fresh held-out set at the Phase 3 gate."""
    from datetime import date
    from app.legal.diligence import run as ddrun
    text = ("DISTRIBUTION AGREEMENT (fictional)\n\n4. Targets\n4.1 The Supplier may terminate this Agreement if the Distributor fails to "
            "meet the Minimum Purchase Target.\n")
    rows = [{"number": "4", "heading": "Targets", "level": 1, "parent": None, "start": text.index("4. Targets"), "end": len(text)},
            {"number": "4.1", "heading": None, "level": 2, "parent": "4", "start": text.index("4.1"), "end": len(text)}]
    out = ddrun.ask(["termination"], text, rows, None, date(2026, 10, 10))
    assert [(i["kind"], i["severity"], i["clause"]) for i in out] == [("termination", "low", "4.1")]

