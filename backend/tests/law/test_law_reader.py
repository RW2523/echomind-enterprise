"""S2.1c — the document reader and per-version status (RAG-01).

The reader returns a version's stored text with its pages, lines, clauses, defined terms and marks,
under the same rules as downloading the file: matter access, source permissions, no caching, and
an audit record without content. Every version keeps its own status, so "v3 executed, v2 draft" is
known when answers must prefer signed text (S2.2).
"""
from __future__ import annotations

import json
import sqlite3

from conftest import as_user, csrf_headers, index_all, upload
from fixtures_s21 import contract_docx

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def reader_url(mid, doc):
    return f"/api/legal/matters/{mid}/documents/{doc['id']}/versions/{doc['current_version']['id']}/reader"


def test_the_reader_returns_the_text_and_everything_found_in_it(docs, fresh_db):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "harrowgate-msa.docx", contract_docx(), mime=DOCX_MIME).json()["document"]
    index_all()
    r = alice.get(reader_url(mid, doc))
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()["reader"]
    with sqlite3.connect(fresh_db) as c:
        stored = c.execute("SELECT text FROM legal_version_text").fetchone()[0]
    assert body["text"] == stored
    assert [c["number"] for c in body["clauses"]][:3] == ["Article I", "12", "12.1"]
    assert {t["term"] for t in body["terms"]} == {"Services"}
    kinds = sorted(m["kind"] for m in body["marks"])
    assert kinds == ["comment", "hidden_text", "tracked_delete", "tracked_insert"]
    comment = next(m for m in body["marks"] if m["kind"] == "comment")
    assert body["text"][comment["start"]:comment["end"]] == "Service credits are the sole remedy."
    assert body["lines"] and body["pages"] == [] and body["version"]["number"] == 1
    with sqlite3.connect(fresh_db) as c:
        row = c.execute("SELECT action, details_json FROM audit_events WHERE action='document.read_text'").fetchone()
    assert row and "Service credits" not in (row[1] or "")


def test_the_reader_follows_matter_access_and_source_permissions(docs):
    alice, bob, carol = as_user(docs, "alice"), as_user(docs, "bob"), as_user(docs, "carol")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "harrowgate-msa.docx", contract_docx(), mime=DOCX_MIME).json()["document"]
    index_all()
    assert carol.get(reader_url(mid, doc)).status_code == 200             # read-only member
    assert bob.get(reader_url(mid, doc)).status_code == 404               # not on the matter
    from app.legal.documents import repo
    repo.set_acl(doc["id"], {docs["users"]["alice"]["id"]})               # the source names alice only
    assert carol.get(reader_url(mid, doc)).status_code == 404
    assert alice.get(reader_url(mid, doc)).status_code == 200


def test_a_version_read_before_the_reader_existed_asks_to_be_read_again(docs, fresh_db):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "note.txt", "A short fictional note.").json()["document"]
    index_all()
    with sqlite3.connect(fresh_db) as c:
        c.execute("DELETE FROM legal_version_text")
        c.commit()
    r = alice.get(reader_url(mid, doc))
    assert r.status_code == 409 and "Read it again" in r.json()["detail"]


def test_each_version_keeps_its_own_status(docs):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    v1 = upload(alice, mid, "msa-signed.txt", "Signed fictional MSA.", status="executed").json()["document"]
    assert v1["status"] == "executed" and v1["current_version"]["status"] == "executed"
    v2 = upload(alice, mid, "msa-redraft.txt", "Redraft of the fictional MSA.", status="draft", version_of=v1["id"]).json()["document"]
    assert v2["status"] == "draft" and v2["current_version"]["status"] == "draft"
    versions = alice.get(f"/api/legal/matters/{mid}/documents/{v1['id']}/versions").json()["versions"]
    assert {v["number"]: v["status"] for v in versions} == {1: "executed", 2: "draft"}
    r = alice.patch(f"/api/legal/matters/{mid}/documents/{v1['id']}", headers=csrf_headers(alice),
                    json={"status": "final", "version": v2["version"]})
    assert r.status_code == 200 and r.json()["document"]["current_version"]["status"] == "final"
    versions = alice.get(f"/api/legal/matters/{mid}/documents/{v1['id']}/versions").json()["versions"]
    assert {v["number"]: v["status"] for v in versions} == {1: "executed", 2: "final"}


def test_progress_and_flags_are_on_the_version_while_and_after_reading(docs):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "harrowgate-msa.docx", contract_docx(), mime=DOCX_MIME).json()["document"]
    assert doc["current_version"]["progress"] is None and doc["current_version"]["index_status"] == "queued"
    index_all()
    v = alice.get(f"/api/legal/matters/{mid}/documents/{doc['id']}").json()["document"]["current_version"]
    assert v["progress"] == {"stage": "done"} and v["flags"]["comments"] == 1
    assert json.dumps(v).count("Service credits") == 0                     # flags are counts, never text
