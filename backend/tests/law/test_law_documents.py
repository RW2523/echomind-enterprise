"""S1.6 / SEC-03 (documents) — matter documents: import, encryption at rest, versions, limits, audit."""
from __future__ import annotations

import io
import json
import os
import sqlite3
import zipfile

import pytest

from conftest import as_user, csrf_headers, index_all, search_in, upload

MSA = ("MASTER SERVICES AGREEMENT between Harrowgate Ltd and Voss-Lindqvist AB.\n\n"
       "Clause 9.2 Liability. The aggregate liability of the Supplier shall not exceed the PELICAN-4471 cap "
       "of two times the annual fees.\n\nClause 12 Termination. Either party may terminate on ninety days notice.")


def _doc_rows(db, table="legal_documents"):
    with sqlite3.connect(db) as c:
        return c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _blob_paths(fresh_db):
    root = fresh_db.parent / "legal" / "blobs"
    return [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]


# ── Import, storage, download ────────────────────────────────────────────────────
def test_import_stores_the_file_encrypted_and_indexes_it(docs, fresh_db):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    r = upload(alice, mid, "harrowgate-msa.txt", MSA, title="Harrowgate MSA")
    assert r.status_code == 202, r.text
    doc = r.json()["document"]
    assert doc["title"] == "Harrowgate MSA" and doc["current_version"]["index_status"] == "queued"
    blobs = _blob_paths(fresh_db)
    assert len(blobs) == 1 and b"PELICAN" not in open(blobs[0], "rb").read()      # encrypted at rest
    assert index_all() == 1
    listed = alice.get(f"/api/legal/matters/{mid}/documents").json()["documents"]
    assert [d["current_version"]["index_status"] for d in listed] == ["indexed"]
    assert listed[0]["current_version"]["chunk_count"] >= 1
    vid = listed[0]["current_version"]["id"]
    f = alice.get(f"/api/legal/matters/{mid}/documents/{doc['id']}/versions/{vid}/file")
    assert f.status_code == 200 and f.content == MSA.encode()
    assert f.headers["cache-control"] == "no-store"
    assert f.headers["x-content-type-options"] == "nosniff"
    assert f.headers["content-disposition"].startswith("attachment;")


def test_a_tampered_or_swapped_file_is_refused(docs, fresh_db):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    a = upload(alice, mid, "a.txt", "first document text").json()["document"]
    b = upload(alice, mid, "b.txt", "second document text").json()["document"]
    pa, pb = (os.path.join(fresh_db.parent, "legal", "blobs", mid, d["current_version"]["id"] + ".bin") for d in (a, b))
    with open(pa, "rb") as f:
        blob_a = f.read()
    with open(pb, "wb") as f:                                   # someone swaps the stored files
        f.write(blob_a)
    r = alice.get(f"/api/legal/matters/{mid}/documents/{b['id']}/versions/{b['current_version']['id']}/file")
    assert r.status_code == 500 and "integrity" in r.json()["detail"]
    flipped = bytearray(blob_a)
    flipped[-1] ^= 1
    with open(pa, "wb") as f:
        f.write(bytes(flipped))
    r = alice.get(f"/api/legal/matters/{mid}/documents/{a['id']}/versions/{a['current_version']['id']}/file")
    assert r.status_code == 500 and "integrity" in r.json()["detail"]


@pytest.mark.parametrize("name, content, mime, status", [
    ("tool.exe", b"MZ\x90\x00binary", "application/octet-stream", 415),
    ("fake.pdf", b"just text pretending", "application/pdf", 415),
    ("notes.txt", b"text with a \x00 null", "text/plain", 415),
    ("latin1.txt", "café".encode("latin-1"), "text/plain", 415),
    ("fake.docx", b"PK\x03\x04not really a word file", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 415),
    ("mismatch.txt", b"plain words", "application/pdf", 415),
    ("empty.txt", b"", "text/plain", 422),
])
def test_uploads_must_be_what_they_claim(docs, name, content, mime, status):
    alice = as_user(docs, "alice")
    r = upload(alice, docs["matters"]["M-101"]["id"], name, content, mime=mime)
    assert r.status_code == status, r.text


def test_size_limit(docs, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_UPLOAD_MAX_MB", 1)
    alice = as_user(docs, "alice")
    r = upload(alice, docs["matters"]["M-101"]["id"], "big.txt", "a" * (1024 * 1024 + 10))
    assert r.status_code == 413


def test_a_real_word_file_is_accepted(docs):
    from docx import Document
    buf = io.BytesIO()
    d = Document()
    d.add_paragraph("Side letter. The OSPREY-5520 rebate applies from 1 January.")
    d.save(buf)
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    r = upload(alice, mid, "side-letter.docx", buf.getvalue(),
               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert r.status_code == 202, r.text
    index_all()
    hits = search_in(alice, mid, "OSPREY-5520 rebate").json()["passages"]
    assert hits and "OSPREY-5520" in hits[0]["text"]


def test_a_zip_bomb_dressed_as_word_is_refused(docs, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_DOCX_MAX_UNCOMPRESSED_MB", 2)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", "0" * (3 * 1024 * 1024))          # 3 MB unpacked from a few KB
    alice = as_user(docs, "alice")
    r = upload(alice, docs["matters"]["M-101"]["id"], "bomb.docx", buf.getvalue(),
               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    assert r.status_code == 415


# ── Who may import and read ──────────────────────────────────────────────────────
def test_only_contributors_import_and_only_members_read(docs):
    mid = docs["matters"]["M-101"]["id"]
    carol, bob, alice = (as_user(docs, n) for n in ("carol", "bob", "alice"))
    assert upload(carol, mid, "c.txt", "read only member").status_code == 403
    assert upload(bob, mid, "b.txt", "not a member").status_code == 404
    doc = upload(alice, mid, "a.txt", "lead imports").json()["document"]
    assert carol.get(f"/api/legal/matters/{mid}/documents").status_code == 200
    assert bob.get(f"/api/legal/matters/{mid}/documents").status_code == 404
    v = doc["current_version"]["id"]
    assert bob.get(f"/api/legal/matters/{mid}/documents/{doc['id']}/versions/{v}/file").status_code == 404
    # A document id from one matter cannot be read through another matter the caller can see.
    other = docs["matters"]["M-114"]["id"]
    assert alice.get(f"/api/legal/matters/{other}/documents/{doc['id']}").status_code == 404
    assert alice.get(f"/api/legal/matters/{other}/documents/{doc['id']}/versions/{v}/file").status_code == 404


def test_closed_matters_take_no_new_documents(docs):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    alice.patch(f"/api/legal/matters/{mid}", headers=csrf_headers(alice), json={"status": "closed", "version": 1})
    assert upload(alice, mid, "late.txt", "after closing").status_code == 409


# ── Versions ─────────────────────────────────────────────────────────────────────
def test_a_new_version_replaces_the_old_one_in_search(docs):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "msa-v1.txt", "Draft one sets the HERON-1 cap.").json()["document"]
    index_all()
    r = upload(alice, mid, "msa-v2.txt", "Draft two sets the IBIS-2 cap.", version_of=doc["id"])
    assert r.status_code == 202 and r.json()["document"]["current_version"]["number"] == 2
    index_all()
    assert search_in(alice, mid, "HERON-1", modes=["sparse", "keyword"]).json()["passages"] == []
    assert search_in(alice, mid, "IBIS-2").json()["passages"]
    versions = alice.get(f"/api/legal/matters/{mid}/documents/{doc['id']}/versions").json()["versions"]
    assert [v["number"] for v in versions] == [2, 1]
    old = alice.get(f"/api/legal/matters/{mid}/documents/{doc['id']}/versions/{versions[1]['id']}/file")
    assert old.status_code == 200 and b"HERON-1" in old.content                   # history is kept
    same = upload(alice, mid, "msa-v2.txt", "Draft two sets the IBIS-2 cap.", version_of=doc["id"])
    assert same.status_code == 409                                                  # identical to current


def test_metadata_edits_need_a_contributor(docs):
    alice, carol = as_user(docs, "alice"), as_user(docs, "carol")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "a.txt", "text").json()["document"]
    url = f"/api/legal/matters/{mid}/documents/{doc['id']}"
    v = doc["version"]
    assert carol.patch(url, headers=csrf_headers(carol), json={"title": "x", "version": v}).status_code == 403
    r = alice.patch(url, headers=csrf_headers(alice), json={"title": "Engagement letter", "kind": "correspondence", "version": v})
    assert r.status_code == 200 and r.json()["document"]["title"] == "Engagement letter"
    assert alice.patch(url, headers=csrf_headers(alice), json={"title": "y", "version": v}).status_code == 409   # stale


# ── D-012: original text indexed; personal data recorded as findings ─────────────
def test_personal_data_is_searchable_inside_the_matter_and_listed_as_findings(docs):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "notice.txt", "Send notices to legal@harrowgate.example or call 020-555-0182 quoting 4410098812.").json()["document"]
    index_all()
    hits = search_in(alice, mid, "legal@harrowgate.example", modes=["keyword"]).json()["passages"]
    assert hits and "legal@harrowgate.example" in hits[0]["text"]               # not "[REDACTED_EMAIL]"
    d = alice.get(f"/api/legal/matters/{mid}/documents/{doc['id']}").json()["document"]
    assert d["findings"] == {"email": 1, "phone": 1, "long_number": 1}


# ── Audit and isolation from the legacy stores ───────────────────────────────────
def test_document_actions_are_audited_without_content(docs, fresh_db):
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "harrowgate-msa.txt", MSA).json()["document"]
    index_all()
    alice.get(f"/api/legal/matters/{mid}/documents/{doc['id']}/versions/{doc['current_version']['id']}/file")
    search_in(alice, mid, "PELICAN-4471 liability cap")
    with sqlite3.connect(fresh_db) as c:
        rows = c.execute("SELECT action, outcome, matter_id, details_json FROM audit_events ORDER BY seq").fetchall()
    actions = [(r[0], r[1]) for r in rows if r[2] == mid]
    for expected in [("document.import", "allow"), ("document.index", "allow"), ("document.file", "allow"), ("documents.search", "allow")]:
        assert expected in actions, expected
    blob = json.dumps(rows)
    assert "PELICAN" not in blob and "harrowgate-msa" not in blob
    q = json.loads([r for r in rows if r[0] == "documents.search"][-1][3])["query"]
    assert q["chars"] == len("PELICAN-4471 liability cap") and len(q["sha256"]) == 64


def test_law_documents_never_reach_the_legacy_index(docs, fresh_db):
    alice = as_user(docs, "alice")
    upload(alice, docs["matters"]["M-101"]["id"], "msa.txt", MSA)
    index_all()
    with sqlite3.connect(fresh_db) as c:
        legacy = c.execute("SELECT COUNT(*) FROM chunks WHERE text LIKE '%PELICAN%'").fetchone()[0]
        legacy_docs = c.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    assert legacy == 0 and legacy_docs == 0
    for name in os.listdir(fresh_db.parent):
        p = fresh_db.parent / name
        if p.is_file() and name != "echomind.sqlite" and not name.startswith("echomind.sqlite"):
            assert b"PELICAN" not in p.read_bytes(), name


def test_t02_a_removed_member_cannot_download_what_they_could_a_moment_ago(docs):
    """T02: removing a membership ends later downloads; the earlier one was never cacheable."""
    alice, carol = as_user(docs, "alice"), as_user(docs, "carol")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "engagement.txt", "Engagement letter (fictional).").json()["document"]
    url = f"/api/legal/matters/{mid}/documents/{doc['id']}/versions/{doc['current_version']['id']}/file"
    first = carol.get(url)
    assert first.status_code == 200 and first.headers["cache-control"] == "no-store"
    members = alice.get(f"/api/legal/matters/{mid}/memberships").json()["memberships"]
    mine = next(m for m in members if m["username"] == "carol")
    assert alice.delete(f"/api/legal/matters/{mid}/memberships/{mine['id']}", headers=csrf_headers(alice)).status_code == 200
    assert carol.get(url).status_code == 404
    assert carol.get(f"/api/legal/matters/{mid}/documents").status_code == 404
