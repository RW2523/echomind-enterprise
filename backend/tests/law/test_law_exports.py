"""S1.10 / SEC-06 — the export gateway: redaction that really removes text (spec test T09; P-006).

For each format the redacted strings are absent from the output's text, metadata, comments and
revisions; hidden and deleted text never leaves; the watermark is on every page; unauthorized
exports are refused; and the audit chain records every step.
"""
from __future__ import annotations

import io
import json
import sqlite3
import zipfile

import pytest

from conftest import as_user, csrf_headers, index_all, upload

EMAIL, ACCOUNT, SECRET_NOTE = "legal@harrowgate.example", "4410098812", "INTERNAL-ONLY-NOTE"
BODY = f"Settlement terms for M-101 (related file M-114). Pay into account {ACCOUNT}. Notices to {EMAIL}. The cap is USD 250,000."


# ── Fixture documents ────────────────────────────────────────────────────────────
def make_pdf() -> bytes:
    import fitz
    doc = fitz.open()
    for n in range(2):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 540, 300), BODY if n == 0 else f"Page two repeats {EMAIL} and {ACCOUNT}.", fontsize=10)
    doc.set_metadata({"author": "Alice Hart", "title": f"Settlement {ACCOUNT}", "subject": EMAIL, "keywords": SECRET_NOTE})
    doc.set_xml_metadata(f"<x:xmpmeta xmlns:x='adobe:ns:meta/'><note>{SECRET_NOTE}</note></x:xmpmeta>")
    doc[0].add_text_annot((300, 300), f"Reviewer: {SECRET_NOTE}")
    return doc.tobytes()


def make_docx() -> bytes:
    """A Word file with tracked changes, a comment, hidden text, custom XML and author metadata."""
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Settlement terms</w:t></w:r></w:p>
<w:p><w:r><w:t xml:space="preserve">Pay into account </w:t></w:r><w:r><w:rPr><w:b/></w:rPr><w:t>{ACCOUNT}</w:t></w:r><w:r><w:t xml:space="preserve">. Notices to {EMAIL}.</w:t></w:r></w:p>
<w:p><w:ins w:id="1" w:author="Bob"><w:r><w:t>Inserted clause stays.</w:t></w:r></w:ins><w:del w:id="2" w:author="Bob"><w:r><w:delText>DELETED-OLD-CAP 900,000</w:delText></w:r></w:del></w:p>
<w:p><w:commentRangeStart w:id="0"/><w:r><w:t>Commented text.</w:t></w:r><w:commentRangeEnd w:id="0"/><w:r><w:commentReference w:id="0"/></w:r></w:p>
<w:p><w:r><w:rPr><w:vanish/></w:rPr><w:t>HIDDEN-RUN-TEXT</w:t></w:r><w:r><w:t>Visible end.</w:t></w:r></w:p>
</w:body></w:document>"""
    comments = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="{W}"><w:comment w:id="0" w:author="Carol"><w:p><w:r><w:t>{SECRET_NOTE} do not send</w:t></w:r></w:p></w:comment></w:comments>"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>Settlement {ACCOUNT}</dc:title><dc:creator>Alice Hart</dc:creator><cp:keywords>{SECRET_NOTE}</cp:keywords></cp:coreProperties>"""
    types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/comments.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>"""
    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/customXml" Target="../customXml/item1.xml"/></Relationships>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/comments.xml", comments)
        z.writestr("docProps/core.xml", core)
        z.writestr("customXml/item1.xml", f"<data><secret>{SECRET_NOTE}</secret></data>")
    return buf.getvalue()


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture
def ex(docs):
    """M-101 holds a PDF, a Word file and a text note; alice may export (can_export)."""
    from app.legal.matters import repo
    mid = docs["matters"]["M-101"]["id"]
    alice = as_user(docs, "alice")
    lead = [m for m in repo.list_live_memberships(mid) if m["username"] == "alice"][0]
    repo.change_membership(mid, lead["id"], by=docs["users"]["alice"]["id"], can_export=True)
    out = {}
    for name, data, mime in (("terms.pdf", make_pdf(), "application/pdf"), ("terms.docx", make_docx(), DOCX_MIME), ("note.txt", BODY, "text/plain")):
        r = upload(alice, mid, name, data, mime=mime)
        assert r.status_code == 202, r.text
        out[name] = r.json()["document"]["id"]
    index_all()
    # a second lawyer on the matter to approve
    repo.add_membership(mid, docs["users"]["dave"]["id"], "attorney", granted_by=docs["users"]["alice"]["id"])
    docs.update(mid=mid, docids=out, base=f"/api/legal/matters/{mid}/exports")
    return docs


def start(c, w, name, fmt, purpose="Disclosure to opposing counsel"):
    return c.post(w["base"], headers=csrf_headers(c), json={"object_type": "document", "object_id": w["docids"][name],
                                                            "format": fmt, "purpose": purpose})


def approve(c, w, eid, **body):
    return c.post(f"{w['base']}/{eid}/approve", headers=csrf_headers(c), json=body)


def run_export(w, name, fmt) -> tuple[dict, bytes]:
    alice, dave = as_user(w, "alice"), as_user(w, "dave")
    e = start(alice, w, name, fmt).json()["export"]
    assert alice.post(f"{w['base']}/{e['id']}/submit", headers=csrf_headers(alice)).status_code == 200
    r = approve(dave, w, e["id"])
    assert r.status_code == 200, r.text
    done = r.json()["export"]
    assert done["status"] == "ready", done
    f = alice.get(f"{w['base']}/{e['id']}/file")
    assert f.status_code == 200 and f.headers["cache-control"] == "no-store"
    return done, f.content


# ── The plan ─────────────────────────────────────────────────────────────────────
def test_the_plan_finds_personal_data_numbers_and_matter_references(ex):
    e = start(as_user(ex, "alice"), ex, "note.txt", "pdf").json()["export"]
    items = e["plan"]["items"]
    kinds = {i["kind"]: i for i in items}
    assert kinds["email"]["text"] == EMAIL and kinds["email"]["redact"] is True
    assert kinds["long_number"]["text"] == ACCOUNT and kinds["long_number"]["redact"] is True
    numbers = {i["text"]: i["redact"] for i in items if i["kind"] == "matter_number"}
    assert numbers == {"M-101": False, "M-114": True}        # this matter's own number is in the watermark anyway
    assert e["status"] == "draft" and e["watermark"] is None


def test_the_requester_can_adjust_the_plan_before_submitting(ex):
    alice = as_user(ex, "alice")
    e = start(alice, ex, "note.txt", "pdf").json()["export"]
    mn = next(i for i in e["plan"]["items"] if i["kind"] == "matter_number" and i["text"] == "M-114")
    r = alice.put(f"{ex['base']}/{e['id']}/plan", headers=csrf_headers(alice),
                  json={"items": [{"id": mn["id"], "redact": False}], "add": ["USD 250,000"]})
    items = r.json()["export"]["plan"]["items"]
    assert next(i for i in items if i["text"] == "M-114")["redact"] is False
    assert any(i["kind"] == "manual" and i["text"] == "USD 250,000" and i["redact"] for i in items)
    alice.post(f"{ex['base']}/{e['id']}/submit", headers=csrf_headers(alice))
    assert alice.put(f"{ex['base']}/{e['id']}/plan", headers=csrf_headers(alice), json={"items": [], "add": ["x"]}).status_code == 409


# ── T09 per format ───────────────────────────────────────────────────────────────
def test_t09_pdf_redaction_removes_the_text_and_scrubs_the_file(ex):
    import fitz
    e, data = run_export(ex, "terms.pdf", "pdf")
    doc = fitz.open(stream=data, filetype="pdf")
    text = "".join(p.get_text() for p in doc)
    for s in (EMAIL, ACCOUNT, "M-114"):
        assert s not in text, s
    meta = json.dumps(doc.metadata) + (doc.get_xml_metadata() or "")
    for s in (EMAIL, ACCOUNT, SECRET_NOTE, "Alice Hart"):
        assert s not in meta, s
    assert all(not list(p.annots()) for p in doc)
    assert all(e["watermark"] in p.get_text() for p in doc)                 # every page
    assert e["watermark"].startswith("CONFIDENTIAL — M-101 — alice — ")
    assert "USD 250,000" in text                                            # only what was planned goes
    assert e["verification"]["ok"] is True and all(e["verification"]["checks"].values())
    for s in (EMAIL.encode(), ACCOUNT.encode(), SECRET_NOTE.encode()):
        assert s not in fitz.open(stream=data, filetype="pdf").tobytes(garbage=4, expand=255)


def test_t09_docx_is_rebuilt_without_comments_revisions_hidden_text_or_metadata(ex):
    e, data = run_export(ex, "terms.docx", "docx")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        everything = "".join(z.read(n).decode("utf-8", "replace") for n in names)
        body = z.read("word/document.xml").decode()
    assert not any(n.startswith("customXml") or n == "word/comments.xml" for n in names)
    for s in (EMAIL, ACCOUNT, SECRET_NOTE, "Alice Hart", "DELETED-OLD-CAP", "HIDDEN-RUN-TEXT"):
        assert s not in everything, s
    for tag in ("w:ins", "w:del", "w:vanish", "w:commentReference"):
        assert f"<{tag}" not in body, tag
    assert "Inserted clause stays." in body and "Visible end." in body and "Settlement terms" in body
    assert e["watermark"] in everything
    from docx import Document
    d = Document(io.BytesIO(data))
    assert d.paragraphs[0].style.name.startswith("Heading")


def test_t09_text_sources_render_to_both_formats(ex):
    import fitz
    _, pdf = run_export(ex, "note.txt", "pdf")
    assert EMAIL not in "".join(p.get_text() for p in fitz.open(stream=pdf, filetype="pdf"))
    _, docx = run_export(ex, "note.txt", "docx")
    with zipfile.ZipFile(io.BytesIO(docx)) as z:
        assert EMAIL not in "".join(z.read(n).decode("utf-8", "replace") for n in z.namelist())


def test_a_failed_verification_releases_nothing(ex, monkeypatch):
    from app.legal.exports import render
    real = render.render
    monkeypatch.setattr(render, "render", lambda job: real({**job, "items": []}))   # a renderer that forgets to redact
    alice, dave = as_user(ex, "alice"), as_user(ex, "dave")
    e = start(alice, ex, "note.txt", "pdf").json()["export"]
    alice.post(f"{ex['base']}/{e['id']}/submit", headers=csrf_headers(alice))
    r = approve(dave, ex, e["id"])
    assert r.status_code == 200 and r.json()["export"]["status"] == "failed"
    assert r.json()["export"]["verification"]["checks"]["redactions_absent"] is False
    assert alice.get(f"{ex['base']}/{e['id']}/file").status_code == 409


# ── Who may export ───────────────────────────────────────────────────────────────
def test_exports_need_the_export_right_and_a_second_lawyer(ex):
    from app.legal.matters import repo
    alice, carol, bob, dave = (as_user(ex, n) for n in ("alice", "carol", "bob", "dave"))
    assert start(carol, ex, "note.txt", "pdf").status_code == 403                 # read only
    assert start(dave, ex, "note.txt", "pdf").status_code == 403                  # attorney without can_export
    assert start(bob, ex, "note.txt", "pdf").status_code == 404                   # not a member
    e = start(alice, ex, "note.txt", "pdf").json()["export"]
    assert approve(dave, ex, e["id"]).status_code == 409                          # not submitted yet
    alice.post(f"{ex['base']}/{e['id']}/submit", headers=csrf_headers(alice))
    assert approve(alice, ex, e["id"]).status_code == 403                         # not by the requester
    assert approve(carol, ex, e["id"]).status_code == 403                         # read-only cannot approve
    assert alice.get(f"{ex['base']}/{e['id']}/file").status_code == 409           # not ready yet
    assert dave.get(f"{ex['base']}/{e['id']}/file").status_code == 404            # not (yet) the approver
    approve(dave, ex, e["id"])
    assert carol.get(f"{ex['base']}/{e['id']}/file").status_code == 404           # only requester and approver
    assert dave.get(f"{ex['base']}/{e['id']}/file").status_code == 200


def test_privileged_documents_need_an_explicit_acknowledgement(ex):
    alice, dave = as_user(ex, "alice"), as_user(ex, "dave")
    url = f"/api/legal/matters/{ex['mid']}/labels/document/{ex['docids']['note.txt']}"
    alice.post(url, headers=csrf_headers(alice), json={"label": "privileged"})
    e = start(alice, ex, "note.txt", "pdf").json()["export"]
    assert e["label"] == "privileged"
    alice.post(f"{ex['base']}/{e['id']}/submit", headers=csrf_headers(alice))
    r = approve(dave, ex, e["id"])
    assert r.status_code == 409 and "privileged" in r.json()["detail"]
    assert approve(dave, ex, e["id"], acknowledge_privilege=True).json()["export"]["status"] == "ready"


def test_exports_expire(ex, monkeypatch):
    from app.legal.exports import service
    e, _ = run_export(ex, "note.txt", "pdf")
    monkeypatch.setattr(service, "_now_iso", lambda: "2999-01-01T00:00:00.000Z")
    alice = as_user(ex, "alice")
    assert alice.get(f"{ex['base']}/{e['id']}/file").status_code == 410
    assert alice.get(f"{ex['base']}/{e['id']}").json()["export"]["status"] == "expired"


def test_the_audit_chain_records_every_step_without_content(ex, fresh_db):
    from app.legal.audit import chain
    run_export(ex, "terms.pdf", "pdf")
    with sqlite3.connect(fresh_db) as c:
        rows = c.execute("SELECT action, outcome, details_json FROM audit_events WHERE action LIKE 'export.%' ORDER BY seq").fetchall()
    assert [r[0] for r in rows] == ["export.request", "export.submit", "export.approve", "export.download"]
    assert all(r[1] == "allow" for r in rows)
    blob = json.dumps(rows)
    assert EMAIL not in blob and ACCOUNT not in blob and "Disclosure" not in blob
    assert chain.verify()["ok"] is True


def test_purging_a_document_also_purges_its_exports(ex, fresh_db):
    import os
    from conftest import make_user
    e, _ = run_export(ex, "note.txt", "pdf")
    make_user("rhea", mfa=True, firm_roles=["records_admin"])
    ex["users"]["rhea"] = {"username": "rhea"}
    alice = as_user(ex, "alice")
    d = alice.post(f"/api/legal/matters/{ex['mid']}/documents/{ex['docids']['note.txt']}/delete", headers=csrf_headers(alice),
                   json={"reason": "x"}).json()["deletion"]
    assert d["report"]["exports"] == 1
    from app.legal.records import deletion
    out = deletion.purge(d["id"], approved_by=ex["users"]["dave"]["id"])
    assert out["verification"]["ok"] and out["verification"]["checks"]["exports"]
    with sqlite3.connect(fresh_db) as c:
        assert c.execute("SELECT COUNT(*) FROM legal_exports WHERE id=?", (e["id"],)).fetchone()[0] == 0
    assert not os.path.exists(os.path.join(fresh_db.parent, "legal", "exports", ex["mid"], e["id"] + ".bin"))


def test_the_preview_shows_pages_with_the_redactions_and_is_for_the_matter_team_only(ex):
    alice, bob = as_user(ex, "alice"), as_user(ex, "bob")
    e = start(alice, ex, "terms.pdf", "pdf").json()["export"]
    info = alice.get(f"{ex['base']}/{e['id']}/preview").json()
    assert info["pages"] == 2
    png = alice.get(f"{ex['base']}/{e['id']}/preview/1")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n" and png.headers["cache-control"] == "no-store"
    assert bob.get(f"{ex['base']}/{e['id']}/preview/1").status_code == 404
    assert as_user(ex, "carol").get(f"{ex['base']}/{e['id']}/preview").status_code == 403
