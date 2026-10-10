"""S1.9 / SEC-07 and SEC-06 labels — legal holds, approved deletion with verification, retention
dry runs and privilege labels (spec test T04; policies P-002, P-003, P-004 safe defaults).

Held data survives every path: a person's deletion request, an approval racing a new hold, a
direct service call, the retention runner, and the (closed) legacy routes.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from conftest import as_user, csrf_headers, index_all, make_user, search_in, upload

TOKEN = "HERON-4242"


@pytest.fixture
def records(docs):
    """The T01 world plus two records administrators and one indexed document in M-101 that a
    research answer has quoted."""
    from app.legal.research import llm
    from conftest import HashEmbedder  # noqa: F401  (docs already installs it)
    for name in ("rhea", "rex"):
        docs["users"][name] = make_user(name, mfa=True, firm_roles=["records_admin"])
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    doc = upload(alice, mid, "settlement-memo.txt", f"Settlement memo. The {TOKEN} figure is privileged and confidential.").json()["document"]
    index_all()

    class Quoting:
        async def chat_stream(self, messages, temperature, max_tokens):
            yield "The memo records the figure [1]."
    llm._override = Quoting()
    alice.post(f"/api/legal/matters/{mid}/research/query", headers=csrf_headers(alice), json={"question": f"What is the {TOKEN} figure?"})
    llm._override = None
    docs.update(mid=mid, doc=doc, dbase=f"/api/legal/matters/{mid}/documents/{doc['id']}")
    return docs


def q(db, sql, *a):
    with sqlite3.connect(db) as c:
        return c.execute(sql, a).fetchall()


def request_deletion(c, w, reason="Duplicate of the signed copy"):
    return c.post(f"{w['dbase']}/delete", headers=csrf_headers(c), json={"reason": reason})


def place_hold(w, matter_ids, by="rhea", approver="rex"):
    req, appr = as_user(w, by), as_user(w, approver)
    h = req.post("/api/legal/holds", headers=csrf_headers(req), json={"name": "Beta litigation hold", "reason": "Anticipated claim",
                                                                       "matter_ids": matter_ids}).json()["hold"]
    r = appr.post(f"/api/legal/holds/{h['id']}/approve", headers=csrf_headers(appr))
    assert r.status_code == 200, r.text
    return h


# ── Deletion: request, preview, approval, purge, verification ────────────────────
def test_the_preview_lists_everything_a_purge_would_remove(records):
    alice = as_user(records, "alice")
    p = alice.get(f"{records['dbase']}/deletion-preview").json()["report"]
    assert p["versions"] == 1 and p["passages"] >= 1 and p["files"] == 1 and p["index_vectors"] == p["passages"]
    assert p["research_citations"] == 1 and p["findings"] >= 0
    assert "audit trail" in " ".join(p["kept"])


def test_a_request_hides_the_document_at_once_and_it_can_be_restored(records):
    alice = as_user(records, "alice")
    mid = records["mid"]
    d = request_deletion(alice, records).json()["deletion"]
    assert d["status"] == "requested"
    assert all(x["id"] != records["doc"]["id"] for x in alice.get(f"/api/legal/matters/{mid}/documents").json()["documents"])
    assert alice.get(records["dbase"]).status_code == 404
    assert search_in(alice, mid, TOKEN).json()["passages"] == []
    pending = alice.get(f"/api/legal/matters/{mid}/deletions").json()["deletions"]
    assert pending[0]["id"] == d["id"] and pending[0]["object_label"] == "settlement-memo.txt"
    assert alice.post(f"/api/legal/deletions/{d['id']}/restore", headers=csrf_headers(alice)).status_code == 200
    assert alice.get(records["dbase"]).status_code == 200
    assert TOKEN in search_in(alice, mid, TOKEN).json()["passages"][0]["text"]


def test_only_a_second_person_with_records_authority_can_approve(records):
    alice = as_user(records, "alice")
    d = request_deletion(alice, records).json()["deletion"]
    for name in ("alice", "bob", "carol", "admin"):
        c = as_user(records, name)
        assert c.post(f"/api/legal/deletions/{d['id']}/approve", headers=csrf_headers(c)).status_code in (403, 404), name
    assert as_user(records, "carol").post(f"{records['dbase']}/delete", headers=csrf_headers(as_user(records, "carol")),
                                          json={"reason": "x"}).status_code in (403, 404)


def test_an_approved_deletion_purges_everything_and_proves_it(records, fresh_db):
    from app.legal.audit import chain
    alice, rhea = as_user(records, "alice"), as_user(records, "rhea")
    mid, did = records["mid"], records["doc"]["id"]
    vid = records["doc"]["current_version"]["id"]
    d = request_deletion(alice, records).json()["deletion"]
    r = rhea.post(f"/api/legal/deletions/{d['id']}/approve", headers=csrf_headers(rhea))
    assert r.status_code == 200, r.text
    out = r.json()["deletion"]
    assert out["status"] == "purged" and out["verification"]["ok"] is True
    assert set(out["verification"]["checks"]) >= {"database", "index", "files", "research_quotes"}
    for table, col in (("legal_documents", "id"), ("legal_document_versions", "document_id"), ("legal_chunks", "document_id")):
        assert q(fresh_db, f"SELECT COUNT(*) FROM {table} WHERE {col}=?", did)[0][0] == 0, table
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_findings WHERE version_id=?", vid)[0][0] == 0
    assert not os.path.exists(os.path.join(fresh_db.parent, "legal", "blobs", mid, vid + ".bin"))
    part = os.path.join(fresh_db.parent, "legal", "partitions", "m_" + mid, "meta.json")
    assert TOKEN.lower() not in open(part).read().lower()
    turns = q(fresh_db, "SELECT citations_json FROM legal_research_turns WHERE matter_id=?", mid)
    assert turns and TOKEN not in json.dumps(turns)
    assert q(fresh_db, "SELECT object_label FROM legal_deletions WHERE id=?", d["id"]) == [(None,)]   # no title kept after purge
    assert chain.verify()["ok"] is True                                    # the audit trail is never purged
    acts = {r[0] for r in q(fresh_db, "SELECT action FROM audit_events")}
    assert {"deletion.request", "deletion.approve"} <= acts


def test_a_rejected_request_restores_the_document(records):
    alice, rex = as_user(records, "alice"), as_user(records, "rex")
    d = request_deletion(alice, records).json()["deletion"]
    r = rex.post(f"/api/legal/deletions/{d['id']}/reject", headers=csrf_headers(rex), json={"note": "Needed for the claim"})
    assert r.json()["deletion"]["status"] == "rejected"
    assert alice.get(records["dbase"]).status_code == 200


# ── T04: legal holds win over every deletion path ────────────────────────────────
def test_holds_need_two_people_and_members_see_the_flag(records):
    rhea, rex, mp = as_user(records, "rhea"), as_user(records, "rex"), as_user(records, "mp")
    h = rhea.post("/api/legal/holds", headers=csrf_headers(rhea), json={"name": "Hold", "reason": "r", "matter_ids": [records["mid"]]}).json()["hold"]
    assert h["status"] == "requested"
    assert rhea.post(f"/api/legal/holds/{h['id']}/approve", headers=csrf_headers(rhea)).status_code == 403     # not the requester
    assert as_user(records, "alice").post(f"/api/legal/holds/{h['id']}/approve", headers=csrf_headers(as_user(records, "alice"))).status_code == 403
    assert mp.post(f"/api/legal/holds/{h['id']}/approve", headers=csrf_headers(mp)).status_code == 200
    m = as_user(records, "alice").get(f"/api/legal/matters/{records['mid']}").json()["matter"]
    assert m["on_hold"] is True
    rhea.post(f"/api/legal/holds/{h['id']}/release", headers=csrf_headers(rhea))
    assert rhea.post(f"/api/legal/holds/{h['id']}/release/approve", headers=csrf_headers(rhea)).status_code == 403
    assert rex.post(f"/api/legal/holds/{h['id']}/release/approve", headers=csrf_headers(rex)).status_code == 200
    assert as_user(records, "alice").get(f"/api/legal/matters/{records['mid']}").json()["matter"]["on_hold"] is False


def test_t04_a_person_cannot_request_deletion_of_held_material(records):
    place_hold(records, [records["mid"]])
    r = request_deletion(as_user(records, "alice"), records)
    assert r.status_code == 409 and "legal hold" in r.json()["detail"]


def test_t04_a_hold_placed_after_the_request_blocks_the_purge(records, fresh_db):
    alice, rhea = as_user(records, "alice"), as_user(records, "rhea")
    d = request_deletion(alice, records).json()["deletion"]
    place_hold(records, [records["mid"]], by="rhea", approver="mp")
    r = rhea.post(f"/api/legal/deletions/{d['id']}/approve", headers=csrf_headers(rhea))
    assert r.status_code == 409 and r.json()["detail"].startswith("This material is under legal hold")
    assert q(fresh_db, "SELECT status FROM legal_deletions WHERE id=?", d["id"]) == [("blocked",)]
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_document_versions WHERE document_id=?", records["doc"]["id"])[0][0] == 1
    assert alice.post(f"/api/legal/deletions/{d['id']}/restore", headers=csrf_headers(alice)).status_code == 200


def test_t04_a_direct_service_call_refuses_held_material(records):
    from app.legal.records import deletion, holds
    d = request_deletion(as_user(records, "alice"), records).json()["deletion"]
    place_hold(records, [records["mid"]], by="rhea", approver="rex")
    with pytest.raises(holds.Held):
        deletion.purge(d["id"], approved_by=records["users"]["rex"]["id"])


def test_t04_the_retention_runner_never_purges_and_skips_held_matters(records, monkeypatch, fresh_db):
    from app.core.config import settings
    alice, rhea, mp = as_user(records, "alice"), as_user(records, "rhea"), as_user(records, "mp")
    m114 = records["matters"]["M-114"]["id"]
    upload(alice, m114, "old.txt", "Old diligence note.")
    index_all()
    for mid, num in ((records["mid"], 1), (m114, 1)):
        alice.patch(f"/api/legal/matters/{mid}", headers=csrf_headers(alice), json={"status": "closed", "version": num})
    pol = rhea.post("/api/legal/retention/policies", headers=csrf_headers(rhea),
                    json={"object_type": "document", "days_after_close": 0, "note": "test"}).json()["policy"]
    dry = rhea.post("/api/legal/retention/run", headers=csrf_headers(rhea), json={"mode": "dry_run"}).json()["run"]
    assert dry["mode"] == "dry_run" and dry["report"]["candidates"] == 0          # a draft policy selects nothing
    assert rhea.post("/api/legal/retention/run", headers=csrf_headers(rhea), json={"mode": "apply"}).status_code == 409
    assert rhea.post(f"/api/legal/retention/policies/{pol['id']}/approve", headers=csrf_headers(rhea)).status_code == 403
    assert mp.post(f"/api/legal/retention/policies/{pol['id']}/approve", headers=csrf_headers(mp)).status_code == 200
    place_hold(records, [records["mid"]])
    dry = rhea.post("/api/legal/retention/run", headers=csrf_headers(rhea), json={"mode": "dry_run"}).json()["run"]
    assert dry["report"]["candidates"] == 1 and dry["report"]["held_skipped"] == 1
    assert rhea.post("/api/legal/retention/run", headers=csrf_headers(rhea), json={"mode": "apply"}).status_code == 409   # switch off
    monkeypatch.setattr(settings, "LEGAL_RETENTION_APPLY", True)
    applied = rhea.post("/api/legal/retention/run", headers=csrf_headers(rhea), json={"mode": "apply"}).json()["run"]
    assert applied["report"]["requests_created"] == 1
    rows = q(fresh_db, "SELECT matter_id, status, origin FROM legal_deletions")
    assert rows == [(m114, "requested", "retention")]                               # a request for a person, never a purge
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_documents WHERE matter_id=?", records["mid"])[0][0] == 1
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_documents WHERE matter_id=? AND deleted_at IS NULL", m114)[0][0] == 1


def test_t04_the_legacy_delete_routes_are_closed(records):
    alice = as_user(records, "alice")
    for method, path in (("DELETE", "/api/docs/x"), ("POST", "/api/docs/delete-all"), ("DELETE", "/api/chat/x"),
                         ("DELETE", "/api/transcribe/transcripts/x"), ("DELETE", "/api/boardroom/sessions/x"), ("DELETE", "/api/docgen/jobs/x")):
        assert alice.request(method, path, headers=csrf_headers(alice)).status_code == 404, path


def test_meeting_transcripts_go_through_the_same_service(records, fresh_db, monkeypatch):
    from app.legal.meetings import capture

    class T:
        async def feed(self, pcm, sr):
            return [("Sentence one. ", 100)]

        async def flush(self):
            return []
    monkeypatch.setattr(capture, "transcriber_factory", lambda: T())
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_SEGMENT_MIN_WORDS", 1)      # one stored segment per test sentence
    alice, rhea = as_user(records, "alice"), as_user(records, "rhea")
    mid = records["mid"]
    m = alice.post(f"/api/legal/matters/{mid}/meetings", headers=csrf_headers(alice), json={"title": "Call"}).json()["meeting"]
    base = f"/api/legal/matters/{mid}/meetings/{m['id']}"
    p = alice.post(f"{base}/participants", headers=csrf_headers(alice), json={"name": "A", "role": "host"}).json()["participant"]
    alice.post(f"{base}/participants/{p['id']}/consent", headers=csrf_headers(alice), json={"method": "written", "statement_read": True})
    with alice.websocket_connect(f"wss://testserver{base}/capture", headers={"origin": "https://testserver"}) as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00\x00" * 800)
        ws.receive_json()
        ws.send_json({"type": "stop"})
        ws.receive_json()
    d = alice.post(f"{base}/transcript/delete", headers=csrf_headers(alice), json={"reason": "Minutes approved"}).json()["deletion"]
    assert alice.get(f"{base}/transcript").json()["segments"] == []
    rhea.post(f"/api/legal/deletions/{d['id']}/approve", headers=csrf_headers(rhea))
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_transcript_segments WHERE meeting_id=?", m["id"])[0][0] == 0
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_consent_events WHERE meeting_id=?", m["id"])[0][0] == 1   # consent record kept


def test_records_staff_see_requests_firm_wide_with_matter_numbers(records):
    request_deletion(as_user(records, "alice"), records)
    rows = as_user(records, "rhea").get("/api/legal/deletions").json()["deletions"]
    assert rows[0]["matter"]["number"] == "M-101" and rows[0]["report"]["versions"] == 1
    assert as_user(records, "alice").get("/api/legal/deletions").status_code == 403
    d = as_user(records, "rhea").get("/api/legal/matter-directory?q=M-2").json()["matters"]
    assert {m["number"] for m in d} == {"M-202", "M-215"}


# ── Labels (P-004 safe defaults: advisory) ───────────────────────────────────────
def test_labels_are_set_by_reviewers_and_suggested_by_paralegals(records, fresh_db):
    from app.legal.matters import repo
    mid = records["mid"]
    url = f"/api/legal/matters/{mid}/labels/document/{records['doc']['id']}"
    alice, carol = as_user(records, "alice"), as_user(records, "carol")
    first = alice.get(url).json()
    assert first["current"] is None and first["suggestions"][0]["label"] == "privileged"           # the detector
    assert first["suggestions"][0]["source"] == "detector"
    assert carol.post(url, headers=csrf_headers(carol), json={"label": "privileged"}).status_code == 403   # read only
    mm = [m for m in repo.list_live_memberships(mid) if m["username"] == "carol"][0]
    repo.change_membership(mid, mm["id"], by=records["users"]["alice"]["id"], role="paralegal")
    r = carol.post(url, headers=csrf_headers(carol), json={"label": "work_product"}).json()
    assert r["current"] is None and r["suggestions"][0]["label"] == "work_product"
    r = alice.post(url, headers=csrf_headers(alice), json={"label": "privileged"}).json()
    assert r["current"]["label"] == "privileged" and r["current"]["by"] == "alice" and r["suggestions"] == []
    doc = alice.get(records["dbase"]).json()["document"]
    assert doc["label"] == "privileged"
    assert ("label.set", "allow") in [(a, o) for a, o in q(fresh_db, "SELECT action, outcome FROM audit_events")]
    assert alice.post(url, headers=csrf_headers(alice), json={"label": "secret"}).status_code == 422
    assert as_user(records, "bob").get(url).status_code == 404
