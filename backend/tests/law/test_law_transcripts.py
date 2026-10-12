"""S2.4 — legal transcription (LIVE-01; D-017, D-040).

The speech model's text is the record: stored as produced, in whole-sentence segments, never
changed. Who was speaking is marked live from the meeting's roster; every segment has wall-clock
times, so pauses show. If the model falls behind or fails, the capture says so and the transcript
shows a gap: nothing is lost silently. Corrections are revisions (who, when, why), some offered
from the matter's own vocabulary with their provenance. A finalised transcript becomes a matter
document that research can cite, and purging the transcript purges that copy too.
"""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from conftest import as_user, csrf_headers, index_all, upload

WSS = "wss://testserver"
ORIGIN = {"origin": "https://testserver"}
FRAME = b"\x01\x00" * 1600          # 100 ms of 16 kHz PCM16


class Scripted:
    """The speech model, scripted: each frame returns the next pieces; times restart after reset()."""
    def __init__(self, pieces=None, fail_on=(), delay=0.0):
        self.pieces = list(pieces or [])
        self.fed, self.resets, self.fail_on, self.delay = 0, 0, set(fail_on), delay
        self.clock = 0

    async def feed(self, pcm16, sample_rate):
        import asyncio
        self.fed += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fed in self.fail_on:
            raise RuntimeError("decoder error")
        self.clock += 100
        text = self.pieces.pop(0) if self.pieces else ""
        return [(text, self.clock)] if text else []

    async def flush(self):
        return []

    def reset(self):
        self.resets += 1
        self.clock = 0


@pytest.fixture
def tx(docs, monkeypatch):
    from app.core.config import settings
    from app.legal.meetings import capture
    monkeypatch.setattr(settings, "LEGAL_WS_RECHECK_SEC", 0.1)
    monkeypatch.setattr(settings, "LEGAL_CONSENT_RECHECK_SEC", 0.1)
    model = {"t": Scripted()}
    monkeypatch.setattr(capture, "transcriber_factory", lambda: model["t"])
    alice = as_user(docs, "alice")
    mid = docs["matters"]["M-101"]["id"]
    m = alice.post(f"/api/legal/matters/{mid}/meetings", headers=csrf_headers(alice),
                   json={"title": "Renewal call", "kind": "client_call"}).json()["meeting"]
    base = f"/api/legal/matters/{mid}/meetings/{m['id']}"
    people = {}
    for name, role in (("Helena Marsh", "attorney"), ("Petra Voss", "client"), ("Thomas Reyes", "counterparty")):
        people[name] = alice.post(f"{base}/participants", headers=csrf_headers(alice), json={"name": name, "role": role}).json()["participant"]
        alice.post(f"{base}/participants/{people[name]['id']}/consent", headers=csrf_headers(alice),
                   json={"method": "verbal_on_record", "statement_read": True})
    docs.update(mid=mid, meeting=m, base=base, people=people, model=model)
    return docs


def record(c, w, steps):
    """Run a capture: steps are FRAME, dicts (control messages) or ("sleep", s). Returns every message."""
    got = []
    with c.websocket_connect(f"{WSS}{w['base']}/capture", headers=ORIGIN) as ws:
        got.append(ws.receive_json())
        for step in steps:
            if step is FRAME:
                ws.send_bytes(FRAME)
            elif isinstance(step, tuple):
                time.sleep(step[1])
            else:
                ws.send_json(step)
                if step.get("type") in ("speaker", "pause", "resume"):
                    while True:
                        msg = ws.receive_json()
                        got.append(msg)
                        if msg["type"] in ("speaker", "state", "error"):
                            break
        ws.send_json({"type": "stop"})
        while True:
            msg = ws.receive_json()
            got.append(msg)
            if msg["type"] == "stopped":
                break
    return got


def view(c, w):
    r = c.get(f"{w['base']}/transcript")
    assert r.status_code == 200, r.text
    return r.json()


def q(db, sql, *args):
    with sqlite3.connect(db) as c:
        return c.execute(sql, args).fetchall()


# ── The record ───────────────────────────────────────────────────────────────────
def test_the_text_is_stored_as_the_model_produced_it_and_never_changed(tx, fresh_db):
    alice = as_user(tx, "alice")
    tx["model"]["t"] = Scripted(["Uh, the FEE is USD 19,750 — ", "not 18,500; we did not agree. ", "Clause 8.1 stays."])
    record(alice, tx, [FRAME, FRAME, FRAME])
    segs = view(alice, tx)["segments"]
    assert [s["raw_text"] for s in segs] == ["Uh, the FEE is USD 19,750 — not 18,500; we did not agree.", "Clause 8.1 stays."]
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with sqlite3.connect(fresh_db) as db:
            db.execute("UPDATE legal_transcript_segments SET text='changed'")


def test_speech_is_kept_in_whole_sentences_not_model_chunks(tx):
    alice = as_user(tx, "alice")
    chunks = ["The fee is due. ", "Within thirty days. ", "Of each invoice. ", "Interest runs after that. ", "At one percent. ",
              "Per month. ", "Until paid. "]
    tx["model"]["t"] = Scripted(chunks)
    record(alice, tx, [FRAME] * len(chunks))
    texts = [s["raw_text"] for s in view(alice, tx)["segments"]]
    assert texts == ["The fee is due. Within thirty days. Of each invoice. Interest runs after that.",
                     "At one percent. Per month. Until paid."]          # at least 12 words, ended at a sentence end; the rest at stop


def test_the_speaker_is_marked_from_the_roster_and_each_turn_starts_a_new_segment(tx, fresh_db):
    alice = as_user(tx, "alice")
    helena, petra = tx["people"]["Helena Marsh"]["id"], tx["people"]["Petra Voss"]["id"]
    model = tx["model"]["t"] = Scripted(["Can you confirm the cap? ", "Yes, it is five hundred thousand. "])
    got = record(alice, tx, [{"type": "speaker", "participant_id": helena}, FRAME,
                             {"type": "speaker", "participant_id": petra}, FRAME,
                             {"type": "speaker", "participant_id": "someone-else"}])
    assert {"type": "error", "message": "That person is not a participant of this meeting."} in got
    segs = view(alice, tx)["segments"]
    assert [(s["speaker"], s["raw_text"]) for s in segs] == [("Helena Marsh", "Can you confirm the cap?"),
                                                             ("Petra Voss", "Yes, it is five hundred thousand.")]
    assert model.resets == 2                          # the model restarts at each turn, as the legacy toggle did
    assert segs[1]["t_start_ms"] > segs[0]["t_start_ms"]   # its times continue on the capture's clock
    marks = [(m["kind"], m["participant_id"]) for m in view(alice, tx)["marks"]]
    assert marks == [("speaker", helena), ("speaker", petra)]


def test_wall_clock_times_show_a_pause(tx):
    alice = as_user(tx, "alice")
    tx["model"]["t"] = Scripted(["We start here. ", "And continue after the break. "])
    record(alice, tx, [FRAME, {"type": "pause"}, ("sleep", 1.2), {"type": "resume"}, FRAME])
    a, b = view(alice, tx)["segments"]
    from datetime import datetime
    gap = datetime.fromisoformat(b["wall_start"].replace("Z", "+00:00")) - datetime.fromisoformat(a["wall_end"].replace("Z", "+00:00"))
    assert gap.total_seconds() >= 1.0                 # the pause shows in wall-clock time
    assert b["t_start_ms"] - a["t_start_ms"] == 100   # while the audio clock has no gap


def test_a_transcription_error_is_shown_as_a_gap_never_lost_silently(tx):
    alice = as_user(tx, "alice")
    tx["model"]["t"] = Scripted(["Before the error. ", "Never seen. ", "After the error. "], fail_on={2})
    got = record(alice, tx, [FRAME, FRAME, FRAME])
    assert {"type": "degraded", "reason": "transcription_error"} in got
    data = view(alice, tx)
    assert [m["kind"] for m in data["marks"]] == ["gap"]
    from app.legal.meetings import transcript
    doc = transcript.as_document({"title": "Renewal call", "kind": "client_call"}, data)
    assert "[Not transcribed: a transcription error at this point.]" in doc


def test_falling_behind_is_reported(tx, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_CAPTURE_LAG_WARN_SEC", 0.3)
    alice = as_user(tx, "alice")
    tx["model"]["t"] = Scripted(["One. ", "Two. ", "Three. ", "Four. "], delay=0.25)   # 250 ms to read 100 ms of audio
    got = record(alice, tx, [FRAME, FRAME, FRAME, FRAME])
    behind = [m for m in got if m["type"] == "degraded"]
    assert behind and behind[0]["reason"] == "behind" and behind[0]["lag_s"] >= 0.3
    assert "degraded" in [m["kind"] for m in view(alice, tx)["marks"]]


def test_a_slow_model_never_stops_the_socket_being_read(tx):
    """T12 (D-066): under load the model fell behind, the socket went unread and the browser's keepalive
    timed out. Reading never waits for the model; audio and turns are transcribed in order as it catches up."""
    alice = as_user(tx, "alice")
    helena, petra = tx["people"]["Helena Marsh"]["id"], tx["people"]["Petra Voss"]["id"]
    model = tx["model"]["t"] = Scripted(["Can you confirm the cap? ", "", "Yes, it is five hundred thousand. ", ""], delay=0.4)
    got = []
    with alice.websocket_connect(f"{WSS}{tx['base']}/capture", headers=ORIGIN) as ws:
        got.append(ws.receive_json())
        ws.send_json({"type": "speaker", "participant_id": helena})
        ws.send_bytes(FRAME)
        ws.send_bytes(FRAME)
        ws.send_json({"type": "speaker", "participant_id": petra})
        ws.send_bytes(FRAME)
        ws.send_bytes(FRAME)
        asked = time.monotonic()
        ws.send_json({"type": "speaker", "participant_id": "someone-else"})
        while (msg := ws.receive_json())["type"] != "error":
            got.append(msg)
        answered = time.monotonic() - asked
        ws.send_json({"type": "stop"})
        while (msg := ws.receive_json())["type"] != "stopped":
            got.append(msg)
    assert answered < 0.6                             # at once, not after the model's 1.6 s of work
    assert model.fed == 4 and model.resets == 2
    segs = view(alice, tx)["segments"]
    assert [(s["speaker"], s["raw_text"]) for s in segs] == [("Helena Marsh", "Can you confirm the cap?"),
                                                             ("Petra Voss", "Yes, it is five hundred thousand.")]


def test_a_model_too_far_behind_leaves_a_marked_gap_never_a_silent_loss(tx, monkeypatch, fresh_db):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_CAPTURE_BACKLOG_SEC", 0.3)
    alice = as_user(tx, "alice")
    model = tx["model"]["t"] = Scripted(["One. ", "Two. ", "Three. ", "Four. ", "After the gap. "], delay=0.3)
    got = record(alice, tx, [FRAME] * 10 + [("sleep", 1.5), FRAME, FRAME])
    assert any(m["type"] == "degraded" and m["reason"] == "skipped" for m in got)
    assert model.fed < 12
    data = view(alice, tx)
    assert [s["raw_text"] for s in data["segments"]] == ["One. Two. Three. Four.", "After the gap."]
    gaps = [m for m in data["marks"] if m["kind"] == "gap"]
    assert [m["detail"] for m in gaps] == ["transcription_behind"]
    assert data["segments"][0]["wall_start"] <= gaps[0]["at"] <= data["segments"][1]["wall_start"]
    from app.legal.meetings import transcript
    doc = transcript.as_document({"title": "Renewal call", "kind": "client_call"}, data)
    assert doc.index("Four.") < doc.index("[Not transcribed: speech recognition fell too far behind here.]") < doc.index("After the gap.")
    cid = data["segments"][0]["capture_id"] if "capture_id" in data["segments"][0] else gaps[0]["capture_id"]
    assert q(fresh_db, "SELECT SUM(bytes) FROM legal_capture_custody WHERE capture_id=?", cid) == [(12 * len(FRAME),)]   # every frame received


# ── Corrections ──────────────────────────────────────────────────────────────────
def revise(c, w, sid, **body):
    return c.post(f"{w['base']}/transcript/segments/{sid}/revisions", headers=csrf_headers(c), json=body)


def test_corrections_are_revisions_with_who_when_and_why(tx, fresh_db):
    alice, carol, bob = as_user(tx, "alice"), as_user(tx, "carol"), as_user(tx, "bob")
    tx["model"]["t"] = Scripted(["The fee is USD 19,570 per month. "])
    record(alice, tx, [FRAME])
    seg = view(alice, tx)["segments"][0]
    petra = tx["people"]["Petra Voss"]["id"]
    assert revise(alice, tx, seg["id"], text="The fee is USD 19,750 per month.").status_code == 422      # a reason is required
    r = revise(alice, tx, seg["id"], text="The fee is USD 19,750 per month.", speaker_participant_id=petra,
               reason="Checked against the redline")
    assert r.status_code == 200 and [m["kind"] for m in r.json()["revisions"]] == ["text", "speaker"]
    seg = view(alice, tx)["segments"][0]
    assert seg["text"] == "The fee is USD 19,750 per month." and seg["raw_text"] == "The fee is USD 19,570 per month."
    assert seg["speaker"] == "Petra Voss" and seg["revised"]
    assert [(h["kind"], h["by"], h["reason"], h["source"]) for h in seg["revisions"]] == [
        ("text", "alice", "Checked against the redline", "person"), ("speaker", "alice", "Checked against the redline", "person")]
    assert revise(carol, tx, seg["id"], text="x", reason="read-only member").status_code == 403
    assert revise(bob, tx, seg["id"], text="x", reason="not on the matter").status_code == 404
    assert revise(alice, tx, seg["id"], speaker_participant_id="nobody", reason="wrong person").status_code == 409
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with sqlite3.connect(fresh_db) as db:
            db.execute("UPDATE legal_transcript_revisions SET reason='rewritten'")
    details = " ".join(r[0] or "" for r in q(fresh_db, "SELECT details_json FROM audit_events WHERE action='transcript.revise'"))
    assert "19,750" not in details and "redline" not in details


def test_the_matter_s_vocabulary_offers_corrections_with_their_source(tx):
    alice, carol = as_user(tx, "alice"), as_user(tx, "carol")
    msa = upload(alice, tx["mid"], "msa.txt", 'This Agreement is made between Harrowgate Systems Inc. ("Provider") and Voss-Lindqvist AB. '
                 '"Harbourside Premises" means Unit 7 on the estate.', status="executed").json()["document"]
    index_all()
    tx["model"]["t"] = Scripted(["harrow gate systems will vacate the harbor side premises, said petra fos, not the provider's partner. "])
    record(alice, tx, [FRAME])
    got = alice.get(f"{tx['base']}/transcript/suggestions").json()["suggestions"]
    pairs = {(s["heard"], s["suggested"], s["kind"]) for s in got}
    assert ("harrow gate systems", "Harrowgate Systems", "company") in pairs
    assert ("petra fos", "Petra Voss", "participant") in pairs
    assert ("harbor side premises", "Harbourside Premises", "defined_term") in pairs
    assert len(got) == 3                                 # nothing for ordinary words, or "provider's" for the defined "Provider"
    company = next(s for s in got if s["kind"] == "company")
    assert company["source"]["document_id"] == msa["id"] and company["source"]["document_title"] == "msa.txt"
    seg = view(alice, tx)["segments"][0]
    assert seg["raw_text"] == seg["text"]                              # nothing applied by itself
    r = revise(alice, tx, seg["id"], suggestion_id=company["id"], reason="Matter vocabulary")
    assert r.status_code == 200
    seg = view(alice, tx)["segments"][0]
    assert seg["text"].startswith("Harrowgate Systems will vacate")
    rev = seg["revisions"][-1]
    assert rev["source"] == "vocabulary" and rev["provenance"]["heard"] == "harrow gate systems" and rev["provenance"]["document_id"] == msa["id"]
    assert revise(alice, tx, seg["id"], suggestion_id=company["id"], reason="again").status_code == 409   # no longer applies
    # A document the person cannot see lends no vocabulary.
    from app.legal.documents import repo
    repo.set_acl(msa["id"], {tx["users"]["alice"]["id"]})
    assert not [s for s in carol.get(f"{tx['base']}/transcript/suggestions").json()["suggestions"] if s["kind"] != "participant"]


# ── Finalised: a research source ─────────────────────────────────────────────────
def end(c, w):
    r = c.post(f"{w['base']}/end", headers=csrf_headers(c))
    assert r.status_code == 200, r.text


def finalise(c, w):
    return c.post(f"{w['base']}/transcript/finalise", headers=csrf_headers(c))


def test_a_finalised_transcript_is_a_matter_document_that_research_can_cite(tx, monkeypatch):
    from app.legal.research import llm
    alice = as_user(tx, "alice")
    helena = tx["people"]["Helena Marsh"]["id"]
    tx["model"]["t"] = Scripted(["The client confirmed it will not renew the Northbridge licence after 31 December 2026. "])
    record(alice, tx, [{"type": "speaker", "participant_id": helena}, FRAME])
    assert finalise(alice, tx).status_code == 409                          # the meeting is still running
    end(alice, tx)
    r = finalise(alice, tx)
    assert r.status_code == 200 and r.json()["changed"] and r.json()["version_number"] == 1
    index_all()
    doc = next(d for d in alice.get(f"/api/legal/matters/{tx['mid']}/documents").json()["documents"] if d["id"] == r.json()["document_id"])
    assert doc["title"] == "Transcript — Renewal call" and doc["kind"] == "note" and doc["status"] == "final"
    text = alice.get(f"/api/legal/matters/{tx['mid']}/documents/{doc['id']}/versions/{doc['current_version']['id']}/reader").json()["reader"]["text"]
    assert "Helena Marsh: The client confirmed it will not renew the Northbridge licence after 31 December 2026." in text

    class Cite:
        async def chat_stream(self, messages, temperature, max_tokens):
            yield 'The client will not renew the licence [1: "will not renew the Northbridge licence"].'
    monkeypatch.setattr(llm, "_override", Cite())
    ev = [json.loads(line) for line in alice.post(f"/api/legal/matters/{tx['mid']}/research/query", headers=csrf_headers(alice),
                                                  json={"question": "Will the client renew the Northbridge licence?"}).text.splitlines()]
    done = ev[-1]
    assert done["status"] == "answered" and done["evidence"][0]["document_title"] == "Transcript — Renewal call"
    assert done["evidence"][0]["verification"] == "verbatim"

    seg = view(alice, tx)["segments"][0]
    revise(alice, tx, seg["id"], text=seg["text"].replace("31 December 2026", "30 November 2026"), reason="Client email")
    assert view(alice, tx)["finalised"]["stale"] is True                  # corrected since it was finalised
    again = finalise(alice, tx).json()
    assert again["changed"] and again["version_number"] == 2 and again["document_id"] == doc["id"]
    assert finalise(alice, tx).json()["changed"] is False                  # nothing new to finalise


def test_purging_a_transcript_purges_its_corrections_and_its_finalised_copy(tx, fresh_db):
    from app.legal.records import deletion
    alice = as_user(tx, "alice")
    tx["model"]["t"] = Scripted(["Privileged advice on the settlement value. "])
    record(alice, tx, [FRAME])
    seg = view(alice, tx)["segments"][0]
    revise(alice, tx, seg["id"], text="Privileged advice on the settlement.", reason="Tidy")
    end(alice, tx)
    doc_id = finalise(alice, tx).json()["document_id"]
    index_all()
    org = tx["matters"]["M-101"]["org_id"]
    d = deletion.request(org, tx["mid"], "meeting_transcript", tx["meeting"]["id"], "Minutes approved", tx["users"]["alice"]["id"])
    assert d["report"]["revisions"] == 1 and d["report"]["finalised_document"] == 1
    listed = [x["id"] for x in alice.get(f"/api/legal/matters/{tx['mid']}/documents").json()["documents"]]
    assert doc_id not in listed                                            # hidden with the transcript at once
    done = deletion.purge(d["id"], tx["users"]["mp"]["id"])
    assert done["verification"]["ok"] and done["verification"]["checks"]["finalised_document"], done["verification"]
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_transcript_revisions")[0][0] == 0
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_documents WHERE id=?", doc_id)[0][0] == 0
    assert q(fresh_db, "SELECT COUNT(*) FROM legal_chunks WHERE document_id=?", doc_id)[0][0] == 0


def test_a_restored_transcript_brings_back_its_finalised_copy(tx):
    from app.legal.records import deletion
    alice = as_user(tx, "alice")
    tx["model"]["t"] = Scripted(["Notes for the file. "])
    record(alice, tx, [FRAME])
    end(alice, tx)
    doc_id = finalise(alice, tx).json()["document_id"]
    index_all()
    org = tx["matters"]["M-101"]["org_id"]
    d = deletion.request(org, tx["mid"], "meeting_transcript", tx["meeting"]["id"], "Mistake", tx["users"]["alice"]["id"])
    deletion.restore(d["id"], tx["users"]["alice"]["id"])
    assert doc_id in [x["id"] for x in alice.get(f"/api/legal/matters/{tx['mid']}/documents").json()["documents"]]
    assert view(alice, tx)["segments"]


def test_exports_use_the_corrected_transcript_with_speakers(tx):
    from app.legal.exports import source
    alice = as_user(tx, "alice")
    helena = tx["people"]["Helena Marsh"]["id"]
    tx["model"]["t"] = Scripted(["The cap is USD 250,000. "])
    record(alice, tx, [FRAME])
    seg = view(alice, tx)["segments"][0]
    revise(alice, tx, seg["id"], text="The cap is USD 500,000.", speaker_participant_id=helena, reason="Draft v2")
    s = source.load(tx["mid"], "meeting_transcript", tx["meeting"]["id"])
    assert "Helena Marsh: The cap is USD 500,000." in s.text and "250,000" not in s.text


def test_a_rollback_refuses_to_delete_corrections(tx):
    from app.core.config import settings
    from app.legal.migrations import runner
    alice = as_user(tx, "alice")
    tx["model"]["t"] = Scripted(["Text. "])
    record(alice, tx, [FRAME])
    seg = view(alice, tx)["segments"][0]
    revise(alice, tx, seg["id"], text="Corrected text.", reason="Checked")
    with sqlite3.connect(settings.DB_PATH) as conn, pytest.raises(Exception, match="transcript correction"):
        runner.migrate_down(conn, 17)
