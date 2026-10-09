"""S1.8 / SEC-05 — consent and capture control (spec test T03; policy P-001 safe defaults).

Capture of a matter's meeting opens only when every current participant has given explicit,
fresh consent; withdrawal stops it at once and no audio after the withdrawal is used; pause and
resume are recorded; everything is matter-scoped and audited.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest
from starlette.websockets import WebSocketDisconnect

from conftest import as_user, csrf_headers

WSS = "wss://testserver"
ORIGIN = {"origin": "https://testserver"}
FRAME = b"\x01\x00" * 1600          # 100 ms of 16 kHz PCM16


class FakeTranscriber:
    """Stands in for live speech recognition: one sentence per audio frame."""
    instances: list = []

    def __init__(self):
        self.fed = 0
        FakeTranscriber.instances.append(self)

    async def feed(self, pcm16: bytes, sample_rate: int):
        self.fed += 1
        return [(f"Sentence {self.fed}. ", self.fed * 100)]

    async def flush(self):
        return []


@pytest.fixture
def meeting(world, monkeypatch):
    from app.core.config import settings
    from app.legal.meetings import capture
    FakeTranscriber.instances = []
    monkeypatch.setattr(capture, "transcriber_factory", lambda: FakeTranscriber())
    monkeypatch.setattr(settings, "LEGAL_WS_RECHECK_SEC", 0.1)
    monkeypatch.setattr(settings, "LEGAL_CONSENT_RECHECK_SEC", 0.1)
    alice = as_user(world, "alice")
    mid = world["matters"]["M-101"]["id"]
    m = alice.post(f"/api/legal/matters/{mid}/meetings", headers=csrf_headers(alice),
                   json={"title": "Call with Voss-Lindqvist", "kind": "client_call"}).json()["meeting"]
    base = f"/api/legal/matters/{mid}/meetings/{m['id']}"
    people = {}
    for name, role in (("Alice Hart", "host"), ("Petra Voss", "client"), ("Jon Hale", "counterparty")):
        people[name] = alice.post(f"{base}/participants", headers=csrf_headers(alice), json={"name": name, "role": role}).json()["participant"]
    world.update(meeting=m, base=base, mid=mid, people=people)
    return world


def consent(c, w, name, method="verbal_on_record"):
    r = c.post(f"{w['base']}/participants/{w['people'][name]['id']}/consent", headers=csrf_headers(c),
               json={"method": method, "statement_read": True})
    assert r.status_code == 200, r.text


def withdraw(c, w, name):
    r = c.post(f"{w['base']}/participants/{w['people'][name]['id']}/consent/withdraw", headers=csrf_headers(c))
    assert r.status_code == 200, r.text


def everyone_consents(w):
    alice = as_user(w, "alice")
    for name in w["people"]:
        consent(alice, w, name)


def connect(c, w):
    return c.websocket_connect(f"{WSS}{w['base']}/capture", headers=ORIGIN)


def refused(c, w) -> int:
    with pytest.raises(WebSocketDisconnect) as e:
        with connect(c, w) as ws:
            ws.receive_json()
    return e.value.code


def rows(db, sql, *args):
    with sqlite3.connect(db) as c:
        return c.execute(sql, args).fetchall()


# ── Who may do what ──────────────────────────────────────────────────────────────
def test_meetings_belong_to_the_matter(meeting):
    w = meeting
    carol, bob = as_user(w, "carol"), as_user(w, "bob")
    assert carol.post(f"/api/legal/matters/{w['mid']}/meetings", headers=csrf_headers(carol), json={"title": "x"}).status_code == 403
    assert bob.get(f"/api/legal/matters/{w['mid']}/meetings").status_code == 404
    listed = carol.get(f"/api/legal/matters/{w['mid']}/meetings").json()["meetings"]
    assert listed[0]["title"] == "Call with Voss-Lindqvist" and listed[0]["consent"] == {"granted": 0, "needed": 3}
    other = w["matters"]["M-114"]["id"]
    assert as_user(w, "alice").get(f"/api/legal/matters/{other}/meetings/{w['meeting']['id']}").status_code == 404


# ── T03: consent gates capture ───────────────────────────────────────────────────
def test_capture_is_refused_until_every_participant_has_consented(meeting):
    w = meeting
    alice = as_user(w, "alice")
    assert refused(alice, w) == 4412
    consent(alice, w, "Alice Hart")
    consent(alice, w, "Petra Voss")
    assert refused(alice, w) == 4412
    assert FakeTranscriber.instances == []                      # speech recognition never started
    consent(alice, w, "Jon Hale")
    with connect(alice, w) as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["transcription"] == "available" and len(ready["consent_snapshot"]) == 64


def test_withdrawn_stale_or_new_participants_refuse_start(meeting, monkeypatch):
    from app.core.config import settings
    w = meeting
    alice = as_user(w, "alice")
    everyone_consents(w)
    withdraw(alice, w, "Jon Hale")
    assert refused(alice, w) == 4412
    consent(alice, w, "Jon Hale")
    with connect(alice, w) as ws:
        assert ws.receive_json()["type"] == "ready"
    monkeypatch.setattr(settings, "LEGAL_CONSENT_MAX_AGE_HOURS", 0.0)
    assert refused(alice, w) == 4412                             # stale
    monkeypatch.setattr(settings, "LEGAL_CONSENT_MAX_AGE_HOURS", 12.0)
    alice.post(f"{w['base']}/participants", headers=csrf_headers(alice), json={"name": "Late Joiner", "role": "other"})
    assert refused(alice, w) == 4412                             # someone new has not consented


def test_withdrawal_mid_capture_stops_it_and_later_audio_is_never_used(meeting, fresh_db):
    w = meeting
    alice = as_user(w, "alice")
    everyone_consents(w)
    with connect(alice, w) as ws:
        assert ws.receive_json()["type"] == "ready"
        for _ in range(3):
            ws.send_bytes(FRAME)
            assert ws.receive_json()["type"] == "segment"
        withdraw(as_user(w, "carol"), w, "Petra Voss")            # any matter member may record a withdrawal
        stopped = ws.receive_json()
        assert stopped == {"type": "stopped", "reason": "consent_withdrawn"}
        with pytest.raises(Exception):
            ws.send_bytes(FRAME)
            ws.receive_json()
    t_withdraw = rows(fresh_db, "SELECT recorded_at FROM legal_consent_events WHERE kind='withdrawn'")[0][0]
    assert FakeTranscriber.instances[0].fed == 3
    for (received_at,) in rows(fresh_db, "SELECT received_at FROM legal_capture_custody"):
        assert received_at <= t_withdraw
    events = [r[0] for r in rows(fresh_db, "SELECT kind FROM legal_capture_events ORDER BY at, rowid")]
    assert events[0] == "started" and events[-1] == "stopped"
    assert rows(fresh_db, "SELECT reason FROM legal_capture_events WHERE kind='stopped'") == [("consent_withdrawn",)]


def test_a_withdrawal_racing_the_start_never_captures_after_it(meeting, fresh_db):
    """Start and withdrawal at the same moment: whichever wins, no accepted audio is later than the
    withdrawal and no transcript segment is written after it."""
    w = meeting
    alice, carol = as_user(w, "alice"), as_user(w, "carol")
    for attempt in range(5):
        everyone_consents(w)
        t = threading.Thread(target=withdraw, args=(carol, w, "Jon Hale"))
        try:
            with connect(alice, w) as ws:
                t.start()
                ws.receive_json()
                for _ in range(3):
                    ws.send_bytes(FRAME)
                    ws.receive_json()
        except (WebSocketDisconnect, Exception):
            pass
        t.join()
    assert rows(fresh_db, "SELECT COUNT(*) FROM legal_consent_events WHERE kind='withdrawn'")[0][0] == 5
    checked = 0
    for (cap, received_at, withdrawn_at) in rows(fresh_db, """
            SELECT k.capture_id, k.received_at, (SELECT MIN(e.recorded_at) FROM legal_consent_events e
              WHERE e.kind='withdrawn' AND e.meeting_id = c.meeting_id AND e.recorded_at >= c.started_at)
            FROM legal_capture_custody k JOIN legal_captures c ON c.id = k.capture_id"""):
        if withdrawn_at:
            checked += 1
            assert received_at <= withdrawn_at, (cap, received_at, withdrawn_at)
    for (cap, created_at, withdrawn_at) in rows(fresh_db, """
            SELECT s.capture_id, s.created_at, (SELECT MIN(e.recorded_at) FROM legal_consent_events e
              WHERE e.kind='withdrawn' AND e.meeting_id = c.meeting_id AND e.recorded_at >= c.started_at)
            FROM legal_transcript_segments s JOIN legal_captures c ON c.id = s.capture_id"""):
        if withdrawn_at:
            assert created_at <= withdrawn_at, (cap, created_at, withdrawn_at)


def test_pause_is_recorded_and_paused_audio_is_not_transcribed(meeting, fresh_db):
    w = meeting
    alice = as_user(w, "alice")
    everyone_consents(w)
    with connect(alice, w) as ws:
        ws.receive_json()
        ws.send_bytes(FRAME)
        ws.receive_json()
        ws.send_json({"type": "pause"})
        assert ws.receive_json() == {"type": "state", "state": "paused"}
        ws.send_bytes(FRAME)
        ws.send_bytes(FRAME)
        ws.send_json({"type": "resume"})
        assert ws.receive_json() == {"type": "state", "state": "live"}
        ws.send_bytes(FRAME)
        assert ws.receive_json()["type"] == "segment"
        ws.send_json({"type": "stop"})
        assert ws.receive_json() == {"type": "stopped", "reason": "ended"}
    assert FakeTranscriber.instances[0].fed == 2
    kinds = [r[0] for r in rows(fresh_db, "SELECT kind FROM legal_capture_events ORDER BY at, rowid")]
    assert kinds == ["started", "paused", "resumed", "stopped"]


def test_losing_access_stops_capture(meeting):
    from app.legal.matters import repo
    w = meeting
    carol_mm = [m for m in repo.list_live_memberships(w["mid"]) if m["username"] == "carol"][0]
    repo.change_membership(w["mid"], carol_mm["id"], by=w["users"]["alice"]["id"], role="paralegal")
    carol = as_user(w, "carol")
    everyone_consents(w)
    with connect(carol, w) as ws:
        assert ws.receive_json()["type"] == "ready"
        for m in repo.list_live_memberships(w["mid"]):
            if m["username"] == "carol":
                repo.revoke_membership(w["mid"], m["id"], by=w["users"]["alice"]["id"], reason="left")
        deadline = time.time() + 3
        msg = ws.receive_json()
        while msg.get("type") != "stopped" and time.time() < deadline:
            msg = ws.receive_json()
        assert msg == {"type": "stopped", "reason": "access_revoked"}


def test_capture_needs_a_contributor_on_an_open_matter_and_a_known_meeting(meeting):
    w = meeting
    everyone_consents(w)
    assert refused(as_user(w, "carol"), w) == 4403                # read-only member
    assert refused(as_user(w, "bob"), w) == 4404                  # not a member
    alice = as_user(w, "alice")
    alice.patch(f"/api/legal/matters/{w['mid']}", headers=csrf_headers(alice), json={"status": "closed", "version": 1})
    assert refused(alice, w) == 4409


def test_without_speech_recognition_capture_control_still_works(meeting, monkeypatch, fresh_db):
    from app.legal.meetings import capture
    monkeypatch.setattr(capture, "transcriber_factory", lambda: None)
    w = meeting
    everyone_consents(w)
    with connect(as_user(w, "alice"), w) as ws:
        assert ws.receive_json()["transcription"] == "unavailable"
        ws.send_bytes(FRAME)
        ws.send_json({"type": "stop"})
        assert ws.receive_json() == {"type": "stopped", "reason": "ended"}
    assert rows(fresh_db, "SELECT COUNT(*) FROM legal_capture_custody")[0][0] == 1


# ── Records ──────────────────────────────────────────────────────────────────────
def test_consent_and_capture_timelines_are_append_only(meeting, fresh_db):
    everyone_consents(meeting)
    with sqlite3.connect(fresh_db) as c:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            c.execute("UPDATE legal_consent_events SET kind='withdrawn'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            c.execute("DELETE FROM legal_consent_events")


def test_meeting_detail_shows_consent_and_the_capture_timeline_and_audit_has_no_names(meeting, fresh_db):
    w = meeting
    alice = as_user(w, "alice")
    everyone_consents(w)
    with connect(alice, w) as ws:
        ws.receive_json()
        ws.send_bytes(FRAME)
        ws.receive_json()
        ws.send_json({"type": "stop"})
        ws.receive_json()
    d = alice.get(f"{w['base']}").json()["meeting"]
    assert {p["name"]: p["consent"]["state"] for p in d["participants"]} == {n: "granted" for n in w["people"]}
    assert [e["kind"] for e in d["captures"][0]["events"]] == ["started", "stopped"]
    seg = alice.get(f"{w['base']}/transcript").json()["segments"]
    assert seg[0]["text"].startswith("Sentence 1")
    assert as_user(w, "bob").get(f"{w['base']}/transcript").status_code == 404
    acts = {r[0] for r in rows(fresh_db, "SELECT action FROM audit_events")}
    assert {"meeting.create", "participant.add", "consent.grant", "capture.start", "capture.stop"} <= acts
    blob = json.dumps(rows(fresh_db, "SELECT details_json, reason FROM audit_events"))
    assert "Petra" not in blob and "Voss-Lindqvist" not in blob and "Sentence 1" not in blob
