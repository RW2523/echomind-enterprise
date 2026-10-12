"""S5.1 — the conversation assistant (decision D-062).

A lawyer speaks a question about a matter. The server transcribes it with its own speech recognition, the
research pipeline answers it from the matter's documents with checked quotes (as a typed question would be), and
the answer is read aloud by the server's own voice. Off unless the firm turns it on (LEGAL_CONVERSATION). Nothing
leaves the server, the audio is not kept, and the audit holds lengths only.
"""
from __future__ import annotations

import json
import sqlite3
import struct

import pytest

from conftest import as_user, csrf_headers

PCM = struct.pack("<h", 1000) * 16000                                   # one second at 16 kHz
QUESTION = "What is the liability cap under the agreement?"


class Ears:
    """Stands in for speech recognition."""
    def __init__(self):
        self.calls, self.fail = [], None

    async def __call__(self, pcm: bytes, rate: int) -> str:
        if self.fail:
            raise self.fail
        self.calls.append((len(pcm), rate))
        return QUESTION


class Voice:
    """Stands in for the voice service."""
    def __init__(self):
        self.calls, self.fail = [], None

    async def __call__(self, text: str) -> bytes:
        if self.fail:
            raise self.fail
        self.calls.append(text)
        return b"RIFF\x24\x00\x00\x00WAVEfmt "


@pytest.fixture
def cv(docs, monkeypatch):
    from app.core.config import settings
    from app.legal.conversation import routes
    monkeypatch.setattr(settings, "LEGAL_CONVERSATION", True)
    ears, voice = Ears(), Voice()
    monkeypatch.setattr(routes, "transcribe_clip", ears)
    monkeypatch.setattr(routes, "synthesise", voice)
    mid = docs["matters"]["M-101"]["id"]
    docs.update(mid=mid, ears=ears, voice=voice, base=f"/api/legal/matters/{mid}/conversation")
    return docs


def hear(c, base, body=PCM, rate=16000):
    return c.post(f"{base}/transcribe?sample_rate={rate}", headers={**csrf_headers(c), "Content-Type": "application/octet-stream"},
                  content=body)


def speak(c, base, text):
    return c.post(f"{base}/speak", headers=csrf_headers(c), json={"text": text})


def test_off_unless_the_firm_turns_it_on(docs):
    alice = as_user(docs, "alice")
    base = f"/api/legal/matters/{docs['matters']['M-101']['id']}/conversation"
    assert hear(alice, base).status_code == 404 and speak(alice, base, "x").status_code == 404
    assert alice.get("/api/auth/me").json()["features"]["conversation"] is False


def test_a_spoken_question_is_transcribed_on_the_server(cv):
    alice = as_user(cv, "alice")
    r = hear(alice, cv["base"])
    assert r.status_code == 200 and r.json() == {"text": QUESTION, "seconds": 1.0}
    assert cv["ears"].calls == [(len(PCM), 16000)]
    assert alice.get("/api/auth/me").json()["features"]["conversation"] is True


def test_an_answer_is_read_aloud_without_its_citation_marks(cv):
    alice = as_user(cv, "alice")
    r = speak(alice, cv["base"], 'The cap is USD 250,000 [1: "limited to USD 250,000"] under clause 9.1 [2][3].\n\n**Note:** check 9.2.')
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.headers["cache-control"] == "no-store"
    assert r.content.startswith(b"RIFF")
    assert cv["voice"].calls == ["The cap is USD 250,000 under clause 9.1. Note: check 9.2."]


def test_only_people_on_the_matter_can_use_it(cv):
    bob = as_user(cv, "bob")                                              # not on M-101: the matter does not exist for him
    assert hear(bob, cv["base"]).status_code == 404 and speak(bob, cv["base"], "x").status_code == 404
    assert cv["ears"].calls == [] and cv["voice"].calls == []


def test_a_clip_is_bounded_and_must_be_audio(cv, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LEGAL_CONVERSATION_MAX_SECONDS", 2)
    alice = as_user(cv, "alice")
    assert hear(alice, cv["base"], PCM * 3).status_code == 413                                  # three seconds
    assert hear(alice, cv["base"], b"").status_code == 400
    assert hear(alice, cv["base"], PCM + b"\x00").status_code == 400                           # not 16-bit samples
    assert hear(alice, cv["base"], rate=4000).status_code == 400
    assert speak(alice, cv["base"], "x" * 2001).status_code == 422
    assert speak(alice, cv["base"], " [1] ").status_code == 400                                # nothing to say
    assert cv["ears"].calls == [] and cv["voice"].calls == []


def test_without_speech_recognition_or_a_voice_the_server_says_so(cv):
    from app.legal.conversation.routes import Unavailable
    alice = as_user(cv, "alice")
    cv["ears"].fail = Unavailable("speech")
    r = hear(alice, cv["base"])
    assert r.status_code == 503 and r.json()["detail"] == "Speech recognition is not installed on this server. Type your question instead."
    cv["voice"].fail = Unavailable("voice")
    r = speak(alice, cv["base"], "The cap is USD 250,000.")
    assert r.status_code == 503 and r.json()["detail"] == "Reading aloud is not available on this server."


def test_the_audit_holds_lengths_only(cv, fresh_db):
    alice = as_user(cv, "alice")
    hear(alice, cv["base"])
    speak(alice, cv["base"], "The cap is USD 250,000 under clause 9.1.")
    with sqlite3.connect(fresh_db) as db:
        rows = dict(db.execute("SELECT action, details_json FROM audit_events WHERE action LIKE 'conversation.%'").fetchall())
    assert json.loads(rows["conversation.transcribe"]) == {"policy": "research.query", "seconds": 1.0, "words": 8}
    assert json.loads(rows["conversation.speak"]) == {"policy": "research.query", "chars": 40}
    assert "liability" not in " ".join(rows.values()) and "250,000" not in " ".join(rows.values())
