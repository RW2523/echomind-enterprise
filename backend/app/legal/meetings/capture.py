"""Consent-gated capture (S1.8, SEC-05; spec test T03).

The socket /api/legal/matters/{m}/meetings/{id}/capture opens only when the person may control
capture on the matter (an open matter, a contributor), the meeting is in that matter, and every
current participant has given fresh, explicit consent. Otherwise it closes before anything is
loaded: 4403 not permitted, 4404 not visible, 4409 the
matter or meeting is closed, 4412 consent required; no session or a foreign origin: 1008 (as
every Law socket). A session that ends while capturing closes it with 4401.

Withdrawal versus audio: one lock per meeting. Recording a withdrawal stamps its time and flags
every live capture of the meeting while holding the lock; accepting an audio frame and storing a
transcript segment check the flag and stamp their own time while holding the same lock. So no
audio accepted, and no text stored, is ever later than the withdrawal. A database re-check every
LEGAL_CONSENT_RECHECK_SEC covers captures running in another process.

Raw audio is not kept: each ~5 s of received audio leaves a SHA-256 custody record. Paused
audio is discarded unread. Transcription runs through a `Transcriber` (`feed`, `flush`); the
default wraps the live speech-recognition model when it is installed, and capture control works
without it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from ...core.config import settings
from ...core.db import get_conn
from .. import policy
from ..audit.record import record
from ..identity import sessions
from ..identity.principal import from_websocket, origin_allowed
from . import consent, repo

logger = logging.getLogger(__name__)

CLOSE = {"unauthenticated": 4401, "origin": 1008, "forbidden": 4403, "not_found": 4404, "closed": 4409, "consent": 4412}
_MAX_FRAME = 64 * 1024
_SENTENCE_END = re.compile(r"[.!?…]\s*$")


# ── Transcription ────────────────────────────────────────────────────────────────
class NemotronTranscriber:
    """The live speech-recognition model, one streaming context per capture."""

    def __init__(self, stt):
        self._stt = stt
        self._adapter = stt.get_shared_asr_adapter()
        self._ctx = stt.NemotronStreamContext(self._adapter)

    async def feed(self, pcm16: bytes, sample_rate: int):
        audio = self._stt._pcm16_to_float32(pcm16)
        return await self._ctx.process_pcm_chunk(audio, sample_rate, asyncio.get_running_loop(), self._adapter.device)

    async def flush(self):
        return await self._ctx.flush(asyncio.get_running_loop())


def transcriber_factory():
    """A Transcriber, or None when live speech recognition is not installed on this server."""
    try:
        from ...transcribe import stt_streaming as stt
        if not getattr(stt, "NEMOTRON_AVAILABLE", False):
            return None
        return NemotronTranscriber(stt)
    except Exception as e:
        logger.warning("Capture: speech recognition unavailable (%s)", type(e).__name__)
        return None


# ── Live captures and the per-meeting lock ───────────────────────────────────────
class Handle:
    def __init__(self, meeting_id: str, loop: asyncio.AbstractEventLoop):
        self.meeting_id = meeting_id
        self.loop = loop
        self.stopped = asyncio.Event()
        self.reason: Optional[str] = None

    def stop(self, reason: str) -> None:
        """Thread-safe; call with the meeting lock held when the stop must be ordered with frames."""
        if self.reason is None:
            self.reason = reason
        self.loop.call_soon_threadsafe(self.stopped.set)


_registry_lock = threading.Lock()
_locks: dict[str, threading.Lock] = {}
_live: dict[str, set[Handle]] = {}


def meeting_lock(meeting_id: str) -> threading.Lock:
    with _registry_lock:
        return _locks.setdefault(meeting_id, threading.Lock())


def stop_meeting(meeting_id: str, reason: str) -> int:
    """Flag every live capture of the meeting (caller holds the meeting lock)."""
    with _registry_lock:
        handles = list(_live.get(meeting_id, ()))
    for h in handles:
        h.stop(reason)
    return len(handles)


def _register(h: Handle) -> None:
    with _registry_lock:
        _live.setdefault(h.meeting_id, set()).add(h)


def _unregister(h: Handle) -> None:
    with _registry_lock:
        _live.get(h.meeting_id, set()).discard(h)


# ── Records ──────────────────────────────────────────────────────────────────────
def _event(capture_id: str, kind: str, reason: Optional[str] = None, by: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_capture_events (id, capture_id, kind, reason, at, by) VALUES (?,?,?,?,?,?)",
                     (str(uuid.uuid4()), capture_id, kind, reason, policy.now_iso(), by))
        conn.commit()


class _Custody:
    """Hash of each received batch of audio (the audio itself is not stored)."""

    def __init__(self, capture_id: str):
        self.capture_id, self.seq, self.h, self.n, self.last_at = capture_id, 0, hashlib.sha256(), 0, None

    def add(self, data: bytes, at: str) -> None:
        self.h.update(data)
        self.n += len(data)
        self.last_at = at
        if self.n >= settings.LEGAL_CUSTODY_BATCH_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.n:
            return
        with get_conn() as conn:
            conn.execute("INSERT INTO legal_capture_custody (capture_id, seq, sha256, bytes, received_at) VALUES (?,?,?,?,?)",
                         (self.capture_id, self.seq, self.h.hexdigest(), self.n, self.last_at))
            conn.commit()
        self.seq, self.h, self.n = self.seq + 1, hashlib.sha256(), 0


# ── The socket ───────────────────────────────────────────────────────────────────
async def _refuse(ws: WebSocket, code: int, principal: Optional[dict], matter_id: str, reason: str) -> None:
    if principal is not None:
        await asyncio.to_thread(record, "capture.start", "deny", principal=principal, object_type="meeting",
                                matter_id=matter_id, reason=reason)
    await ws.close(code=code)


async def handle(ws: WebSocket, matter_id: str, meeting_id: str) -> None:
    if not origin_allowed(ws.headers.get("origin")):
        await ws.close(code=CLOSE["origin"])
        return
    p = await asyncio.to_thread(from_websocket, ws)
    if not p:
        await ws.close(code=CLOSE["origin"])             # no session at the handshake: same refusal as S1.1
        return
    d = await asyncio.to_thread(policy.authorize, p, "capture.control", matter_id)
    if not d.allowed:
        code = {404: CLOSE["not_found"], 403: CLOSE["forbidden"], 409: CLOSE["closed"]}.get(d.status, CLOSE["forbidden"])
        await _refuse(ws, code, p, matter_id, d.reason)
        return
    meeting = await asyncio.to_thread(repo.get_meeting_row, matter_id, meeting_id)
    if not meeting:
        await _refuse(ws, CLOSE["not_found"], p, matter_id, "not_found")
        return
    if meeting["status"] != "planned":
        await _refuse(ws, CLOSE["closed"], p, matter_id, "meeting_ended")
        return
    try:
        sample_rate = int(ws.query_params.get("sample_rate") or 16000)
    except ValueError:
        sample_rate = 0
    if not 8000 <= sample_rate <= 48000:
        await ws.close(code=1003)
        return

    h = Handle(meeting_id, asyncio.get_running_loop())
    lock = meeting_lock(meeting_id)

    def open_capture() -> Optional[tuple[str, str]]:
        # Registration and the consent check happen under the lock, so a withdrawal is either seen
        # here or flags this capture — never missed.
        with lock:
            ok, _, snapshot = consent.check(meeting_id)
            if not ok:
                return None
            _register(h)
            cid = repo.create_capture(meeting_id, matter_id, p["sub"], snapshot, "available" if transcriber else "unavailable")
            _event(cid, "started", by=p["sub"])
            return cid, snapshot

    if not (await asyncio.to_thread(consent.check, meeting_id))[0]:
        await _refuse(ws, CLOSE["consent"], p, matter_id, "consent_required")
        return
    transcriber = transcriber_factory()
    opened = await asyncio.to_thread(open_capture)
    if opened is None:
        await _refuse(ws, CLOSE["consent"], p, matter_id, "consent_required")
        return
    capture_id, snapshot = opened
    await asyncio.to_thread(record, "capture.start", "allow", principal=p, object_type="capture", object_id=capture_id,
                            matter_id=matter_id, details={"transcription": "available" if transcriber else "unavailable"})
    send_lock = asyncio.Lock()

    async def send(obj: dict) -> None:
        async with send_lock:
            await ws.send_text(json.dumps(obj))

    await ws.accept()
    await send({"type": "ready", "capture_id": capture_id, "consent_snapshot": snapshot,
                "transcription": "available" if transcriber else "unavailable"})

    custody = _Custody(capture_id)
    state = {"paused": False, "seq": 0, "text": "", "t0": None, "t1": None}

    def stop_locked(reason: str) -> None:
        with lock:
            h.stop(reason)

    def accept_frame(data: bytes) -> bool:
        with lock:
            if h.reason is not None:
                return False
            custody.add(data, policy.now_iso())
            return True

    def store_segment(text: str, t0, t1) -> Optional[dict]:
        with lock:
            if h.reason is not None:
                return None                           # never store text once capture was stopped
            seg = repo.add_segment(capture_id, meeting_id, matter_id, state["seq"], t0, t1, text)
        state["seq"] += 1
        return seg

    async def emit(pieces) -> None:
        for piece, ts in pieces:
            if not piece.strip():
                continue
            state["text"] += piece
            state["t0"] = ts if state["t0"] is None else state["t0"]
            state["t1"] = ts
            if _SENTENCE_END.search(state["text"]) or len(state["text"]) > 400:
                await commit()

    async def commit() -> None:
        text = state["text"].strip()
        t0, t1 = state["t0"], state["t1"]
        state["text"], state["t0"] = "", None
        if not text:
            return
        seg = await asyncio.to_thread(store_segment, text, t0, t1)
        if seg is not None:
            await send({"type": "segment", **seg})

    async def watch() -> None:
        """Access and consent re-checks; ends the capture when either lapses."""
        last_policy = last_consent = time.monotonic()
        while h.reason is None:
            await asyncio.sleep(0.05)
            now = time.monotonic()
            if now - last_policy >= settings.LEGAL_WS_RECHECK_SEC:
                last_policy = now
                live = await asyncio.to_thread(sessions.lookup_hash, p["sid"], True)
                allowed = (await asyncio.to_thread(policy.authorize, p, "capture.control", matter_id)).allowed
                if not live or not live.get("mfa_verified"):
                    h.stop("signed_out")
                elif not allowed:
                    h.stop("access_revoked")
            if now - last_consent >= settings.LEGAL_CONSENT_RECHECK_SEC and h.reason is None:
                last_consent = now
                if not (await asyncio.to_thread(consent.check, meeting_id))[0]:
                    await asyncio.to_thread(stop_locked, "consent_withdrawn")

    async def receive() -> None:
        while h.reason is None:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                h.stop("disconnected")
                return
            data = msg.get("bytes")
            if data is not None:
                if len(data) > _MAX_FRAME:
                    h.stop("frame_too_large")
                    return
                if state["paused"]:
                    continue                          # paused audio is discarded unread
                if not await asyncio.to_thread(accept_frame, data):
                    await asyncio.to_thread(_event, capture_id, "frame_rejected", h.reason)
                    return
                if transcriber is not None:
                    try:
                        await emit(await transcriber.feed(data, sample_rate))
                    except Exception as e:
                        logger.warning("Capture: transcription error (%s)", type(e).__name__)
                continue
            try:
                obj = json.loads(msg.get("text") or "{}")
            except ValueError:
                continue
            kind = obj.get("type") if isinstance(obj, dict) else None
            if kind == "pause" and not state["paused"]:
                state["paused"] = True
                await asyncio.to_thread(_event, capture_id, "paused", None, p["sub"])
                await send({"type": "state", "state": "paused"})
            elif kind == "resume" and state["paused"]:
                state["paused"] = False
                await asyncio.to_thread(_event, capture_id, "resumed", None, p["sub"])
                await send({"type": "state", "state": "live"})
            elif kind == "stop":
                h.stop("ended")
                return

    tasks = [asyncio.create_task(watch()), asyncio.create_task(receive()), asyncio.create_task(h.stopped.wait())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        reason = h.reason or "ended"
        _unregister(h)
        if reason in ("ended", "disconnected") and transcriber is not None:
            try:
                await emit(await transcriber.flush())
            except Exception:
                pass
        if reason in ("ended", "disconnected"):
            await commit()
        await asyncio.to_thread(custody.flush)
        await asyncio.to_thread(_event, capture_id, "stopped", reason, p["sub"])
        await asyncio.to_thread(record, "capture.stop", "allow", principal=p, object_type="capture", object_id=capture_id,
                                matter_id=matter_id, reason=reason)
        if reason != "disconnected":
            try:
                await send({"type": "stopped", "reason": reason})
                await ws.close(code=CLOSE["unauthenticated"] if reason == "signed_out" else 1000)
            except Exception:
                pass
