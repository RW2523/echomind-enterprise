"""
WebSocket handler for real-time transcription & knowledge capture.
Protocol: binary PCM16 chunks, text JSON (start/pause/resume/eos/refine/store).
Uses SessionState for stabilization and segmentation; refine and store for KB.
"""
from __future__ import annotations
import asyncio
import base64
import json
import time
import uuid
import numpy as np
from fastapi import WebSocket
from typing import Optional

from ..core.config import settings
from ..utils.ids import now_iso
from .session_state import SessionState, Paragraph
from .stt_streaming import (
    _pcm16_to_float32,
    _load_whisper,
    get_kyutai_stt,
    KYUTAI_AVAILABLE,
    _resample_linear,
)
from ..refine import refine_text
from ..tagging import get_metadata
from .. import kb

SAMPLE_RATE_WHISPER = getattr(settings, "SAMPLE_RATE", 16000)
# Emit partial every this many seconds of audio (Whisper chunk)
WHISPER_CHUNK_SEC = 2.5
# Rate limit partials to client
EMIT_MIN_INTERVAL = 1.0 / max(0.1, getattr(settings, "TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC", 15))


async def handler(ws: WebSocket):
    await ws.accept()
    await _send(ws, {"type": "loading"})

    loop = asyncio.get_running_loop()
    use_kyutai = False
    sample_rate = SAMPLE_RATE_WHISPER
    kyutai_stt = None
    if KYUTAI_AVAILABLE:
        try:
            kyutai_stt = await loop.run_in_executor(None, get_kyutai_stt)
            if kyutai_stt is not None:
                use_kyutai = True
                sample_rate = kyutai_stt.sample_rate
        except Exception:
            pass
    if not use_kyutai:
        try:
            await loop.run_in_executor(None, _load_whisper)
        except Exception as e:
            await _send(ws, {"type": "error", "message": f"STT load failed: {e}"})
            return
    await _send(ws, {"type": "ready", "sample_rate": sample_rate})

    session_id: Optional[str] = None
    session: Optional[SessionState] = None
    mode = "transcribe"
    language = "en"
    auto_store = settings.ECHOMIND_AUTO_STORE_DEFAULT
    started_at: Optional[float] = None
    audio_buffer: list = []
    last_emit_time = 0.0
    last_whisper_time = 0.0
    client_sample_rate: Optional[int] = None
    last_auto_stored_length: list = [0]  # mutable for closure
    periodic_auto_store_task: Optional[asyncio.Task] = None
    auto_store_interval_sec = max(0, getattr(settings, "AUTO_STORE_INTERVAL_SEC", 60))

    async def _periodic_auto_store_fn() -> None:
        """Every auto_store_interval_sec, store new transcript content since last store."""
        while True:
            await asyncio.sleep(auto_store_interval_sec)
            if session is None:
                break
            try:
                full_text = session.get_display_text()
                to_store = full_text[last_auto_stored_length[0] :].strip()
                if to_store:
                    conv_type, tags = get_metadata(to_store)
                    meta = {"session_id": session_id, "kind": "raw", "tags": tags, "conversation_type": conv_type, "ts": now_iso()}
                    kid = await kb.kb_add_text(to_store, meta)
                    await _send(ws, {"type": "stored", "session_id": session_id, "items": [{"id": kid, "kind": "raw", "tags": tags, "ts": now_iso()}]})
                    last_auto_stored_length[0] = len(full_text)
            except asyncio.CancelledError:
                break
            except Exception as e:
                try:
                    await _send(ws, {"type": "error", "message": str(e)})
                except Exception:
                    pass

    def _start_periodic_auto_store() -> None:
        nonlocal periodic_auto_store_task
        if not auto_store or auto_store_interval_sec <= 0:
            return
        if periodic_auto_store_task is not None and not periodic_auto_store_task.done():
            periodic_auto_store_task.cancel()
        periodic_auto_store_task = asyncio.create_task(_periodic_auto_store_fn())

    def _cancel_periodic_auto_store() -> None:
        nonlocal periodic_auto_store_task
        if periodic_auto_store_task is not None and not periodic_auto_store_task.done():
            periodic_auto_store_task.cancel()

    def _ensure_session():
        nonlocal session_id, session, started_at
        if session is None:
            session_id = str(uuid.uuid4())
            session = SessionState(session_id)
            started_at = time.time()
            last_auto_stored_length[0] = 0
            _start_periodic_auto_store()

    async def _maybe_emit_partial(ts_ms: int):
        nonlocal last_emit_time
        if session is None:
            return
        if not session.differs_from_last_emit():
            return
        if time.time() - last_emit_time < EMIT_MIN_INTERVAL:
            return
        last_emit_time = time.time()
        session.mark_emitted()
        segments_payload = [{"paragraph_id": p.paragraph_id, "text": p.raw_text} for p in session.segments]
        await _send(ws, {
            "type": "partial",
            "session_id": session_id,
            "text": session.get_display_text(),
            "segments": segments_payload,
        })

    async def _run_kyutai_frames(pcm_float32: np.ndarray, sr: int):
        """Feed PCM to Kyutai frame-by-frame; emit pieces into session."""
        if kyutai_stt is None:
            return
        if sr != kyutai_stt.sample_rate:
            pcm_float32 = _resample_linear(pcm_float32, sr, kyutai_stt.sample_rate)
        ts_ms = int(time.time() * 1000)
        def run():
            return kyutai_stt.add_audio(pcm_float32)
        pieces = await loop.run_in_executor(None, run)
        for piece in pieces:
            if not piece.strip():
                continue
            _ensure_session()
            session.append_piece(piece, ts_ms)
            if session.maybe_commit(ts_ms):
                new_p = session.maybe_new_paragraph(ts_ms)
                if new_p:
                    await _send(ws, {"type": "segment", "session_id": session_id, "paragraph_id": new_p.paragraph_id, "text": new_p.raw_text})
        if pieces:
            await _maybe_emit_partial(ts_ms)

    async def _run_whisper_chunk():
        nonlocal last_whisper_time, audio_buffer
        if not audio_buffer:
            return
        total = sum(b.shape[0] for b in audio_buffer)
        if total < int(WHISPER_CHUNK_SEC * SAMPLE_RATE_WHISPER):
            return
        audio = np.concatenate(audio_buffer)
        audio_buffer.clear()
        last_whisper_time = time.time()
        ts_ms = int(last_whisper_time * 1000)
        try:
            def run():
                model = _load_whisper()
                return (model.transcribe(audio, fp16=False).get("text", "") or "").strip()
            text = await loop.run_in_executor(None, run)
        except Exception:
            text = ""
        if not text:
            return
        _ensure_session()
        session.append_piece(text, ts_ms)
        if session.maybe_commit(ts_ms):
            new_p = session.maybe_new_paragraph(ts_ms)
            if new_p:
                await _send(ws, {"type": "segment", "session_id": session_id, "paragraph_id": new_p.paragraph_id, "text": new_p.raw_text})
        await _maybe_emit_partial(ts_ms)

    try:
        while True:
            msg = await ws.receive()
            if not isinstance(msg, dict) or msg.get("type") != "websocket.receive":
                if msg.get("type") == "websocket.disconnect":
                    break
                continue
            # Binary: PCM16 audio (client sends at ready.sample_rate: 24kHz for Kyutai, 16kHz for Whisper)
            raw_bytes = msg.get("bytes")
            if raw_bytes and len(raw_bytes) > 0:
                _ensure_session()
                if session and session._paused:
                    continue
                pcm16 = bytes(raw_bytes)
                f32 = _pcm16_to_float32(pcm16)
                sr = client_sample_rate if client_sample_rate is not None else 16000
                if use_kyutai and kyutai_stt is not None:
                    await _run_kyutai_frames(f32, sr)
                else:
                    audio_buffer.append(f32)
                    await _run_whisper_chunk()
                continue
            # Text: JSON or "EOS"
            text_msg = msg.get("text")
            if not text_msg:
                continue
            if text_msg.strip().upper() == "EOS":
                data = {"type": "eos"}
            else:
                try:
                    data = json.loads(text_msg)
                except json.JSONDecodeError:
                    continue
            t = data.get("type")
            # Backward compat: JSON audio chunk (base64) and stop
            if t == "audio":
                b64 = data.get("pcm16_b64")
                if b64:
                    pcm16 = base64.b64decode(b64)
                    _ensure_session()
                    if session and not session._paused:
                        f32 = _pcm16_to_float32(pcm16)
                        sr = client_sample_rate if client_sample_rate is not None else 16000
                        if use_kyutai and kyutai_stt is not None:
                            await _run_kyutai_frames(f32, sr)
                        else:
                            audio_buffer.append(f32)
                            await _run_whisper_chunk()
                continue
            if t == "stop":
                data = {"type": "eos"}
                t = "eos"
            if t == "start":
                session_id = data.get("session_id") or str(uuid.uuid4())
                session = SessionState(session_id)
                started_at = time.time()
                mode = data.get("mode", "transcribe")
                language = data.get("language", "en")
                auto_store = data.get("auto_store", settings.ECHOMIND_AUTO_STORE_DEFAULT)
                client_sample_rate = data.get("sample_rate")
                last_auto_stored_length[0] = 0
                _start_periodic_auto_store()
                if use_kyutai and kyutai_stt is not None:
                    kyutai_stt.reset_streaming()
                await _send(ws, {"type": "ready", "session_id": session_id, "sample_rate": sample_rate})
                continue
            if t == "pause":
                if session:
                    session.pause()
                continue
            if t == "resume":
                if session:
                    session.resume()
                continue
            if t == "eos":
                _cancel_periodic_auto_store()
                _ensure_session()
                if use_kyutai and kyutai_stt is not None:
                    # Flush Kyutai: pad with silence and collect final pieces
                    def run_flush():
                        return kyutai_stt.flush()
                    pieces = await loop.run_in_executor(None, run_flush)
                    ts_ms = int(time.time() * 1000)
                    for piece in pieces:
                        if piece.strip():
                            session.append_piece(piece, ts_ms)
                else:
                    # Flush remaining audio through Whisper
                    if audio_buffer:
                        audio = np.concatenate(audio_buffer)
                        audio_buffer.clear()
                        if audio.size >= int(0.25 * SAMPLE_RATE_WHISPER):
                            try:
                                def run():
                                    model = _load_whisper()
                                    return (model.transcribe(audio, fp16=False).get("text", "") or "").strip()
                                text = await loop.run_in_executor(None, run)
                                if text:
                                    session.append_piece(text, int(time.time() * 1000))
                            except Exception:
                                pass
                session.finalize()
                final_text = session.get_display_text()
                segments_payload = [{"paragraph_id": p.paragraph_id, "text": p.raw_text} for p in session.segments]
                await _send(ws, {
                    "type": "final",
                    "session_id": session_id,
                    "text": final_text,
                    "segments": segments_payload,
                })
                if auto_store and final_text.strip():
                    try:
                        # Store only the remainder since last periodic store to avoid duplicating content
                        to_store = final_text[last_auto_stored_length[0] :].strip() if last_auto_stored_length[0] > 0 else final_text
                        if to_store:
                            conv_type, tags = get_metadata(to_store)
                            meta = {"session_id": session_id, "kind": "raw", "tags": tags, "conversation_type": conv_type, "ts": now_iso()}
                            kid = await kb.kb_add_text(to_store, meta)
                            await _send(ws, {"type": "stored", "session_id": session_id, "items": [{"id": kid, "kind": "raw", "tags": tags, "ts": now_iso()}]})
                    except Exception as e:
                        await _send(ws, {"type": "error", "message": str(e)})
                break
            if t == "refine":
                scope = data.get("scope", "all")
                paragraph_id = data.get("paragraph_id")
                _ensure_session()
                if scope == "all":
                    text_to_refine = session.get_display_text()
                    if not text_to_refine.strip():
                        await _send(ws, {"type": "error", "message": "No transcript to refine"})
                        continue
                    refined = await refine_text(text_to_refine)
                    await _send(ws, {"type": "refined", "session_id": session_id, "scope": scope, "text": refined})
                elif scope == "last_paragraph" and session.segments:
                    p = session.segments[-1]
                    refined = await refine_text(p.raw_text)
                    p.polished_text = refined
                    await _send(ws, {"type": "refined", "session_id": session_id, "scope": scope, "paragraph_id": p.paragraph_id, "text": refined})
                elif scope == "paragraph" and paragraph_id:
                    p = next((x for x in session.segments if x.paragraph_id == paragraph_id), None)
                    if not p:
                        await _send(ws, {"type": "error", "message": f"Paragraph {paragraph_id} not found"})
                        continue
                    refined = await refine_text(p.raw_text)
                    p.polished_text = refined
                    await _send(ws, {"type": "refined", "session_id": session_id, "scope": scope, "paragraph_id": p.paragraph_id, "text": refined})
                else:
                    await _send(ws, {"type": "error", "message": "Invalid refine scope"})
                continue
            if t == "store":
                scope = data.get("scope", "all")
                paragraph_id = data.get("paragraph_id")
                _ensure_session()
                items = []
                if scope == "all":
                    full = session.get_display_text()
                    if full.strip():
                        conv_type, tags = get_metadata(full)
                        meta = {"session_id": session_id, "kind": "raw", "paragraph_id": None, "tags": tags, "conversation_type": conv_type, "ts": now_iso()}
                        kid = await kb.kb_add_text(full, meta)
                        items.append({"id": kid, "kind": "raw", "paragraph_id": None, "tags": tags, "ts": now_iso()})
                        full_refined = await refine_text(full)
                        if full_refined.strip():
                            conv_type2, tags2 = get_metadata(full_refined)
                            meta2 = {"session_id": session_id, "kind": "refined", "paragraph_id": None, "tags": tags2, "conversation_type": conv_type2, "ts": now_iso()}
                            kid2 = await kb.kb_add_text(full_refined, meta2)
                            items.append({"id": kid2, "kind": "refined", "paragraph_id": None, "tags": tags2, "ts": now_iso()})
                    for p in session.segments:
                        if p.polished_text:
                            conv_type, tags = get_metadata(p.polished_text)
                            meta = {"session_id": session_id, "kind": "refined", "paragraph_id": p.paragraph_id, "tags": tags, "conversation_type": conv_type, "ts": now_iso()}
                            kid = await kb.kb_add_text(p.polished_text, meta)
                            items.append({"id": kid, "kind": "refined", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
                        conv_type, tags = get_metadata(p.raw_text)
                        meta = {"session_id": session_id, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "conversation_type": conv_type, "ts": now_iso()}
                        kid = await kb.kb_add_text(p.raw_text, meta)
                        items.append({"id": kid, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
                elif scope == "last_paragraph" and session.segments:
                    p = session.segments[-1]
                    conv_type, tags = get_metadata(p.raw_text)
                    meta = {"session_id": session_id, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "conversation_type": conv_type, "ts": now_iso()}
                    kid = await kb.kb_add_text(p.raw_text, meta)
                    items.append({"id": kid, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
                    if p.polished_text:
                        kid2 = await kb.kb_add_text(p.polished_text, {**meta, "kind": "refined"})
                        items.append({"id": kid2, "kind": "refined", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
                elif scope == "paragraph" and paragraph_id:
                    p = next((x for x in session.segments if x.paragraph_id == paragraph_id), None)
                    if not p:
                        await _send(ws, {"type": "error", "message": f"Paragraph {paragraph_id} not found"})
                        continue
                    conv_type, tags = get_metadata(p.raw_text)
                    meta = {"session_id": session_id, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "conversation_type": conv_type, "ts": now_iso()}
                    kid = await kb.kb_add_text(p.raw_text, meta)
                    items.append({"id": kid, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
                    if p.polished_text:
                        kid2 = await kb.kb_add_text(p.polished_text, {**meta, "kind": "refined"})
                        items.append({"id": kid2, "kind": "refined", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
                await _send(ws, {"type": "stored", "session_id": session_id, "items": items})
    except Exception as e:
        try:
            await _send(ws, {"type": "error", "message": str(e)})
        except Exception:
            pass


async def _send(ws: WebSocket, obj: dict):
    await ws.send_text(json.dumps(obj))
