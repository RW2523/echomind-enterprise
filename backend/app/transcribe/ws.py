"""
Real-time transcription WebSocket: stabilization, segments, polish, store.
Backward compatible: binary-like PCM via JSON { type: "audio", pcm16_b64 }, text "EOS" or JSON.
"""
from __future__ import annotations
import asyncio
import base64
import time
import uuid
import numpy as np
from fastapi import WebSocket

from ..core.config import settings
from ..utils.ids import now_iso
from .session_state import SessionState
from ..kb import get_kb
from ..polish import get_polisher
from ..tagging import get_tagger

# -----------------------------------------------------------------------------
# STT: Whisper (no Moshi in this backend; can be swapped for streaming STT later)
# -----------------------------------------------------------------------------
_whisper_model = None
SAMPLES_PER_SEC = 16000


def _samples(sec: float) -> int:
    return int(SAMPLES_PER_SEC * sec)


def pcm16_b64_to_float32(b64: str) -> np.ndarray:
    pcm = base64.b64decode(b64)
    return (np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)


def _transcribe_sync(model, audio: np.ndarray) -> str:
    if audio.size < _samples(0.25):
        return ""
    return (model.transcribe(audio, fp16=False).get("text", "") or "").strip()


async def _ensure_stt_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            _whisper_model = whisper.load_model(settings.WHISPER_MODEL)
        except Exception as e:
            _whisper_model = False
            raise RuntimeError(f"STT model unavailable: {e}") from e
    return _whisper_model


# -----------------------------------------------------------------------------
# WebSocket handler
# -----------------------------------------------------------------------------
async def handler(ws: WebSocket):
    await ws.accept()

    # Send loading then ready (sample_rate for client)
    try:
        await ws.send_json({"type": "loading"})
    except Exception:
        pass
    try:
        model = await _ensure_stt_model()
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        return
    await ws.send_json({"type": "ready", "sample_rate": SAMPLES_PER_SEC, "model": getattr(settings, "WHISPER_MODEL", "base")})

    session_id: str = f"sess_{uuid.uuid4().hex[:12]}"
    session: SessionState | None = None
    auto_store: bool = settings.ECHOMIND_AUTO_STORE_DEFAULT
    last_partial_emit_ts = 0.0
    min_partial_interval = 1.0 / max(0.1, settings.TRANSCRIPT_PARTIAL_RATE_LIMIT_PER_SEC)

    # Audio buffer for Whisper (same as before)
    audio_buffer = np.zeros(0, dtype=np.float32)
    committed_samples = 0
    last_stt_time = time.time()
    loop = asyncio.get_running_loop()

    def _get_or_create_session() -> SessionState:
        nonlocal session
        if session is None:
            session = SessionState(session_id=session_id, paragraph_id_prefix="p")
        return session

    async def _emit_partial():
        nonlocal last_partial_emit_ts
        if session is None:
            return
        now = time.time()
        if not session.should_emit_partial(now, last_partial_emit_ts + min_partial_interval):
            return
        text = session.get_display_text()
        try:
            await ws.send_json({
                "type": "partial",
                "session_id": session_id,
                "text": text,
                "segments": session.get_segments_for_emit(),
            })
            session.mark_partial_emitted()
            last_partial_emit_ts = now
        except Exception:
            pass

    async def _run_stt_and_feed_session():
        """Run Whisper on recent audio and feed text into session as one piece."""
        nonlocal last_stt_time, audio_buffer, committed_samples
        total = audio_buffer.size
        if time.time() - last_stt_time < 0.9 or total < _samples(1):
            return
        last_stt_time = time.time()
        partial_audio = audio_buffer[-_samples(8):] if total >= _samples(1) else audio_buffer
        if partial_audio.size < _samples(0.25):
            return
        try:
            txt = await loop.run_in_executor(None, lambda a=partial_audio: _transcribe_sync(model, a))
        except Exception:
            return
        if not txt or not txt.strip():
            return
        sess = _get_or_create_session()
        if sess.is_paused():
            return
        ts_ms = int(time.time() * 1000)
        sess.append_piece(txt, ts_ms)
        sess.maybe_commit(ts_ms)
        para = sess.maybe_new_paragraph(ts_ms)
        if para:
            try:
                await ws.send_json({
                    "type": "segment",
                    "session_id": session_id,
                    "paragraph_id": para.paragraph_id,
                    "text": para.raw_text,
                })
            except Exception:
                pass
        await _emit_partial()

    try:
        while True:
            # Receive: JSON (audio, start, pause, resume, eos, polish, store) or could be bytes in future
            msg = await ws.receive()
            text_payload = msg.get("text")
            bytes_payload = msg.get("bytes")

            if text_payload is not None:
                # Raw "EOS" (backward compat)
                if text_payload.strip().upper() == "EOS":
                    break
                try:
                    data = __import__("json").loads(text_payload)
                except Exception:
                    continue
                t = data.get("type")
                if t == "start":
                    sid = (data.get("session_id") or "").strip()
                    if sid:
                        session_id = sid
                    auto_store = data.get("auto_store", auto_store)
                    _get_or_create_session()
                    try:
                        await ws.send_json({"type": "session_started", "session_id": session_id})
                    except Exception:
                        pass
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
                    break
                if t == "polish":
                    scope = data.get("scope", "all")
                    paragraph_id = data.get("paragraph_id")
                    sess = _get_or_create_session()
                    polisher = get_polisher()
                    if scope == "all":
                        text_to_polish = sess.get_display_text()
                        polished = await polisher.polish_text(text_to_polish)
                        try:
                            await ws.send_json({
                                "type": "polished",
                                "session_id": session_id,
                                "scope": scope,
                                "text": polished,
                            })
                        except Exception:
                            pass
                    elif scope == "last_paragraph":
                        p = sess.get_last_paragraph()
                        if p:
                            polished = await polisher.polish_text(p.raw_text)
                            p.polished_text = polished
                            try:
                                await ws.send_json({
                                    "type": "polished",
                                    "session_id": session_id,
                                    "scope": scope,
                                    "paragraph_id": p.paragraph_id,
                                    "text": polished,
                                })
                            except Exception:
                                pass
                    elif scope == "paragraph" and paragraph_id:
                        p = sess.get_paragraph_by_id(paragraph_id)
                        if p:
                            polished = await polisher.polish_text(p.raw_text)
                            p.polished_text = polished
                            try:
                                await ws.send_json({
                                    "type": "polished",
                                    "session_id": session_id,
                                    "scope": scope,
                                    "paragraph_id": p.paragraph_id,
                                    "text": polished,
                                })
                            except Exception:
                                pass
                    continue
                if t == "store":
                    scope = data.get("scope", "all")
                    paragraph_id = data.get("paragraph_id")
                    sess = _get_or_create_session()
                    kb = get_kb()
                    tagger = get_tagger()
                    items = []
                    if scope == "all":
                        full = sess.get_display_text()
                        if full.strip():
                            ctype, tags = tagger.tag(full)
                            try:
                                raw_id = await kb.add_text(
                                    full,
                                    {"kind": "raw", "session_id": session_id, "conversation_type": ctype, "tags": tags},
                                    "transcript_raw",
                                )
                                items.append({"id": raw_id, "kind": "raw", "paragraph_id": None, "tags": tags, "ts": now_iso()})
                            except Exception as e:
                                await ws.send_json({"type": "error", "message": f"Store failed: {e}"})
                    else:
                        segments_to_store = []
                        if scope == "last_paragraph":
                            p = sess.get_last_paragraph()
                            if p:
                                segments_to_store.append(p)
                        elif scope == "paragraph" and paragraph_id:
                            p = sess.get_paragraph_by_id(paragraph_id)
                            if p:
                                segments_to_store.append(p)
                        else:
                            segments_to_store = list(sess.segments)
                        for p in segments_to_store:
                            ctype, tags = tagger.tag(p.raw_text)
                            meta = {"kind": "raw", "paragraph_id": p.paragraph_id, "session_id": session_id, "conversation_type": ctype, "tags": tags}
                            try:
                                rid = await kb.add_text(p.raw_text, meta, "transcript_raw")
                                items.append({"id": rid, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
                            except Exception:
                                pass
                            if p.polished_text:
                                meta_p = {"kind": "polished", "paragraph_id": p.paragraph_id, "session_id": session_id, "conversation_type": ctype, "tags": tags}
                                try:
                                    pid = await kb.add_text(p.polished_text, meta_p, "transcript_polished")
                                    items.append({"id": pid, "kind": "polished", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
                                except Exception:
                                    pass
                    try:
                        await ws.send_json({"type": "stored", "session_id": session_id, "items": items})
                    except Exception:
                        pass
                    continue
                if t == "audio":
                    # JSON audio (base64 PCM16): append buffer; commit 10s chunks into session; live partials every 0.9s
                    b64 = data.get("pcm16_b64", "")
                    if b64:
                        x = pcm16_b64_to_float32(b64)
                        audio_buffer = np.concatenate([audio_buffer, x]) if audio_buffer.size else x
                        total = audio_buffer.size
                        segment_samples = _samples(10)
                        uncommitted = total - committed_samples
                        while uncommitted >= segment_samples:
                            chunk = audio_buffer[committed_samples : committed_samples + segment_samples]
                            try:
                                txt = await loop.run_in_executor(None, lambda a=chunk: _transcribe_sync(model, a))
                            except Exception:
                                pass
                            else:
                                if txt:
                                    sess = _get_or_create_session()
                                    if not sess.is_paused():
                                        ts_ms = int(time.time() * 1000)
                                        sess.append_piece(txt, ts_ms)
                                        sess.maybe_commit(ts_ms)
                                        para = sess.maybe_new_paragraph(ts_ms)
                                        if para:
                                            try:
                                                await ws.send_json({
                                                    "type": "segment",
                                                    "session_id": session_id,
                                                    "paragraph_id": para.paragraph_id,
                                                    "text": para.raw_text,
                                                })
                                            except Exception:
                                                pass
                            committed_samples += segment_samples
                            uncommitted = total - committed_samples
                        keep_from = max(0, committed_samples - _samples(8))
                        if keep_from > 0:
                            audio_buffer = audio_buffer[keep_from:]
                            committed_samples -= keep_from
                        await _run_stt_and_feed_session()
                elif t == "stop":
                    break
            if bytes_payload is not None:
                # Raw binary PCM16 (future use)
                x = np.frombuffer(bytes_payload, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, x]) if audio_buffer.size else x
                await _run_stt_and_feed_session()
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        return

    # EOS: flush STT remainder, finalize session, send final, optionally auto_store
    sess = _get_or_create_session()
    remaining = audio_buffer[committed_samples:] if committed_samples < audio_buffer.size else audio_buffer
    if remaining.size >= _samples(0.25):
        try:
            txt = await loop.run_in_executor(None, lambda a=remaining: _transcribe_sync(model, a))
            if txt and sess and not sess.is_paused():
                ts_ms = int(time.time() * 1000)
                sess.append_piece(txt, ts_ms)
        except Exception:
            pass
    sess.finalize()
    final_text = sess.get_display_text()
    try:
        await ws.send_json({
            "type": "final",
            "session_id": session_id,
            "text": final_text,
            "segments": sess.get_segments_for_emit(),
        })
    except Exception:
        pass

    if auto_store and final_text.strip():
        kb = get_kb()
        tagger = get_tagger()
        items = []
        try:
            ctype, tags = tagger.tag(final_text)
            raw_id = await kb.add_text(
                final_text,
                {"kind": "raw", "session_id": session_id, "conversation_type": ctype, "tags": tags},
                "transcript_raw",
            )
            items.append({"id": raw_id, "kind": "raw", "paragraph_id": None, "tags": tags, "ts": now_iso()})
            await ws.send_json({"type": "stored", "session_id": session_id, "items": items})
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "message": f"Auto-store failed: {e}"})
            except Exception:
                pass
