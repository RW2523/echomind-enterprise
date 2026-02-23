"""
WebSocket handler for real-time transcription & knowledge capture.
Protocol: binary PCM16 chunks, text JSON (start/pause/resume/eos/refine/store).
Uses SessionState for stabilization and segmentation; refine and store for KB.
Kyutai STT only (24kHz). Per-session STT instance; PCM queue backpressure; GPU semaphore; audio-relative timestamps.
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
from .session_state import SessionState
from .stt_streaming import (
    _pcm16_to_float32,
    _sliding_window_rms,
    get_or_create_kyutai_stt,
    release_kyutai_stt,
    KYUTAI_AVAILABLE,
    _resample_fast,
    KYUTAI_SAMPLE_RATE,
)
from ..refine import refine_text
from ..tagging import get_metadata
from .. import kb
from .store_to_db import store_transcript_to_db, create_transcript_for_session, append_transcript_chunk, update_transcript_tags_and_echotag

# Kyutai sample rate (24kHz)
SAMPLE_RATE = KYUTAI_SAMPLE_RATE
# Rate limit partials to client
EMIT_MIN_INTERVAL = 1.0 / max(0.1, getattr(settings, "TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC", 15))
# VAD: skip feeding audio when sliding-window max RMS below this (0 = disabled)
VAD_RMS_THRESHOLD = max(0.0, getattr(settings, "TRANSCRIPT_VAD_RMS_THRESHOLD", 0.008))
VAD_WINDOW_SAMPLES = max(1, getattr(settings, "TRANSCRIPT_VAD_WINDOW_SAMPLES", 1024))
VAD_STEP_SAMPLES = max(1, getattr(settings, "TRANSCRIPT_VAD_STEP_SAMPLES", 512))
PCM_QUEUE_MAX = getattr(settings, "TRANSCRIPT_PCM_QUEUE_MAX_SIZE", 256)
INTERVAL_BUFFER_MAX = max(1, getattr(settings, "TRANSCRIPT_INTERVAL_BUFFER_MAX", 2048))

# GPU concurrency limiter (lazy init when first CUDA STT is used)
_gpu_sem: Optional[asyncio.Semaphore] = None


def _get_gpu_sem() -> asyncio.Semaphore:
    global _gpu_sem
    if _gpu_sem is None:
        _gpu_sem = asyncio.Semaphore(max(1, getattr(settings, "TRANSCRIPT_GPU_CONCURRENCY", 2)))
    return _gpu_sem


def _is_valid_english_piece(piece: str) -> bool:
    """Filter out noise and non-English; allow digits so spoken numbers appear in transcript."""
    if not piece or not piece.strip():
        return False
    s = piece.strip()
    alnum = "".join(c for c in s if c.isalnum())
    if not alnum:
        return False
    # Allow purely numeric pieces (e.g. "42", "1,000", "3.14") so numbers are not dropped
    if alnum.isdigit():
        return True
    # Reject single character unless it's a valid word
    if len(s) == 1 and s.upper() not in ("I", "A"):
        return False
    # Reject if majority of letters are non-ASCII (non-English)
    letters = [c for c in s if c.isalpha()]
    if letters:
        non_ascii = sum(1 for c in letters if ord(c) > 127)
        if non_ascii / len(letters) > 0.4:
            return False
    return True


async def handler(ws: WebSocket):
    await ws.accept()
    await _send(ws, {"type": "loading"})

    loop = asyncio.get_running_loop()
    kyutai_stt = None
    if KYUTAI_AVAILABLE:
        try:
            kyutai_stt = await loop.run_in_executor(None, get_or_create_kyutai_stt)
        except Exception as e:
            await _send(ws, {"type": "error", "message": f"Kyutai STT load failed: {e}"})
            return

    if kyutai_stt is None:
        from .stt_streaming import _kyutai_import_error
        hint = f" ({_kyutai_import_error})" if _kyutai_import_error else ""
        await _send(ws, {
            "type": "error",
            "message": f"Kyutai STT not available. Install: pip install moshi huggingface-hub. Live transcript requires Kyutai STT.{hint}"
        })
        return

    sample_rate = kyutai_stt.sample_rate
    await _send(ws, {"type": "ready", "sample_rate": sample_rate})

    session_id: Optional[str] = None
    session: Optional[SessionState] = None
    mode = "transcribe"
    language = "en"
    auto_store = settings.ECHOMIND_AUTO_STORE_DEFAULT
    started_at: Optional[float] = None
    started_at_iso: list = [None]  # mutable: ISO string for transcript echodate
    session_name: list = [""]  # mutable: from start payload
    session_location: list = [""]  # mutable: from start payload
    transcript_id_ref: list = [None]  # mutable: transcript id once created (groups 1-min chunks)
    last_emit_time = 0.0
    client_sample_rate: Optional[int] = None
    last_auto_stored_length: list = [0]  # mutable for closure
    interval_buffer: list = []  # (text, ts) for each interval; cap INTERVAL_BUFFER_MAX
    periodic_auto_store_task: Optional[asyncio.Task] = None
    auto_store_interval_sec = max(0, getattr(settings, "AUTO_STORE_INTERVAL_SEC", 60))
    pcm_queue: asyncio.Queue = asyncio.Queue(maxsize=PCM_QUEUE_MAX if PCM_QUEUE_MAX > 0 else 0)
    consumer_task: Optional[asyncio.Task] = None
    use_cuda = kyutai_stt.device == "cuda"

    async def _periodic_auto_store_fn() -> None:
        """Every auto_store_interval_sec: create or append transcript row, then add chunk to RAG with name/location/time meta. CPU-heavy work off event loop."""
        while True:
            await asyncio.sleep(auto_store_interval_sec)
            if session is None:
                break
            try:
                full_text = session.get_display_text()
                to_store = full_text[last_auto_stored_length[0] :].strip()
                if not to_store:
                    continue
                name = (session_name[0] or "").strip() or None
                location = (session_location[0] or "").strip() or "default"
                echodate_iso = started_at_iso[0] or now_iso()
                tid = transcript_id_ref[0]

                def _do_create_append():
                    nonlocal tid
                    if tid is None:
                        tid = create_transcript_for_session(
                            name=name,
                            location=location,
                            started_at_iso=echodate_iso,
                            initial_text=to_store,
                        )
                    else:
                        append_transcript_chunk(tid, to_store)
                    conv_type, tags = get_metadata(full_text)
                    echotag_val = ",".join(tags) if tags else (name or "transcript")
                    update_transcript_tags_and_echotag(tid, tags, echotag_val)
                    return (tid, conv_type, tags)

                tid, conv_type, tags = await loop.run_in_executor(None, _do_create_append)
                transcript_id_ref[0] = tid
                meta = {
                    "session_id": session_id,
                    "kind": "raw",
                    "tags": tags,
                    "conversation_type": conv_type,
                    "ts": now_iso(),
                    "type": "transcript",
                    "transcript_id": tid,
                    "name": name,
                    "location": location,
                    "echodate": echodate_iso,
                }
                kid = await kb.kb_add_text(to_store, meta)
                interval_buffer.append((to_store, now_iso()))
                if len(interval_buffer) > INTERVAL_BUFFER_MAX:
                    interval_buffer.pop(0)
                await _send(ws, {"type": "stored", "session_id": session_id, "transcript_id": tid, "items": [{"id": kid, "kind": "raw", "tags": tags, "ts": now_iso()}]})
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
            interval_buffer.clear()
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
        """Feed PCM to Kyutai; use sliding-window VAD, audio-relative ts_ms, GPU semaphore if CUDA."""
        if kyutai_stt is None:
            return
        if VAD_RMS_THRESHOLD > 0 and _sliding_window_rms(pcm_float32, VAD_WINDOW_SAMPLES, VAD_STEP_SAMPLES) < VAD_RMS_THRESHOLD:
            return
        if sr != kyutai_stt.sample_rate:
            pcm_float32 = _resample_fast(pcm_float32, sr, kyutai_stt.sample_rate, kyutai_stt.device)
        if use_cuda:
            await _get_gpu_sem().acquire()
        try:
            pieces, ts_ms = await loop.run_in_executor(None, lambda: kyutai_stt.add_audio(pcm_float32))
        finally:
            if use_cuda:
                _get_gpu_sem().release()
        for piece in pieces:
            if not piece.strip() or not _is_valid_english_piece(piece):
                continue
            _ensure_session()
            session.append_piece(piece, ts_ms)
            if session.maybe_commit(ts_ms):
                new_p = session.maybe_new_paragraph(ts_ms)
                if new_p:
                    await _send(ws, {"type": "segment", "session_id": session_id, "paragraph_id": new_p.paragraph_id, "text": new_p.raw_text})
        if pieces:
            await _maybe_emit_partial(ts_ms)

    async def _pcm_consumer() -> None:
        """Consume PCM from queue; backpressure via queue max size."""
        while True:
            item = await pcm_queue.get()
            if item is None:
                break
            pcm_float32, sr = item
            try:
                await _run_kyutai_frames(pcm_float32, sr)
            except Exception:
                pass
            pcm_queue.task_done()

    consumer_task = asyncio.create_task(_pcm_consumer())
    try:
        while True:
            msg = await ws.receive()
            if not isinstance(msg, dict) or msg.get("type") != "websocket.receive":
                if msg.get("type") == "websocket.disconnect":
                    break
                continue
            # Binary: PCM16 audio (client sends at ready.sample_rate: 24kHz for Kyutai)
            raw_bytes = msg.get("bytes")
            if raw_bytes and len(raw_bytes) > 0:
                if session and session._paused:
                    continue
                pcm16 = bytes(raw_bytes)
                f32 = _pcm16_to_float32(pcm16)
                sr = client_sample_rate if client_sample_rate is not None else sample_rate
                try:
                    pcm_queue.put_nowait((f32, sr))
                except asyncio.QueueFull:
                    pass  # drop chunk on backpressure
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
                if b64 and (session is None or not session._paused):
                    pcm16 = base64.b64decode(b64)
                    f32 = _pcm16_to_float32(pcm16)
                    sr = client_sample_rate if client_sample_rate is not None else sample_rate
                    try:
                        pcm_queue.put_nowait((f32, sr))
                    except asyncio.QueueFull:
                        pass
                continue
            if t == "stop":
                data = {"type": "eos"}
                t = "eos"
            if t == "start":
                session_id = data.get("session_id") or str(uuid.uuid4())
                session = SessionState(session_id)
                started_at = time.time()
                started_at_iso[0] = now_iso()
                session_name[0] = (data.get("name") or "").strip() or ""
                session_location[0] = (data.get("location") or "").strip() or "default"
                transcript_id_ref[0] = None
                mode = data.get("mode", "transcribe")
                language = data.get("language", "en")
                auto_store = data.get("auto_store", settings.ECHOMIND_AUTO_STORE_DEFAULT)
                client_sample_rate = data.get("sample_rate")
                last_auto_stored_length[0] = 0
                interval_buffer.clear()
                while not pcm_queue.empty():
                    try:
                        pcm_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                _start_periodic_auto_store()
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
                ts_flush_ms = int((kyutai_stt.audio_offset_samples / kyutai_stt.sample_rate) * 1000)
                if use_cuda:
                    await _get_gpu_sem().acquire()
                try:
                    pieces = await loop.run_in_executor(None, kyutai_stt.flush)
                finally:
                    if use_cuda:
                        _get_gpu_sem().release()
                for piece in pieces:
                    if piece.strip() and _is_valid_english_piece(piece):
                        session.append_piece(piece, ts_flush_ms)
                kyutai_stt.reset_streaming()
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
                        to_store = final_text[last_auto_stored_length[0] :].strip() if last_auto_stored_length[0] > 0 else final_text
                        if to_store:
                            name = (session_name[0] or "").strip() or None
                            location = (session_location[0] or "").strip() or "default"
                            echodate_iso = started_at_iso[0] or now_iso()

                            def _eos_store():
                                tid = transcript_id_ref[0]
                                if tid is None:
                                    tid = create_transcript_for_session(
                                        name=name,
                                        location=location,
                                        started_at_iso=echodate_iso,
                                        initial_text=to_store,
                                    )
                                else:
                                    append_transcript_chunk(tid, to_store)
                                conv_type, tags = get_metadata(final_text)
                                echotag_val = ",".join(tags) if tags else (name or "transcript")
                                update_transcript_tags_and_echotag(tid, tags, echotag_val)
                                return (tid, conv_type, tags)

                            tid, conv_type, tags = await loop.run_in_executor(None, _eos_store)
                            transcript_id_ref[0] = tid
                            meta = {
                                "session_id": session_id,
                                "kind": "raw",
                                "tags": tags,
                                "conversation_type": conv_type,
                                "ts": now_iso(),
                                "type": "transcript",
                                "transcript_id": tid,
                                "name": name,
                                "location": location,
                                "echodate": echodate_iso,
                            }
                            kid = await kb.kb_add_text(to_store, meta)
                            interval_buffer.append((to_store, now_iso()))
                            if len(interval_buffer) > INTERVAL_BUFFER_MAX:
                                interval_buffer.pop(0)
                            await _send(ws, {"type": "stored", "session_id": session_id, "transcript_id": tid, "items": [{"id": kid, "kind": "raw", "tags": tags, "ts": now_iso()}]})
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
            if t == "store_combined":
                _ensure_session()
                parts = [text for (text, _) in interval_buffer]
                remainder = session.get_display_text()[last_auto_stored_length[0] :].strip()
                if remainder:
                    parts.append(remainder)
                combined_raw = "\n\n".join(p for p in parts if p).strip()
                if not combined_raw:
                    await _send(ws, {"type": "error", "message": "No transcript to combine"})
                    continue
                try:
                    refined = await refine_text(combined_raw)
                    result = await store_transcript_to_db(raw_text=combined_raw, refined_text=refined or None)
                    interval_buffer.clear()
                    last_auto_stored_length[0] = len(session.get_display_text())
                    await _send(ws, {
                        "type": "stored_combined",
                        "session_id": session_id,
                        "transcript_id": result["transcript_id"],
                        "tags": result["tags"],
                        "echotag": result["echotag"],
                        "echodate": result["echodate"],
                        "created_at": result["created_at"],
                    })
                except Exception as e:
                    await _send(ws, {"type": "error", "message": str(e)})
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
    finally:
        if kyutai_stt is not None:
            await loop.run_in_executor(None, lambda: release_kyutai_stt(kyutai_stt))
        if consumer_task is not None and not consumer_task.done():
            try:
                pcm_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            try:
                await asyncio.wait_for(consumer_task, timeout=5.0)
            except asyncio.TimeoutError:
                consumer_task.cancel()
                try:
                    await consumer_task
                except asyncio.CancelledError:
                    pass


async def _send(ws: WebSocket, obj: dict):
    await ws.send_text(json.dumps(obj))
