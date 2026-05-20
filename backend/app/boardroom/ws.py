"""
WebSocket handler for Board Room Mode.

Protocol (client → server):
  binary:  raw PCM16 audio at 16 kHz
  JSON:    {"type": "start", "session_id": "…", "title": "…", "location": "…",
             "sample_rate": 16000, "format": "pcm16", "mode": "boardroom"}
  JSON:    {"type": "stop"} | {"type": "eos"}
  JSON:    {"type": "cancel"}
  JSON:    {"type": "status"}
  JSON:    {"type": "ping"}

Protocol (server → client):
  {"type": "loading"}
  {"type": "session_started",   "session_id": "…", "status": "listening"}
  {"type": "partial",           "session_id": "…", "turns": […], "speaker_count": N,
                                 "duration_seconds": N, "audio_level": 0.0, "chunks_received": N}
  {"type": "listening_status",  "session_id": "…", "duration_seconds": N, "audio_level": N,
                                 "chunks_received": N, "status": "listening"}
  {"type": "finalizing",        "session_id": "…", "status": "transcribing",
                                 "message": "Processing full board room audio"}
  {"type": "transcript_completed", "session_id": "…", "transcript_id": "…",
                                 "speaker_count": N, "segments": […]}
  {"type": "report_completed",  "session_id": "…", "report_id": "…",
                                 "summary": "…", "exports": {"pdf_url": "…", "pptx_url": "…"}}
  {"type": "report_generating", "session_id": "…"}
  {"type": "report_ready",      "session_id": "…", "report_id": "…"}   ← legacy alias
  {"type": "report_error",      "session_id": "…", "message": "…"}
  {"type": "error",             "session_id": "…", "message": "…"}

Audio file saving:
  Every PCM16 binary chunk is also written to:
    data/boardroom/audio/{session_id}.wav
  The WAV file is closed/finalized on stop, cancel, disconnect, or error.
  The audio file path is persisted in boardroom_sessions.audio_file_path.

VibeVoice-ASR cleanup:
  After the primary Parakeet multitalker transcription completes, VibeVoice-ASR
  runs on the saved WAV to produce a cleaned transcript.
  If primary transcription fails, VibeVoice acts as fallback.

Speaker detection:
  Uses spectral log-mel centroid profiling inside ParakeetStreamContext.
  Speaker change triggers RNNT decoder reset for per-speaker accuracy.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import time
import uuid
import wave
from typing import Dict, List, Optional

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from ..core.config import settings
from ..core.db import get_conn
from ..utils.ids import now_iso
from .stt_parakeet import (
    PARAKEET_AVAILABLE,
    PARAKEET_SAMPLE_RATE,
    ParakeetStreamContext,
    SpeakerSegment,
    _parakeet_import_error,
    _pcm16_to_float32,
    _resample,
    _sliding_window_rms,
    get_parakeet_process_lock,
    get_shared_parakeet_adapter,
)
from .report import generate_report_async

logger = logging.getLogger(__name__)

SAMPLE_RATE = PARAKEET_SAMPLE_RATE
PCM_QUEUE_MAX = 512
WS_RECEIVE_TIMEOUT = 86400.0

# How often to push listening_status updates when audio is streaming (seconds)
_STATUS_PUSH_INTERVAL = 10.0


# ── WebSocket helpers ─────────────────────────────────────────────────────────

async def _send(ws: WebSocket, obj: dict) -> None:
    """Send a JSON message; silently ignore errors caused by closed sockets."""
    try:
        await ws.send_text(json.dumps(obj))
    except Exception:
        pass


# ── Audio file I/O ────────────────────────────────────────────────────────────

def _ensure_audio_dir() -> str:
    path = settings.BOARDROOM_AUDIO_DIR
    os.makedirs(path, exist_ok=True)
    return path


class _WavWriter:
    """Write PCM16 mono 16 kHz chunks to a WAV file incrementally."""

    def __init__(self, path: str, sample_rate: int = 16000):
        self.path = path
        self.sample_rate = sample_rate
        self._file: Optional[wave.Wave_write] = None
        self._frames_written = 0
        self._closed = False

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._file = wave.open(self.path, "wb")
        self._file.setnchannels(1)
        self._file.setsampwidth(2)  # PCM16 = 2 bytes per sample
        self._file.setframerate(self.sample_rate)
        self._frames_written = 0
        self._closed = False

    def write_float32(self, audio_f32: np.ndarray, src_sr: int) -> None:
        """Resample to 16 kHz if needed, then write as PCM16."""
        if self._file is None or self._closed:
            return
        if src_sr != self.sample_rate:
            audio_f32 = _resample(audio_f32, src_sr, self.sample_rate)
        pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
        self._file.writeframes(pcm16.tobytes())
        self._frames_written += len(pcm16)

    def duration_sec(self) -> float:
        return self._frames_written / max(1, self.sample_rate)

    def close(self) -> None:
        if self._file and not self._closed:
            try:
                self._file.close()
            except Exception:
                pass
            self._closed = True

    def __del__(self):
        self.close()


# ── DB helpers ────────────────────────────────────────────────────────────────

def _store_session(
    session_id: str,
    title: str,
    location: str,
    status: str,
    started_at: str,
    ended_at: Optional[str],
    duration_sec: Optional[float],
    raw_transcript: str,
    speaker_map: Dict[str, str],
    segments: List[dict],
    audio_file_path: Optional[str] = None,
    speaker_count: int = 0,
    cleaned_transcript: Optional[str] = None,
    primary_model_name: Optional[str] = None,
    diarization_model_name: Optional[str] = None,
    cleanup_model_name: Optional[str] = None,
    transcription_source: str = "boardroom_multitalker",
    rag_ingested: bool = False,
    error_message: Optional[str] = None,
    report_id: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO boardroom_sessions
               (id, title, location, status, started_at, ended_at, duration_sec,
                raw_transcript, speaker_map_json, segments_json, created_at, updated_at,
                audio_file_path, speaker_count, cleaned_transcript,
                primary_model_name, diarization_model_name, cleanup_model_name,
                transcription_source, rag_ingested, error_message, report_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id, title, location, status,
                started_at, ended_at, duration_sec,
                raw_transcript,
                json.dumps(speaker_map),
                json.dumps(segments),
                now_iso(), now_iso(),
                audio_file_path, speaker_count, cleaned_transcript,
                primary_model_name or settings.BOARDROOM_ASR_MODEL_NAME,
                diarization_model_name or settings.BOARDROOM_DIAR_MODEL,
                cleanup_model_name,
                transcription_source, int(rag_ingested),
                error_message, report_id,
            ),
        )
        conn.commit()


def _update_session_field(session_id: str, **kwargs) -> None:
    """Patch specific fields in boardroom_sessions without replacing the full row."""
    if not kwargs:
        return
    cols = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [now_iso(), session_id]
    with get_conn() as conn:
        conn.execute(
            f"UPDATE boardroom_sessions SET {cols}, updated_at=? WHERE id=?", vals
        )
        conn.commit()


# ── RAG ingestion helper ──────────────────────────────────────────────────────

async def _ingest_transcript_to_rag(
    session_id: str,
    transcript_id: str,
    segments: List[dict],
    audio_file_path: Optional[str],
    speaker_map: Dict[str, str],
) -> None:
    """Ingest Board Room speaker-tagged transcript into the knowledge base."""
    try:
        from .. import kb

        logger.info("boardroom.rag.ingest.start session_id=%s transcript_id=%s", session_id, transcript_id)
        t0 = time.monotonic()

        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            speaker_id = seg.get("speaker_id") or ""
            speaker_name = speaker_map.get(speaker_id) or seg.get("speaker_name") or "Speaker 1"
            meta = {
                "source_type": "boardroom_transcript",
                "session_id": session_id,
                "transcript_id": transcript_id,
                "speaker": speaker_name,
                "start_time": seg.get("ts_ms", 0) / 1000.0,
                "end_time": None,
                "audio_file_path": audio_file_path or "",
                "mode": "boardroom",
                "model": settings.BOARDROOM_ASR_MODEL_NAME,
                "type": "transcript",
                "epoch": int(time.time()),
            }
            await kb.kb_add_text(f"{speaker_name}: {text}", meta)

        elapsed = time.monotonic() - t0
        logger.info(
            "boardroom.rag.ingest.completed session_id=%s transcript_id=%s segments=%d elapsed_sec=%.1f",
            session_id, transcript_id, len(segments), elapsed,
        )
        _update_session_field(session_id, rag_ingested=1)
    except Exception as e:
        logger.error("boardroom.rag.ingest.error session_id=%s error=%s", session_id, e)


# ── Main handler ──────────────────────────────────────────────────────────────

async def handler(ws: WebSocket) -> None:
    await ws.accept()
    await _send(ws, {"type": "loading"})

    loop = asyncio.get_running_loop()
    adapter = None
    stream_ctx: Optional[ParakeetStreamContext] = None

    if PARAKEET_AVAILABLE:
        try:
            adapter = await loop.run_in_executor(None, get_shared_parakeet_adapter)
        except Exception as e:
            await _send(ws, {"type": "error", "message": f"Parakeet ASR load failed: {e}"})
            return
    else:
        await _send(ws, {
            "type": "error",
            "message": (
                "Parakeet ASR not available for Board Room mode. "
                "Install NeMo ASR (see backend Dockerfile). "
                f"{_parakeet_import_error or ''}"
            ),
        })
        return

    async with get_parakeet_process_lock():
        stream_ctx = ParakeetStreamContext(adapter)
    asr_lock = asyncio.Lock()

    # Signal to the frontend that the model is loaded and we are ready to
    # receive a "start" command.  The frontend waits for this before sending
    # start + beginning mic capture.
    await _send(ws, {"type": "ready", "sample_rate": SAMPLE_RATE})

    # ── Session state ─────────────────────────────────────────────────────────
    session_id: Optional[str] = None
    session_title: str = "Board Room Session"
    session_location: str = "default"
    started_at_iso: Optional[str] = None
    started_at_mono: Optional[float] = None
    client_sample_rate: Optional[int] = None
    chunks_received: int = 0
    last_audio_level: float = 0.0
    _stopped: bool = False          # idempotency guard for stop/eos

    # WAV writer: initialized on session start
    wav_writer: Optional[_WavWriter] = None
    audio_file_path: Optional[str] = None

    speaker_map: Dict[str, str] = {}
    speaker_counter: List[int] = [0]

    def _get_speaker_name(speaker_id: str) -> str:
        if speaker_id not in speaker_map:
            speaker_counter[0] += 1
            speaker_map[speaker_id] = f"Speaker {speaker_counter[0]}"
        return speaker_map[speaker_id]

    raw_segments: List[dict] = []
    speaker_turns: List[dict] = []

    def _add_to_turns(speaker_id: str, delta_text: str) -> None:
        text = delta_text.strip()
        if not text:
            return
        name = _get_speaker_name(speaker_id)
        if speaker_turns and speaker_turns[-1]["speaker_id"] == speaker_id:
            cur = speaker_turns[-1]["text"]
            speaker_turns[-1]["text"] = (cur + " " + text) if (cur and not cur.endswith(" ")) else (cur + text)
        else:
            speaker_turns.append({
                "turn_id": len(speaker_turns),
                "speaker_id": speaker_id,
                "speaker_name": name,
                "text": text,
            })

    def _turns_to_transcript() -> str:
        return "\n\n".join(
            f"{t['speaker_name']}: {t['text'].strip()}" for t in speaker_turns
        )

    # PCM queue + consumer
    pcm_queue: asyncio.Queue = asyncio.Queue(maxsize=PCM_QUEUE_MAX)
    consumer_task: Optional[asyncio.Task] = None
    use_cuda = adapter.device == "cuda"
    gpu_sem = asyncio.Semaphore(max(1, settings.BOARDROOM_GPU_CONCURRENCY)) if use_cuda else None

    # Periodic listening_status push
    last_status_push: float = 0.0

    async def _process_segments(new_segs: List[SpeakerSegment]) -> None:
        if not new_segs:
            return
        changed = False
        for seg in new_segs:
            text = seg.text.strip()
            if not text:
                continue
            name = _get_speaker_name(seg.speaker_id)
            raw_segments.append({
                "speaker_id": seg.speaker_id,
                "speaker_name": name,
                "text": text,
                "ts_ms": seg.ts_ms,
                "segment_index": len(raw_segments),
            })
            _add_to_turns(seg.speaker_id, text)
            changed = True

        if changed:
            duration = (time.monotonic() - started_at_mono) if started_at_mono else 0.0
            await _send(ws, {
                "type": "partial",
                "session_id": session_id,
                "turns": [
                    {"speaker": t["speaker_name"], "text": t["text"]}
                    for t in speaker_turns[-30:]
                ],
                "speaker_count": len(speaker_map),
                "duration_seconds": round(duration, 1),
                "audio_level": round(last_audio_level, 3),
                "chunks_received": chunks_received,
            })

    async def _run_asr(pcm: np.ndarray, sr: int) -> None:
        if stream_ctx is None:
            return
        if gpu_sem:
            await gpu_sem.acquire()
        try:
            async with get_parakeet_process_lock():
                segs = await stream_ctx.process_pcm_chunk(pcm, sr, loop)
        finally:
            if gpu_sem:
                gpu_sem.release()
        await _process_segments(segs)

    async def _pcm_consumer() -> None:
        while True:
            item = await pcm_queue.get()
            if item is None:
                break
            pcm, sr = item
            try:
                async with asr_lock:
                    await _run_asr(pcm, sr)
            except Exception as e:
                logger.warning("boardroom.audio.chunk_error: %s", e)
            pcm_queue.task_done()

    consumer_task = asyncio.create_task(_pcm_consumer())

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=WS_RECEIVE_TIMEOUT)
            except asyncio.TimeoutError:
                await _send(ws, {"type": "error", "session_id": session_id, "message": "Connection timed out"})
                break
            except WebSocketDisconnect:
                break

            if not isinstance(msg, dict) or msg.get("type") != "websocket.receive":
                if msg.get("type") == "websocket.disconnect":
                    break
                continue

            # ── Binary: raw PCM16 audio ───────────────────────────────────────
            raw_bytes = msg.get("bytes")
            if raw_bytes and len(raw_bytes) > 0:
                f32 = _pcm16_to_float32(bytes(raw_bytes))
                sr = client_sample_rate or SAMPLE_RATE

                # Track audio level (sliding window RMS)
                last_audio_level = float(_sliding_window_rms(f32, 1024, 512)) if len(f32) >= 1024 else float(np.sqrt(np.mean(f32 ** 2) + 1e-12))
                chunks_received += 1

                # Save to WAV file for full-session replay / VibeVoice cleanup
                if wav_writer is not None:
                    try:
                        wav_writer.write_float32(f32, sr)
                    except Exception as wav_err:
                        logger.warning("boardroom.audio.wav_write_error: %s", wav_err)

                logger.debug(
                    "boardroom.audio.chunk_received session_id=%s chunk=%d bytes=%d",
                    session_id, chunks_received, len(raw_bytes),
                )

                try:
                    pcm_queue.put_nowait((f32, sr))
                except asyncio.QueueFull:
                    pass

                # Periodic listening_status push
                now = time.monotonic()
                if now - last_status_push >= _STATUS_PUSH_INTERVAL and session_id:
                    last_status_push = now
                    duration = (now - started_at_mono) if started_at_mono else 0.0
                    await _send(ws, {
                        "type": "listening_status",
                        "session_id": session_id,
                        "duration_seconds": round(duration, 1),
                        "audio_level": round(last_audio_level, 3),
                        "chunks_received": chunks_received,
                        "status": "listening",
                    })
                continue

            # ── Text: JSON control message ────────────────────────────────────
            text_msg = msg.get("text")
            if not text_msg:
                continue
            if text_msg.strip().upper() == "EOS":
                data: dict = {"type": "eos"}
            else:
                try:
                    data = json.loads(text_msg)
                except json.JSONDecodeError:
                    continue

            t = data.get("type")

            if t == "ping":
                continue

            if t == "status":
                duration = (time.monotonic() - started_at_mono) if started_at_mono else 0.0
                await _send(ws, {
                    "type": "listening_status",
                    "session_id": session_id,
                    "duration_seconds": round(duration, 1),
                    "audio_level": round(last_audio_level, 3),
                    "chunks_received": chunks_received,
                    "status": "listening" if not _stopped else "stopped",
                })
                continue

            if t == "cancel":
                _stopped = True
                if wav_writer:
                    wav_writer.close()
                await _send(ws, {"type": "session_started", "session_id": session_id, "status": "cancelled"})
                break

            # ── start ─────────────────────────────────────────────────────────
            if t == "start":
                async with asr_lock:
                    session_id = data.get("session_id") or str(uuid.uuid4())
                    session_title = (data.get("title") or "Board Room Session").strip()
                    session_location = (data.get("location") or "default").strip()
                    started_at_iso = now_iso()
                    started_at_mono = time.monotonic()
                    client_sample_rate = data.get("sample_rate")
                    chunks_received = 0
                    last_audio_level = 0.0
                    _stopped = False
                    raw_segments.clear()
                    speaker_turns.clear()
                    speaker_map.clear()
                    speaker_counter[0] = 0
                    while not pcm_queue.empty():
                        try:
                            pcm_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    async with get_parakeet_process_lock():
                        stream_ctx.reset()

                    # Open WAV writer for this session
                    audio_dir = _ensure_audio_dir()
                    audio_file_path = os.path.join(audio_dir, f"{session_id}.wav")
                    wav_writer = _WavWriter(audio_file_path, SAMPLE_RATE)
                    try:
                        wav_writer.open()
                        logger.info("boardroom.session.start session_id=%s audio_path=%s", session_id, audio_file_path)
                    except Exception as wav_err:
                        logger.error("boardroom.audio.open_error: %s — audio will not be saved", wav_err)
                        wav_writer = None
                        audio_file_path = None

                _store_session(
                    session_id=session_id,
                    title=session_title,
                    location=session_location,
                    status="listening",
                    started_at=started_at_iso,
                    ended_at=None,
                    duration_sec=None,
                    raw_transcript="",
                    speaker_map={},
                    segments=[],
                    audio_file_path=audio_file_path,
                )
                await _send(ws, {
                    "type": "session_started",
                    "session_id": session_id,
                    "status": "listening",
                    "sample_rate": SAMPLE_RATE,
                })
                # Keep backward compat with existing frontend (also sends "ready")
                await _send(ws, {
                    "type": "ready",
                    "session_id": session_id,
                    "sample_rate": SAMPLE_RATE,
                })
                continue

            # ── legacy base64 audio ───────────────────────────────────────────
            if t == "audio":
                b64 = data.get("pcm16_b64")
                if b64:
                    f32 = _pcm16_to_float32(base64.b64decode(b64))
                    sr = client_sample_rate or SAMPLE_RATE
                    if wav_writer:
                        try:
                            wav_writer.write_float32(f32, sr)
                        except Exception:
                            pass
                    try:
                        pcm_queue.put_nowait((f32, sr))
                    except asyncio.QueueFull:
                        pass
                continue

            # ── stop / eos ────────────────────────────────────────────────────
            if t in ("stop", "eos"):
                if _stopped:
                    # Idempotency: stop already processed; ignore duplicate
                    continue
                _stopped = True

                logger.info(
                    "boardroom.session.stop session_id=%s chunks=%d duration_sec=%.1f",
                    session_id, chunks_received,
                    (time.monotonic() - started_at_mono) if started_at_mono else 0,
                )

                # ── 1. Close WAV file ─────────────────────────────────────────
                wav_duration_sec = 0.0
                if wav_writer:
                    wav_duration_sec = wav_writer.duration_sec()
                    wav_writer.close()
                    logger.info(
                        "boardroom.audio.finalized session_id=%s path=%s duration_sec=%.1f",
                        session_id, audio_file_path, wav_duration_sec,
                    )

                # ── 2. Drain streaming queue + reset stream context ────────────
                async with asr_lock:
                    while True:
                        try:
                            pcm_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    async with get_parakeet_process_lock():
                        stream_ctx.reset()

                ended_at_iso = now_iso()
                duration = (time.monotonic() - started_at_mono) if started_at_mono else 0.0

                # ── 3. Notify client: post-processing starting ────────────────
                await _send(ws, {
                    "type": "finalizing",
                    "session_id": session_id,
                    "status": "transcribing",
                    "message": "Running speaker diarization (Sortformer)…",
                })

                # ─────────────────────────────────────────────────────────────
                # PHASE A: Sortformer diarization
                #
                # This is the authoritative source of speaker identity.
                # Spectral profiling (used during live streaming) is kept only
                # for real-time badge display; it is NOT used for the final
                # transcript speaker labels.
                # ─────────────────────────────────────────────────────────────
                diar_result: Optional[dict] = None
                diar_speaker_count = 0
                diar_failed = False

                if audio_file_path and os.path.isfile(audio_file_path):
                    try:
                        from .diarization import diarize_wav_async
                        diar_model = settings.BOARDROOM_DIAR_MODEL
                        diar_result = await diarize_wav_async(
                            wav_path=audio_file_path,
                            model_name=diar_model,
                        )
                        diar_speaker_count = diar_result.get("speaker_count", 0)
                        logger.info(
                            "boardroom.diarization.completed session_id=%s "
                            "speaker_count=%d segment_count=%d elapsed_sec=%.2f",
                            session_id,
                            diar_speaker_count,
                            len(diar_result.get("segments", [])),
                            diar_result.get("elapsed_sec", 0),
                        )
                    except Exception as diar_err:
                        diar_failed = True
                        logger.error(
                            "boardroom.diarization.error session_id=%s error=%s",
                            session_id, diar_err,
                        )
                        await _send(ws, {
                            "type": "finalizing",
                            "session_id": session_id,
                            "status": "diarization_failed",
                            "message": f"Diarization failed: {diar_err}. Using spectral fallback.",
                        })
                else:
                    diar_failed = True
                    logger.warning(
                        "boardroom.diarization.skip session_id=%s reason=no_wav_file",
                        session_id,
                    )

                # ─────────────────────────────────────────────────────────────
                # PHASE B: Multitalker per-speaker transcription
                # ─────────────────────────────────────────────────────────────
                await _send(ws, {
                    "type": "finalizing",
                    "session_id": session_id,
                    "status": "transcribing",
                    "message": "Transcribing each speaker with Parakeet multitalker…",
                })

                multitalker_result: Optional[dict] = None
                final_transcription_source = "multitalker_parakeet"
                fallback_used = False

                if audio_file_path and os.path.isfile(audio_file_path) and not diar_failed:
                    try:
                        from .multitalker_transcription import transcribe_with_diarization_async
                        multitalker_result = await transcribe_with_diarization_async(
                            wav_path=audio_file_path,
                            diarization_result=diar_result or {"speaker_count": 0, "segments": []},
                        )
                        final_transcription_source = multitalker_result.get(
                            "transcription_source", "multitalker_parakeet"
                        )
                        fallback_used = multitalker_result.get("fallback_used", False)
                        if multitalker_result.get("warning"):
                            logger.warning(
                                "boardroom.multitalker.warning session_id=%s msg=%s",
                                session_id, multitalker_result["warning"],
                            )
                    except Exception as mt_err:
                        fallback_used = True
                        logger.error(
                            "boardroom.multitalker.error session_id=%s error=%s",
                            session_id, mt_err,
                        )

                if fallback_used:
                    logger.info("boardroom.fallback.used session_id=%s", session_id)

                # ─────────────────────────────────────────────────────────────
                # Build final segments & speaker map from multitalker output.
                # Fall back to streaming spectral result only if multitalker
                # completely failed.
                # ─────────────────────────────────────────────────────────────
                mt_segments: List[dict] = (
                    multitalker_result.get("segments", []) if multitalker_result else []
                )
                mt_speaker_count: int = (
                    multitalker_result.get("speaker_count", 0) if multitalker_result else 0
                )

                if mt_segments:
                    # Rebuild speaker_map and speaker_turns from multitalker output
                    speaker_map.clear()
                    speaker_turns.clear()
                    raw_segments.clear()
                    speaker_counter[0] = 0

                    for seg in mt_segments:
                        spk_label = seg.get("speaker", "Speaker 1")
                        # speaker_map: internal_id → display name
                        # Use speaker label directly as both key and value
                        speaker_map[spk_label] = spk_label
                        text = (seg.get("text") or "").strip()
                        ts_ms = int(seg.get("start_time", 0) * 1000)
                        if text:
                            raw_segments.append({
                                "speaker_name": spk_label,
                                "text":         text,
                                "ts_ms":        ts_ms,
                                "start_time":   seg.get("start_time", 0.0),
                                "end_time":     seg.get("end_time",   0.0),
                            })
                            speaker_turns.append({
                                "speaker_name": spk_label,
                                "text":         text,
                                "ts_ms":        ts_ms,
                            })

                    final_segments = raw_segments
                    primary_speaker_count = mt_speaker_count
                else:
                    # Multitalker returned nothing → use streaming spectral result as last resort
                    fallback_used = True
                    final_transcription_source = "spectral_fallback"
                    final_segments = raw_segments
                    primary_speaker_count = len(speaker_map)
                    logger.warning(
                        "boardroom.fallback.used session_id=%s "
                        "reason=multitalker_returned_no_segments "
                        "spectral_speaker_count=%d",
                        session_id, primary_speaker_count,
                    )
                    await _send(ws, {
                        "type": "finalizing",
                        "session_id": session_id,
                        "status": "warning",
                        "message": (
                            "Multitalker transcription failed. "
                            "Fallback transcription used without reliable speaker labels."
                        ),
                        "transcription_source": "fallback_single_speaker",
                    })

                raw_transcript = " ".join(
                    seg.get("text", "") for seg in final_segments if seg.get("text")
                )

                # ─────────────────────────────────────────────────────────────
                # PHASE C: Pre-store validation
                # ─────────────────────────────────────────────────────────────
                stored_speaker_count = len({
                    seg.get("speaker_name", "Speaker 1") for seg in final_segments
                })

                if stored_speaker_count == 1 and wav_duration_sec > 30:
                    logger.warning(
                        "boardroom.warning.single_speaker_detected session_id=%s "
                        "duration_sec=%.1f stored_speaker_count=%d diar_speaker_count=%d",
                        session_id, wav_duration_sec, stored_speaker_count, diar_speaker_count,
                    )

                if diar_speaker_count > 1 and stored_speaker_count == 1:
                    err_msg = (
                        f"Data integrity error: diarization detected {diar_speaker_count} speakers "
                        f"but final transcript has only {stored_speaker_count}. "
                        "Multi-speaker diarization result will NOT be overwritten."
                    )
                    logger.error("boardroom.error session_id=%s msg=%s", session_id, err_msg)
                    # Log but do NOT abort — send warning to client and continue
                    await _send(ws, {
                        "type": "finalizing",
                        "session_id": session_id,
                        "status": "speaker_count_mismatch",
                        "message": err_msg,
                    })

                logger.info(
                    "boardroom.transcription.completed session_id=%s "
                    "diar_speaker_count=%d stored_speaker_count=%d "
                    "segments=%d duration_sec=%.1f source=%s fallback=%s",
                    session_id, diar_speaker_count, stored_speaker_count,
                    len(final_segments), duration,
                    final_transcription_source, fallback_used,
                )

                # ── VibeVoice cleanup (optional additional cleanup pass) ───────
                cleaned_transcript: Optional[str] = None
                cleanup_model_name: Optional[str] = None

                if settings.FINAL_CLEANUP_ENABLED and audio_file_path and os.path.isfile(audio_file_path):
                    try:
                        logger.info("boardroom.cleanup.start session_id=%s", session_id)
                        from ..cleanup.stt_vibevoice import transcribe_wav_file_async
                        vv_result = await transcribe_wav_file_async(
                            audio_file_path, source="vibevoice_cleanup"
                        )
                        cleaned_transcript = vv_result.get("text") or ""
                        cleanup_model_name = vv_result.get("model")
                        logger.info(
                            "boardroom.cleanup.completed session_id=%s elapsed_sec=%.1f",
                            session_id, vv_result.get("elapsed_sec", 0),
                        )
                    except Exception as vv_err:
                        logger.warning(
                            "boardroom.cleanup.error session_id=%s error=%s", session_id, vv_err
                        )

                # ── Store in DB ────────────────────────────────────────────────
                _store_session(
                    session_id=session_id or str(uuid.uuid4()),
                    title=session_title,
                    location=session_location,
                    status="completed",
                    started_at=started_at_iso or ended_at_iso,
                    ended_at=ended_at_iso,
                    duration_sec=duration,
                    raw_transcript=raw_transcript,
                    speaker_map=speaker_map,
                    segments=final_segments,
                    audio_file_path=audio_file_path,
                    speaker_count=stored_speaker_count,
                    cleaned_transcript=cleaned_transcript,
                    cleanup_model_name=cleanup_model_name,
                    transcription_source=final_transcription_source,
                    diarization_model_name=settings.BOARDROOM_DIAR_MODEL if not diar_failed else None,
                )

                # ── RAG ingestion ──────────────────────────────────────────────
                transcript_id = session_id
                if final_segments:
                    await _send(ws, {
                        "type": "finalizing",
                        "session_id": session_id,
                        "status": "rag_ingesting",
                        "message": "Ingesting transcript into knowledge base",
                    })
                    await _ingest_transcript_to_rag(
                        session_id=session_id,
                        transcript_id=transcript_id,
                        segments=final_segments,
                        audio_file_path=audio_file_path,
                        speaker_map=speaker_map,
                    )

                # ── Send transcript_completed (spec-compliant) ─────────────────
                await _send(ws, {
                    "type": "transcript_completed",
                    "session_id": session_id,
                    "transcript_id": transcript_id,
                    "speaker_count": stored_speaker_count,
                    "segments": [
                        {
                            "speaker":     seg.get("speaker_name") or "Speaker 1",
                            "start_time":  round(seg.get("start_time", seg.get("ts_ms", 0) / 1000.0), 2),
                            "end_time":    seg.get("end_time"),
                            "text":        seg.get("text") or "",
                        }
                        for seg in final_segments
                    ],
                    "duration_seconds":       round(duration, 1),
                    "transcription_source":   final_transcription_source,
                    "diarization_speaker_count": diar_speaker_count,
                    "fallback_used":          fallback_used,
                })

                # Legacy "final" message for existing frontend
                await _send(ws, {
                    "type": "final",
                    "session_id": session_id,
                    "transcript": raw_transcript,
                    "turns": [
                        {"speaker": t["speaker_name"], "text": t["text"]}
                        for t in speaker_turns
                    ],
                    "speaker_map": speaker_map,
                    "duration_sec": duration,
                })

                # ── Report generation ─────────────────────────────────────────
                await _generate_report_task(
                    session_id=session_id or "",
                    raw_transcript=raw_transcript,
                    segments=final_segments,
                    speaker_map=speaker_map,
                    ws=ws,
                )
                break

    except Exception as e:
        logger.error("boardroom.error session_id=%s error=%s", session_id, e)
        try:
            await _send(ws, {"type": "error", "session_id": session_id, "message": str(e)})
        except Exception:
            pass
    finally:
        # Always close WAV writer on disconnect/error
        if wav_writer:
            wav_writer.close()
        if consumer_task and not consumer_task.done():
            try:
                pcm_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            try:
                await asyncio.wait_for(consumer_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                consumer_task.cancel()


# ── Report generation task ────────────────────────────────────────────────────

async def _generate_report_task(
    session_id: str,
    raw_transcript: str,
    segments: List[dict],
    speaker_map: Dict[str, str],
    ws: WebSocket,
) -> None:
    """Generate RAG-enhanced report and deliver result through open WebSocket."""
    try:
        logger.info("boardroom.report.start session_id=%s", session_id)
        t0 = time.monotonic()
        await _send(ws, {
            "type": "finalizing",
            "session_id": session_id,
            "status": "analyzing",
            "message": "Generating meeting report with TensorRT-LLM + RAG",
        })
        await _send(ws, {"type": "report_generating", "session_id": session_id})

        report_id = await generate_report_async(
            session_id=session_id,
            transcript=raw_transcript,
            segments=segments,
            speaker_map=speaker_map,
        )
        elapsed = time.monotonic() - t0
        logger.info("boardroom.report.completed session_id=%s report_id=%s elapsed_sec=%.1f", session_id, report_id, elapsed)

        # Update session with report_id
        _update_session_field(session_id, report_id=report_id)

        # Spec-compliant report_completed message
        pdf_url = f"/api/boardroom/sessions/{session_id}/export?format=pdf"
        pptx_url = f"/api/boardroom/sessions/{session_id}/export?format=pptx"
        await _send(ws, {
            "type": "report_completed",
            "session_id": session_id,
            "report_id": report_id,
            "exports": {
                "pdf_url": pdf_url if settings.REPORT_EXPORT_PDF else None,
                "pptx_url": pptx_url if settings.REPORT_EXPORT_PPTX else None,
            },
        })
        # Legacy alias kept for existing frontend
        await _send(ws, {
            "type": "report_ready",
            "session_id": session_id,
            "report_id": report_id,
        })
    except Exception as e:
        logger.error("boardroom.error session_id=%s stage=report error=%s", session_id, e)
        try:
            await _send(ws, {
                "type": "report_error",
                "session_id": session_id,
                "message": str(e),
            })
        except Exception:
            pass
