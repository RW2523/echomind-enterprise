"""
WebSocket handler for real-time transcription & knowledge capture.
Protocol: binary PCM16 chunks, text JSON (start/pause/resume/eos/refine/store).
Uses SessionState for stabilization and segmentation; refine and store for KB.
Nemotron NeMo streaming ASR (16 kHz). Shared model, per-connection stream state; PCM queue; GPU semaphore.
"""
from __future__ import annotations
import asyncio
import base64
import json
import logging
import os
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

logger = logging.getLogger(__name__)

from ..core.config import settings
from ..core.db import get_conn
from ..utils.ids import now_iso
from .session_state import SessionState
from .stt_streaming import (
    NEMOTRON_AVAILABLE,
    NEMOTRON_SAMPLE_RATE,
    NemotronStreamContext,
    _nemotron_import_error,
    _pcm16_to_float32,
    _sliding_window_rms,
    get_nemotron_process_lock,
    get_shared_asr_adapter,
)
from ..refine import refine_text
from ..tagging import get_metadata
from .. import kb
from .store_to_db import (
    store_transcript_to_db,
    create_transcript_for_session,
    append_transcript_chunk,
    update_transcript_tags_and_echotag,
)
from .analyzer import analyze_segment

SAMPLE_RATE = NEMOTRON_SAMPLE_RATE
EMIT_MIN_INTERVAL = 1.0 / max(0.1, getattr(settings, "TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC", 15))
VAD_RMS_THRESHOLD = max(0.0, getattr(settings, "TRANSCRIPT_VAD_RMS_THRESHOLD", 0.008))
VAD_WINDOW_SAMPLES = max(1, getattr(settings, "TRANSCRIPT_VAD_WINDOW_SAMPLES", 1024))
VAD_STEP_SAMPLES = max(1, getattr(settings, "TRANSCRIPT_VAD_STEP_SAMPLES", 512))
PCM_QUEUE_MAX = getattr(settings, "TRANSCRIPT_PCM_QUEUE_MAX_SIZE", 256)
INTERVAL_BUFFER_MAX = max(1, getattr(settings, "TRANSCRIPT_INTERVAL_BUFFER_MAX", 2048))
WS_RECEIVE_TIMEOUT = max(60.0, getattr(settings, "TRANSCRIPT_WS_RECEIVE_TIMEOUT_SEC", 86400))
# Cap concurrent live-transcription sessions to bound GPU/queue use (0 = unbounded). (H11)
MAX_WS_SESSIONS = max(0, getattr(settings, "TRANSCRIPT_WS_MAX_SESSIONS", 16))

# Single event loop -> a plain int is safe as long as we never await between read and update.
_active_ws_sessions = 0

_gpu_sem: Optional[asyncio.Semaphore] = None


def _get_gpu_sem() -> asyncio.Semaphore:
    global _gpu_sem
    if _gpu_sem is None:
        _gpu_sem = asyncio.Semaphore(max(1, getattr(settings, "TRANSCRIPT_GPU_CONCURRENCY", 1)))
    return _gpu_sem


def _is_valid_english_piece(piece: str) -> bool:
    """Filter out noise and non-English; allow digits so spoken numbers appear in transcript."""
    if not piece or not piece.strip():
        return False
    s = piece.strip()
    alnum = "".join(c for c in s if c.isalnum())
    if not alnum:
        return False
    if alnum.isdigit():
        return True
    if len(s) == 1 and s.upper() not in ("I", "A"):
        return False
    letters = [c for c in s if c.isalpha()]
    if letters:
        non_ascii = sum(1 for c in letters if ord(c) > 127)
        if non_ascii / len(letters) > 0.4:
            return False
    return True


# ── Per-connection state ──────────────────────────────────────────────────────

class _Ctx:
    """All mutable state for a single transcription WebSocket connection."""

    __slots__ = (
        "ws", "loop", "asr_adapter", "stream_ctx", "use_cuda",
        "session_id", "session", "started_at", "started_at_iso",
        "session_name", "session_location", "transcript_id",
        "last_auto_stored_length", "interval_buffer", "last_emit_time",
        "auto_store", "mode", "language", "client_sample_rate",
        "asr_stream_lock", "pcm_queue",
        "periodic_auto_store_task", "consumer_task", "analysis_tasks",
        "emitted_paragraph_ids", "dropped_frames", "last_overload_notify", "store_lock",
        "analysis_sem", "kb_namespace", "analysis_always",
    )

    def __init__(self, ws: WebSocket, loop, asr_adapter, stream_ctx):
        self.ws = ws
        self.loop = loop
        self.asr_adapter = asr_adapter
        self.stream_ctx = stream_ctx
        self.use_cuda: bool = asr_adapter.device == "cuda"

        self.session_id: Optional[str] = None
        self.session: Optional[SessionState] = None
        self.started_at: Optional[float] = None
        self.started_at_iso: Optional[str] = None
        self.session_name: str = ""
        self.session_location: str = ""
        self.transcript_id: Optional[str] = None
        self.last_auto_stored_length: int = 0
        self.interval_buffer: list = []
        self.last_emit_time: float = 0.0

        self.auto_store: bool = settings.ECHOMIND_AUTO_STORE_DEFAULT
        self.mode: str = "transcribe"
        self.language: str = "en"
        self.client_sample_rate: Optional[int] = None
        self.kb_namespace: str = ""          # active vertical KB namespace ("" = whole KB)
        self.analysis_always: bool = False   # Parakeet-style: always surface relevant KB info per paragraph

        self.asr_stream_lock: asyncio.Lock = asyncio.Lock()
        self.pcm_queue: asyncio.Queue = asyncio.Queue(
            maxsize=PCM_QUEUE_MAX if PCM_QUEUE_MAX > 0 else 0
        )

        self.periodic_auto_store_task: Optional[asyncio.Task] = None
        self.consumer_task: Optional[asyncio.Task] = None
        self.analysis_tasks: list = []
        # Paragraph ids already sent as a 'segment' + analyzed, so EOS only analyzes the final one.
        self.emitted_paragraph_ids: set = set()
        self.dropped_frames: int = 0          # audio frames dropped on queue overflow (M9)
        self.last_overload_notify: float = 0.0
        # Serializes periodic auto-store vs EOS store so a cancelled mid-store can't double-write. (M10)
        self.store_lock: asyncio.Lock = asyncio.Lock()
        # Cap concurrent Silent-Assistant analyses so a burst of paragraphs can't fan out unbounded
        # embed+LLM calls. (M11)
        self.analysis_sem: asyncio.Semaphore = asyncio.Semaphore(
            max(1, getattr(settings, "TRANSCRIPT_ANALYSIS_CONCURRENCY", 2))
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _backfill_analysis_cards(session_id: str, tid: str) -> None:
    """Stamp any analysis cards saved before transcript_id existed on this session."""
    try:
        with get_conn() as conn:
            conn.execute(
                "UPDATE transcript_analysis SET transcript_id = ? WHERE session_id = ? AND transcript_id IS NULL",
                (tid, session_id),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("Analysis backfill error (session=%s): %s", session_id, exc)


def _ensure_session(ctx: _Ctx) -> None:
    if ctx.session is None:
        ctx.session_id = str(uuid.uuid4())
        ctx.session = SessionState(ctx.session_id)
        ctx.started_at = time.time()
        ctx.last_auto_stored_length = 0
        ctx.interval_buffer.clear()
        _start_periodic_auto_store(ctx)


def _start_periodic_auto_store(ctx: _Ctx) -> None:
    interval = max(0, getattr(settings, "AUTO_STORE_INTERVAL_SEC", 60))
    if not ctx.auto_store or interval <= 0:
        return
    if ctx.periodic_auto_store_task is not None and not ctx.periodic_auto_store_task.done():
        ctx.periodic_auto_store_task.cancel()
    ctx.periodic_auto_store_task = asyncio.create_task(_periodic_auto_store_fn(ctx))


def _cancel_periodic_auto_store(ctx: _Ctx) -> None:
    if ctx.periodic_auto_store_task is not None and not ctx.periodic_auto_store_task.done():
        ctx.periodic_auto_store_task.cancel()


def _drain_pcm_queue(ctx: _Ctx) -> None:
    while True:
        try:
            ctx.pcm_queue.get_nowait()
        except asyncio.QueueEmpty:
            break


# ── Shared store-to-KB logic (used by periodic auto-store and EOS) ────────────

async def _store_chunk_to_kb(
    ctx: _Ctx,
    to_store: str,
    full_text: str,
    name: Optional[str],
    location: str,
    echodate_iso: str,
) -> tuple:
    """Create/append the transcript row, backfill analysis cards, add chunk to RAG KB."""

    def _do_db():
        tid = ctx.transcript_id
        if tid is None:
            tid = create_transcript_for_session(
                name=name,
                location=location,
                started_at_iso=echodate_iso,
                initial_text=to_store,
                session_id=ctx.session_id,
            )
        else:
            append_transcript_chunk(tid, to_store)
        conv_type, tags = get_metadata(full_text)
        echotag_val = ",".join(tags) if tags else (name or "transcript")
        update_transcript_tags_and_echotag(tid, tags, echotag_val)
        return tid, conv_type, tags

    tid, conv_type, tags = await ctx.loop.run_in_executor(None, _do_db)
    ctx.transcript_id = tid

    await ctx.loop.run_in_executor(None, lambda: _backfill_analysis_cards(ctx.session_id, tid))

    meta = {
        "session_id": ctx.session_id,
        "kind": "raw",
        "tags": tags,
        "conversation_type": conv_type,
        "ts": now_iso(),
        "type": "transcript",
        "transcript_id": tid,
        "name": name,
        "location": location,
        "echodate": echodate_iso,
        "epoch": int(time.time()),
    }
    kid = await kb.kb_add_text(to_store, meta)
    ctx.interval_buffer.append((to_store, now_iso()))
    if len(ctx.interval_buffer) > INTERVAL_BUFFER_MAX:
        ctx.interval_buffer.pop(0)

    await _send(ctx.ws, {
        "type": "stored",
        "session_id": ctx.session_id,
        "transcript_id": tid,
        "items": [{"id": kid, "kind": "raw", "tags": tags, "ts": now_iso()}],
    })
    return tid, conv_type, tags


# ── Async worker functions ────────────────────────────────────────────────────

async def _periodic_auto_store_fn(ctx: _Ctx) -> None:
    interval = max(0, getattr(settings, "AUTO_STORE_INTERVAL_SEC", 60))
    while True:
        await asyncio.sleep(interval)
        if ctx.session is None:
            break
        try:
            full_text = ctx.session.get_display_text()
            to_store = full_text[ctx.last_auto_stored_length:].strip()
            if not to_store:
                continue
            name = (ctx.session_name or "").strip() or None
            location = (ctx.session_location or "").strip() or "default"
            echodate_iso = ctx.started_at_iso or now_iso()
            # Hold the store lock and shield the write so a cancel (e.g. EOS) can't tear a
            # half-finished store and cause the EOS path to re-store the same text. (M10)
            async with ctx.store_lock:
                await asyncio.shield(
                    _store_chunk_to_kb(ctx, to_store, full_text, name, location, echodate_iso)
                )
                ctx.last_auto_stored_length = len(full_text)
        except asyncio.CancelledError:
            break
        except Exception as e:
            try:
                await _send(ctx.ws, {"type": "error", "message": str(e)})
            except Exception:
                pass


async def _maybe_emit_partial(ctx: _Ctx, ts_ms: int) -> None:
    if ctx.session is None:
        return
    if not ctx.session.differs_from_last_emit():
        return
    if time.time() - ctx.last_emit_time < EMIT_MIN_INTERVAL:
        return
    ctx.last_emit_time = time.time()
    ctx.session.mark_emitted()
    segments_payload = [
        {"paragraph_id": p.paragraph_id, "text": p.raw_text}
        for p in ctx.session.segments
    ]
    await _send(ctx.ws, {
        "type": "partial",
        "session_id": ctx.session_id,
        "text": ctx.session.get_display_text(),
        "partial_text": ctx.session.get_live_partial(),
        "segments": segments_payload,
    })


async def _run_segment_analysis(ctx: _Ctx, paragraph_id: str, paragraph_text: str) -> None:
    try:
        await _send(ctx.ws, {"type": "analysis_start", "segment_id": paragraph_id})
    except Exception:
        pass
    try:
        async with ctx.analysis_sem:  # bound concurrent embed+LLM analyses (M11)
            result = await analyze_segment(
                text=paragraph_text,
                segment_id=paragraph_id,
                session_id=ctx.session_id,
                transcript_id=ctx.transcript_id,
                namespace=ctx.kb_namespace,
                always_surface=ctx.analysis_always,
            )
        if result is not None:
            await _send(ctx.ws, result.to_ws_payload())
        else:
            await _send(ctx.ws, {"type": "analysis_done", "segment_id": paragraph_id, "result": None})
    except Exception as exc:
        logger.warning("Segment analysis error for [%s]: %s", paragraph_id, exc)
        try:
            await _send(ctx.ws, {"type": "analysis_done", "segment_id": paragraph_id, "result": None})
        except Exception:
            pass


async def _run_nemotron_frames(ctx: _Ctx, pcm_float32: np.ndarray, sr: int) -> None:
    if ctx.stream_ctx is None:
        return
    if VAD_RMS_THRESHOLD > 0 and _sliding_window_rms(pcm_float32, VAD_WINDOW_SAMPLES, VAD_STEP_SAMPLES) < VAD_RMS_THRESHOLD:
        return
    if ctx.use_cuda:
        await _get_gpu_sem().acquire()
    try:
        async with get_nemotron_process_lock():
            deltas = await ctx.stream_ctx.process_pcm_chunk(
                pcm_float32, sr, ctx.loop, ctx.asr_adapter.device
            )
    finally:
        if ctx.use_cuda:
            _get_gpu_sem().release()
    last_ts = 0
    for piece, ts_ms in deltas:
        last_ts = ts_ms
        if not piece.strip() or not _is_valid_english_piece(piece):
            continue
        _ensure_session(ctx)
        ctx.session.append_piece(piece, ts_ms)
        if ctx.session.maybe_commit(ts_ms):
            new_p = ctx.session.maybe_new_paragraph(ts_ms)
            if new_p:
                await _send(ctx.ws, {
                    "type": "segment",
                    "session_id": ctx.session_id,
                    "paragraph_id": new_p.paragraph_id,
                    "text": new_p.raw_text,
                })
                ctx.emitted_paragraph_ids.add(new_p.paragraph_id)
                t = asyncio.create_task(
                    _run_segment_analysis(ctx, new_p.paragraph_id, new_p.raw_text)
                )
                ctx.analysis_tasks.append(t)
                # Prune finished tasks so the list doesn't grow unbounded over a long session.
                ctx.analysis_tasks = [x for x in ctx.analysis_tasks if not x.done()]
    if deltas:
        await _maybe_emit_partial(ctx, last_ts)


async def _pcm_consumer(ctx: _Ctx) -> None:
    while True:
        item = await ctx.pcm_queue.get()
        if item is None:
            break
        pcm_float32, sr = item
        try:
            async with ctx.asr_stream_lock:
                await _run_nemotron_frames(ctx, pcm_float32, sr)
        except Exception as e:
            logger.warning("PCM consumer STT error (continuing): %s", e)
        ctx.pcm_queue.task_done()


# ── Message handlers ──────────────────────────────────────────────────────────

async def _handle_start(ctx: _Ctx, data: dict) -> None:
    async with ctx.asr_stream_lock:
        # Always server-generate the session id; never trust a client-supplied value. It feeds
        # DB writes and the analysis-card backfill WHERE clause, so an attacker-chosen id could
        # contaminate another session. The server echoes session_id back in partial/final/
        # analysis messages, so clients don't need to send one. (H5)
        ctx.session_id = str(uuid.uuid4())
        ctx.session = SessionState(ctx.session_id)
        ctx.started_at = time.time()
        ctx.started_at_iso = now_iso()
        ctx.session_name = (data.get("name") or "").strip() or ""
        ctx.session_location = (data.get("location") or "").strip() or "default"
        ctx.transcript_id = None
        ctx.mode = data.get("mode", "transcribe")
        ctx.language = data.get("language", "en")
        ctx.auto_store = data.get("auto_store", settings.ECHOMIND_AUTO_STORE_DEFAULT)
        ctx.client_sample_rate = data.get("sample_rate")
        ctx.kb_namespace = (data.get("namespace") or "").strip()
        ctx.analysis_always = bool(data.get("analysis_always_surface", False))
        ctx.last_auto_stored_length = 0
        ctx.interval_buffer.clear()
        _drain_pcm_queue(ctx)
        _start_periodic_auto_store(ctx)
        async with get_nemotron_process_lock():
            ctx.stream_ctx.reset()


async def _handle_eos(ctx: _Ctx) -> None:
    async with ctx.asr_stream_lock:
        _drain_pcm_queue(ctx)
        _cancel_periodic_auto_store(ctx)
        _ensure_session(ctx)
        if ctx.use_cuda:
            await _get_gpu_sem().acquire()
        try:
            async with get_nemotron_process_lock():
                flush_deltas = await ctx.stream_ctx.flush(ctx.loop)
        finally:
            if ctx.use_cuda:
                _get_gpu_sem().release()
        ts_flush_ms = int((ctx.stream_ctx.audio_offset_samples / NEMOTRON_SAMPLE_RATE) * 1000)
        for piece, ts_ms in flush_deltas:
            tsm = ts_ms if ts_ms else ts_flush_ms
            if piece.strip() and _is_valid_english_piece(piece):
                ctx.session.append_piece(piece, tsm)
        async with get_nemotron_process_lock():
            ctx.stream_ctx.reset()
        ctx.session.finalize()
        final_text = ctx.session.get_display_text()
        segments_payload = [
            {"paragraph_id": p.paragraph_id, "text": p.raw_text}
            for p in ctx.session.segments
        ]
        # Any paragraph closed by finalize() (the trailing in-progress one) was never emitted
        # mid-stream, so it has no Silent-Assistant card yet. Schedule analysis for it now so the
        # final, often most important, statement gets fact-checked too. (audit H4)
        final_paragraphs = [p for p in ctx.session.segments if p.paragraph_id not in ctx.emitted_paragraph_ids]

    await _send(ctx.ws, {
        "type": "final",
        "session_id": ctx.session_id,
        "text": final_text,
        "segments": segments_payload,
    })

    for p in final_paragraphs:
        ctx.emitted_paragraph_ids.add(p.paragraph_id)
        await _send(ctx.ws, {
            "type": "segment", "session_id": ctx.session_id,
            "paragraph_id": p.paragraph_id, "text": p.raw_text,
        })
        ctx.analysis_tasks.append(
            asyncio.create_task(_run_segment_analysis(ctx, p.paragraph_id, p.raw_text))
        )

    if ctx.auto_store and final_text.strip():
        try:
            # Wait for any in-flight periodic store to finish (lock) so we read the up-to-date
            # last_auto_stored_length and don't duplicate the chunk. (M10)
            async with ctx.store_lock:
                to_store = (
                    final_text[ctx.last_auto_stored_length:].strip()
                    if ctx.last_auto_stored_length > 0
                    else final_text
                )
                if to_store:
                    name = (ctx.session_name or "").strip() or None
                    location = (ctx.session_location or "").strip() or "default"
                    echodate_iso = ctx.started_at_iso or now_iso()
                    await _store_chunk_to_kb(ctx, to_store, final_text, name, location, echodate_iso)
                    ctx.last_auto_stored_length = len(final_text)
        except Exception as e:
            await _send(ctx.ws, {"type": "error", "message": str(e)})

    logger.info("Live transcript: EOS processed, closing session=%s", ctx.session_id)


async def _handle_refine(ctx: _Ctx, data: dict) -> None:
    scope = data.get("scope", "all")
    paragraph_id = data.get("paragraph_id")
    _ensure_session(ctx)
    if scope == "all":
        text_to_refine = ctx.session.get_display_text()
        if not text_to_refine.strip():
            await _send(ctx.ws, {"type": "error", "message": "No transcript to refine"})
            return
        refined = await refine_text(text_to_refine)
        await _send(ctx.ws, {"type": "refined", "session_id": ctx.session_id, "scope": scope, "text": refined})
    elif scope == "last_paragraph" and ctx.session.segments:
        p = ctx.session.segments[-1]
        refined = await refine_text(p.raw_text)
        p.polished_text = refined
        await _send(ctx.ws, {"type": "refined", "session_id": ctx.session_id, "scope": scope, "paragraph_id": p.paragraph_id, "text": refined})
    elif scope == "paragraph" and paragraph_id:
        p = next((x for x in ctx.session.segments if x.paragraph_id == paragraph_id), None)
        if not p:
            await _send(ctx.ws, {"type": "error", "message": f"Paragraph {paragraph_id} not found"})
            return
        refined = await refine_text(p.raw_text)
        p.polished_text = refined
        await _send(ctx.ws, {"type": "refined", "session_id": ctx.session_id, "scope": scope, "paragraph_id": p.paragraph_id, "text": refined})
    else:
        await _send(ctx.ws, {"type": "error", "message": "Invalid refine scope"})


async def _handle_store_combined(ctx: _Ctx) -> None:
    _ensure_session(ctx)
    parts = [text for (text, _) in ctx.interval_buffer]
    remainder = ctx.session.get_display_text()[ctx.last_auto_stored_length:].strip()
    if remainder:
        parts.append(remainder)
    combined_raw = "\n\n".join(p for p in parts if p).strip()
    if not combined_raw:
        await _send(ctx.ws, {"type": "error", "message": "No transcript to combine"})
        return
    try:
        refined = await refine_text(combined_raw)
        result = await store_transcript_to_db(raw_text=combined_raw, refined_text=refined or None)
        ctx.interval_buffer.clear()
        ctx.last_auto_stored_length = len(ctx.session.get_display_text())
        await _send(ctx.ws, {
            "type": "stored_combined",
            "session_id": ctx.session_id,
            "transcript_id": result["transcript_id"],
            "tags": result["tags"],
            "echotag": result["echotag"],
            "echodate": result["echodate"],
            "created_at": result["created_at"],
        })
    except Exception as e:
        await _send(ctx.ws, {"type": "error", "message": str(e)})


async def _handle_store(ctx: _Ctx, data: dict) -> None:
    scope = data.get("scope", "all")
    paragraph_id = data.get("paragraph_id")
    _ensure_session(ctx)
    items = []

    if scope == "all":
        full = ctx.session.get_display_text()
        if full.strip():
            conv_type, tags = get_metadata(full)
            meta = {"session_id": ctx.session_id, "kind": "raw", "paragraph_id": None, "tags": tags, "conversation_type": conv_type, "ts": now_iso(), "epoch": int(time.time())}
            kid = await kb.kb_add_text(full, meta)
            items.append({"id": kid, "kind": "raw", "paragraph_id": None, "tags": tags, "ts": now_iso()})
            full_refined = await refine_text(full)
            if full_refined.strip():
                conv_type2, tags2 = get_metadata(full_refined)
                meta2 = {"session_id": ctx.session_id, "kind": "refined", "paragraph_id": None, "tags": tags2, "conversation_type": conv_type2, "ts": now_iso(), "epoch": int(time.time())}
                kid2 = await kb.kb_add_text(full_refined, meta2)
                items.append({"id": kid2, "kind": "refined", "paragraph_id": None, "tags": tags2, "ts": now_iso()})
        for p in ctx.session.segments:
            if p.polished_text:
                conv_type, tags = get_metadata(p.polished_text)
                meta = {"session_id": ctx.session_id, "kind": "refined", "paragraph_id": p.paragraph_id, "tags": tags, "conversation_type": conv_type, "ts": now_iso(), "epoch": int(time.time())}
                kid = await kb.kb_add_text(p.polished_text, meta)
                items.append({"id": kid, "kind": "refined", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
            conv_type, tags = get_metadata(p.raw_text)
            meta = {"session_id": ctx.session_id, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "conversation_type": conv_type, "ts": now_iso(), "epoch": int(time.time())}
            kid = await kb.kb_add_text(p.raw_text, meta)
            items.append({"id": kid, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
    elif scope == "last_paragraph" and ctx.session.segments:
        p = ctx.session.segments[-1]
        conv_type, tags = get_metadata(p.raw_text)
        meta = {"session_id": ctx.session_id, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "conversation_type": conv_type, "ts": now_iso()}
        kid = await kb.kb_add_text(p.raw_text, meta)
        items.append({"id": kid, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
        if p.polished_text:
            kid2 = await kb.kb_add_text(p.polished_text, {**meta, "kind": "refined", "epoch": int(time.time())})
            items.append({"id": kid2, "kind": "refined", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
    elif scope == "paragraph" and paragraph_id:
        p = next((x for x in ctx.session.segments if x.paragraph_id == paragraph_id), None)
        if not p:
            await _send(ctx.ws, {"type": "error", "message": f"Paragraph {paragraph_id} not found"})
            return
        conv_type, tags = get_metadata(p.raw_text)
        meta = {"session_id": ctx.session_id, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "conversation_type": conv_type, "ts": now_iso()}
        kid = await kb.kb_add_text(p.raw_text, meta)
        items.append({"id": kid, "kind": "raw", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})
        if p.polished_text:
            kid2 = await kb.kb_add_text(p.polished_text, {**meta, "kind": "refined", "epoch": int(time.time())})
            items.append({"id": kid2, "kind": "refined", "paragraph_id": p.paragraph_id, "tags": tags, "ts": now_iso()})

    await _send(ctx.ws, {"type": "stored", "session_id": ctx.session_id, "items": items})


# ── Main WebSocket handler ────────────────────────────────────────────────────

async def handler(ws: WebSocket):
    global _active_ws_sessions
    await ws.accept()

    # Capacity guard: reject when too many sessions are active. Check + increment with no
    # await in between so concurrent connects can't both slip past the cap. (H11)
    if MAX_WS_SESSIONS and _active_ws_sessions >= MAX_WS_SESSIONS:
        logger.warning("Live transcript: at capacity (%d sessions), rejecting connection", _active_ws_sessions)
        try:
            await _send(ws, {"type": "error", "message": "Server at capacity. Please retry shortly."})
            await ws.close()
        except Exception:
            pass
        return
    _active_ws_sessions += 1

    await _send(ws, {"type": "loading"})

    loop = asyncio.get_running_loop()
    asr_adapter = None
    if NEMOTRON_AVAILABLE:
        try:
            asr_adapter = await loop.run_in_executor(None, get_shared_asr_adapter)
        except Exception as e:
            await _send(ws, {"type": "error", "message": f"Nemotron ASR load failed: {e}"})
            _active_ws_sessions -= 1
            return

    if asr_adapter is None:
        hint = f" ({_nemotron_import_error})" if _nemotron_import_error else ""
        await _send(ws, {
            "type": "error",
            "message": (
                f"Nemotron ASR not available. Install NeMo ASR (see backend Dockerfile)."
                f" Live transcript requires Nemotron.{hint}"
            ),
        })
        _active_ws_sessions -= 1
        return

    try:
        async with get_nemotron_process_lock():
            stream_ctx = NemotronStreamContext(asr_adapter)
    except Exception as e:
        # Don't leak the active-session slot if stream context creation fails.
        _active_ws_sessions -= 1
        try:
            await _send(ws, {"type": "error", "message": f"Stream init failed: {e}"})
        except Exception:
            pass
        return

    # Tell the client the model is loaded and what sample rate to use.
    # Client is blocked waiting for this before it sends "start".
    await _send(ws, {"type": "ready", "sample_rate": NEMOTRON_SAMPLE_RATE})

    ctx = _Ctx(ws, loop, asr_adapter, stream_ctx)
    ctx.consumer_task = asyncio.create_task(_pcm_consumer(ctx))

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=WS_RECEIVE_TIMEOUT)
            except asyncio.TimeoutError:
                logger.info("Live transcript: no message for %.0fs, closing stale connection", WS_RECEIVE_TIMEOUT)
                try:
                    await _send(ws, {"type": "error", "message": "Connection timed out. Reconnecting…"})
                except Exception:
                    pass
                break
            except WebSocketDisconnect:
                logger.info("Live transcript: client disconnected session=%s", ctx.session_id)
                break

            if not isinstance(msg, dict) or msg.get("type") != "websocket.receive":
                if msg.get("type") == "websocket.disconnect":
                    logger.info("Live transcript: websocket.disconnect session=%s", ctx.session_id)
                    break
                continue

            # Binary: raw PCM16 (preferred — no base64 overhead)
            raw_bytes = msg.get("bytes")
            if raw_bytes and len(raw_bytes) > 0:
                if ctx.session and ctx.session._paused:
                    continue
                sr = ctx.client_sample_rate if ctx.client_sample_rate is not None else NEMOTRON_SAMPLE_RATE
                try:
                    ctx.pcm_queue.put_nowait((_pcm16_to_float32(bytes(raw_bytes)), sr))
                except asyncio.QueueFull:
                    await _note_dropped_frame(ctx)
                continue

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

            # Legacy: base64-wrapped audio JSON (kept for backward-compat with older clients)
            if t == "audio":
                b64 = data.get("pcm16_b64")
                if b64 and (ctx.session is None or not ctx.session._paused):
                    sr = ctx.client_sample_rate if ctx.client_sample_rate is not None else NEMOTRON_SAMPLE_RATE
                    try:
                        ctx.pcm_queue.put_nowait((_pcm16_to_float32(base64.b64decode(b64)), sr))
                    except asyncio.QueueFull:
                        await _note_dropped_frame(ctx)
                continue
            if t == "ping":
                continue
            if t == "stop":
                t = "eos"
                data = {"type": "eos"}
            if t == "start":
                await _handle_start(ctx, data)
                continue
            if t == "pause":
                if ctx.session:
                    ctx.session.pause()
                continue
            if t == "resume":
                if ctx.session:
                    ctx.session.resume()
                continue
            if t == "eos":
                await _handle_eos(ctx)
                break
            if t == "refine":
                await _handle_refine(ctx, data)
                continue
            if t == "store_combined":
                await _handle_store_combined(ctx)
                continue
            if t == "store":
                await _handle_store(ctx, data)
                continue

    except Exception as e:
        logger.warning("Live transcript: unhandled exception session=%s: %s", ctx.session_id, e)
        try:
            await _send(ws, {"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Drain in-flight Silent Assistant analyses before cancelling. The trailing
        # paragraph is only closed by finalize() at EOS, so its analysis starts here —
        # the old unconditional cancel meant the LAST statement of every session was
        # never fact-checked (measured: mid-stream card arrived in 3.5s, trailing card
        # never arrived even after 45s). The socket is still open in `finally`, so the
        # card can still be delivered. Cancel only stragglers past the deadline.
        pending = [at for at in ctx.analysis_tasks if not at.done()]
        if pending:
            drain_s = float(os.getenv("TRANSCRIPT_ANALYSIS_DRAIN_SEC", "35"))
            try:
                _, still_pending = await asyncio.wait(pending, timeout=drain_s)
            except Exception:
                still_pending = set(pending)
            if still_pending:
                logger.warning(
                    "Live transcript: %d analysis task(s) exceeded %.0fs drain — cancelling session=%s",
                    len(still_pending), drain_s, ctx.session_id,
                )
                for at in still_pending:
                    at.cancel()
        if ctx.analysis_tasks:
            await asyncio.gather(*ctx.analysis_tasks, return_exceptions=True)

        if ctx.consumer_task is not None and not ctx.consumer_task.done():
            try:
                ctx.pcm_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            try:
                await asyncio.wait_for(ctx.consumer_task, timeout=5.0)
            except asyncio.TimeoutError:
                ctx.consumer_task.cancel()
                try:
                    await ctx.consumer_task
                except asyncio.CancelledError:
                    pass

        _active_ws_sessions = max(0, _active_ws_sessions - 1)
        logger.info("Live transcript: handler fully cleaned up session=%s", ctx.session_id)


async def _note_dropped_frame(ctx: "_Ctx") -> None:
    """Record an audio frame dropped due to queue overflow and warn (rate-limited) so the drop
    isn't silent — the operator gets a log and the client gets a one-shot backpressure signal. (M9)"""
    ctx.dropped_frames += 1
    now = time.time()
    if now - ctx.last_overload_notify >= 5.0:
        ctx.last_overload_notify = now
        logger.warning(
            "Live transcript: audio overload, dropped %d frame(s) so far session=%s",
            ctx.dropped_frames, ctx.session_id,
        )
        try:
            await _send(ctx.ws, {
                "type": "overloaded",
                "message": "Audio is arriving faster than it can be transcribed; some frames were dropped.",
                "dropped_frames": ctx.dropped_frames,
            })
        except Exception:
            pass


async def _send(ws: WebSocket, obj: dict):
    await ws.send_text(json.dumps(obj))
