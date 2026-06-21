"""
Voice path STT: NVIDIA Nemotron (NeMo) utterance-final transcription.
Uses shared nemotron_asr package (same adapter as backend live transcript).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
from typing import Optional

import numpy as np

from ..config import SETTINGS

logger = logging.getLogger(__name__)

for _k in ("TORCHDYNAMO_DISABLE", "TORCHINDUCTOR_DISABLE", "TORCH_COMPILE_DISABLE"):
    if _k not in os.environ:
        os.environ[_k] = "1"

_adapter_lock = threading.Lock()
_shared_adapter: Optional[object] = None
_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

# ── GPU/CUDA fatal-fault detection ───────────────────────────────────────────
# A CUDA error (e.g. cudaErrorUnknown) poisons the process CUDA context so that
# EVERY subsequent inference fails. There is no in-process recovery — the container
# must restart. We flag it here so /health can report unhealthy and a watchdog can
# exit the process for the orchestrator (restart: unless-stopped) to recreate it.
_stt_fatal = threading.Event()
_FATAL_HINTS = (
    "cuda", "cublas", "cudnn", "cudaerror", "device-side assert",
    "illegal memory access", "no kernel image", "misaligned address",
)


def is_fatal_gpu_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return any(h in msg for h in _FATAL_HINTS)


def note_stt_error(e: BaseException) -> bool:
    """Mark STT unrecoverable if the error indicates a poisoned CUDA context. Returns True if fatal."""
    if is_fatal_gpu_error(e):
        if not _stt_fatal.is_set():
            logger.critical(
                "Voice Nemotron: FATAL GPU/CUDA error — context poisoned, STT marked UNHEALTHY "
                "(container restart required): %s", e,
            )
            _stt_fatal.set()
        return True
    return False


def stt_healthy() -> bool:
    return not _stt_fatal.is_set()


def stt_fatal_event() -> threading.Event:
    return _stt_fatal


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        workers = max(1, int(os.getenv("VOICE_NEMOTRON_EXECUTOR_WORKERS", "1")))
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="voice_nemotron_stt",
        )
    return _executor


def get_shared_asr_adapter():
    """Process-wide shared Nemotron weights (matches backend single-model assumption)."""
    global _shared_adapter
    from nemotron_asr.adapter import ASRModelAdapter

    with _adapter_lock:
        if _shared_adapter is None:
            model = SETTINGS.NEMOTRON_MODEL_NAME
            att = SETTINGS.NEMOTRON_ATT_CONTEXT_RIGHT
            logger.info("Voice Nemotron: loading model=%s att_context_right=%s", model, att)
            t0 = time.monotonic()
            ad = ASRModelAdapter(model_name=model, att_context_right=att)
            ad.load()
            logger.info(
                "Voice Nemotron: model ready device=%s load_wall_s=%.2f",
                getattr(ad, "device", "?"),
                time.monotonic() - t0,
            )
            _shared_adapter = ad
        return _shared_adapter


def ensure_nemotron_loaded_at_startup() -> None:
    """Load at process start only when VOICE_NEMOTRON_STARTUP_LOAD=1. Default 0: lazy load on first transcribe."""
    if os.getenv("VOICE_NEMOTRON_STARTUP_LOAD", "0").strip().lower() not in ("1", "true", "yes"):
        logger.warning("Voice Nemotron: startup load skipped (VOICE_NEMOTRON_STARTUP_LOAD=0)")
        return
    try:
        get_shared_asr_adapter()
    except Exception as e:
        logger.exception("Voice Nemotron: FATAL startup load failed: %s", e)
        raise RuntimeError(
            "Nemotron STT failed to load. Install NeMo ASR in the voice image, set HF_TOKEN if needed, "
            "and ensure model weights are available (build-time download or HF cache volume). "
            f"Detail: {e}"
        ) from e


class NemotronUtteranceSTT:
    """Async-friendly utterance STT for OmniSessionA (float32 mono, sample rate from SETTINGS.SR)."""

    def __init__(self):
        self.sample_rate = SETTINGS.SR
        self.chunk_ms = SETTINGS.NEMOTRON_CHUNK_MS

    async def transcribe(self, audio_f32: np.ndarray) -> str:
        from nemotron_asr.utterance import transcribe_utterance_float32

        adapter = get_shared_asr_adapter()
        ex = _get_executor()
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        def _run() -> str:
            return transcribe_utterance_float32(
                adapter,
                audio_f32,
                sample_rate=self.sample_rate,
                chunk_ms=self.chunk_ms,
            )

        try:
            text = await loop.run_in_executor(ex, _run)
        except Exception as e:
            note_stt_error(e)  # flag unhealthy if this is a poisoned-CUDA fault
            logger.exception("Voice Nemotron: transcribe failed: %s", e)
            raise
        ms = (time.monotonic() - t0) * 1000.0
        preview = (text or "")[:120]
        logger.info(
            "Voice Nemotron: transcribe done latency_ms=%.0f chars=%d text_preview=%r",
            ms,
            len(text or ""),
            preview,
        )
        return text or ""


class NemotronStreamingSTT:
    """
    Per-utterance streaming STT using Nemotron's process_chunk API.

    Feed audio frames while the user is still speaking to get a growing
    partial transcript. This lets us classify intent (knowledge-intensive vs.
    casual) before the utterance ends, so we can pre-warm RAG and pick the
    right lead phrase without any extra latency.

    Thread-safety: push_chunk() is designed to be called from a single thread
    (the executor thread in _consume_loop). reset() must be called between
    utterances.
    """

    def __init__(self):
        self.sample_rate = SETTINGS.SR
        # Nemotron streaming chunk: 560ms default, same as live transcript
        frame_ms = int(os.getenv("VOICE_STREAMING_CHUNK_MS", "560"))
        self._frame_samples = int(self.sample_rate * frame_ms / 1000)
        self._buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._state = None
        self._step: int = 0
        self._last_hyp: str = ""

    def reset(self) -> None:
        """Call at the start of each new utterance."""
        self._buf = np.zeros(0, dtype=np.float32)
        self._state = None
        self._step = 0
        self._last_hyp = ""

    def push_chunk(self, audio_f32: np.ndarray) -> str:
        """
        Accumulate audio and process full Nemotron frames synchronously.
        Returns the latest partial hypothesis (may be empty string).
        Safe to call from a thread-pool executor.
        """
        try:
            adapter = get_shared_asr_adapter()
            if self._state is None:
                self._state = adapter.create_session_state()

            self._buf = np.concatenate([self._buf, audio_f32])

            while len(self._buf) >= self._frame_samples:
                frame = self._buf[: self._frame_samples]
                self._buf = self._buf[self._frame_samples :]
                try:
                    hyp, self._state = adapter.process_chunk(
                        frame,
                        self._state,
                        keep_all_outputs=False,
                        step_num=self._step,
                    )
                    if hyp and hyp.strip():
                        self._last_hyp = hyp
                except Exception as e:
                    note_stt_error(e)
                    logger.debug("NemotronStreamingSTT chunk error (step %d): %s", self._step, e)
                self._step += 1

        except Exception as e:
            note_stt_error(e)
            logger.debug("NemotronStreamingSTT push_chunk error: %s", e)

        return self._last_hyp

    def latest_partial(self) -> str:
        """Return the most recent partial hypothesis without feeding new audio."""
        return self._last_hyp
