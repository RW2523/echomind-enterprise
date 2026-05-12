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


def _drop_shared_asr_and_force_cpu_env() -> None:
    """Clear cached adapter so next load uses ECHOMIND_ASR_DEVICE=cpu (runtime escape hatch for GPU contention)."""
    global _shared_adapter
    with _adapter_lock:
        _shared_adapter = None
    os.environ["ECHOMIND_ASR_DEVICE"] = "cpu"
    os.environ["ECHOMIND_ASR_REQUIRE_CUDA"] = "0"


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
            err_l = str(e).lower()
            cudaish = (
                "cuda" in err_l
                or "acceleratorerror" in err_l
                or "cublas" in err_l
                or "nvrtc" in err_l
            )
            if cudaish and getattr(adapter, "device", None) == "cuda":
                logger.warning(
                    "Voice Nemotron: CUDA transcribe failed (%s); reloading ASR on CPU once",
                    type(e).__name__,
                )
                _drop_shared_asr_and_force_cpu_env()
                adapter = get_shared_asr_adapter()

                def _run_cpu() -> str:
                    return transcribe_utterance_float32(
                        adapter,
                        audio_f32,
                        sample_rate=self.sample_rate,
                        chunk_ms=self.chunk_ms,
                    )

                try:
                    text = await loop.run_in_executor(ex, _run_cpu)
                except Exception as e2:
                    logger.exception("Voice Nemotron: transcribe failed after CPU fallback: %s", e2)
                    raise e2 from e
            else:
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
