"""
Final Cleanup / Backup Transcription Service using microsoft/VibeVoice-ASR.

Roles:
  1. Post-processing: after a Board Room session completes primary multitalker
     transcription, run VibeVoice on the saved full-session WAV to produce a
     cleaned / improved transcript (single pass, high-accuracy Whisper-family
     model).
  2. Fallback: if primary Board Room ASR fails, run VibeVoice on the full WAV
     as the fallback transcription source.

Architecture:
  - Uses HuggingFace transformers pipeline (automatic-speech-recognition).
  - Loads the model lazily on first use; optionally warms at startup.
  - Processes audio in chunks (FINAL_CLEANUP_CHUNK_SEC seconds) to avoid OOM
    on long meetings.
  - Does NOT replace the live streaming path; runs only on finalized audio files.
  - Thread-safe: one inference at a time via a threading lock (same GPU shared
    with Nemotron and Parakeet).

Output segment format (EchoMind normalized):
  {
    "speaker": null,          # VibeVoice is single-pass; no diarization
    "start_time": 12.4,
    "end_time":   16.8,
    "text":       "...",
    "confidence": null,
    "source":     "vibevoice_cleanup"    # or "vibevoice_fallback"
  }
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from ..core.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_VIBEVOICE_MODEL_NAME: str = settings.FINAL_CLEANUP_MODEL
_CHUNK_SEC: int = settings.FINAL_CLEANUP_CHUNK_SEC

# ── Status tracking ───────────────────────────────────────────────────────────

ModelStatus = Literal["not_loaded", "loading", "ready", "failed", "disabled"]

_status: ModelStatus = "not_loaded"
_load_error: Optional[str] = None
_model_lock = threading.Lock()
_pipeline: Optional[Any] = None  # transformers ASR pipeline


def get_status() -> Dict[str, Any]:
    return {
        "model": _VIBEVOICE_MODEL_NAME,
        "status": _status,
        "error": _load_error,
        "enabled": settings.FINAL_CLEANUP_ENABLED,
    }


# ── Model loading ─────────────────────────────────────────────────────────────

VIBEVOICE_AVAILABLE = False
_import_error: Optional[str] = None

try:
    import transformers  # noqa: F401
    VIBEVOICE_AVAILABLE = True
except ImportError as _e:
    _import_error = str(_e)
    logger.warning("VibeVoice-ASR: transformers not available — %s", _e)


def _resolve_device() -> str:
    cfg = settings.FINAL_CLEANUP_DEVICE.strip().lower()
    if cfg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return cfg


def _load_pipeline() -> bool:
    """Load the VibeVoice ASR pipeline. Returns True on success."""
    global _pipeline, _status, _load_error
    if not VIBEVOICE_AVAILABLE:
        _status = "failed"
        _load_error = f"transformers not installed: {_import_error}"
        return False
    if not settings.FINAL_CLEANUP_ENABLED:
        _status = "disabled"
        return False

    _status = "loading"
    logger.info("model.cleanup.load.start model=%s", _VIBEVOICE_MODEL_NAME)
    t0 = time.monotonic()

    try:
        import torch
        from transformers import pipeline as hf_pipeline

        device = _resolve_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("VibeVoice-ASR: loading %s on %s (dtype=%s) ...", _VIBEVOICE_MODEL_NAME, device, dtype)

        _pipeline = hf_pipeline(
            "automatic-speech-recognition",
            model=_VIBEVOICE_MODEL_NAME,
            torch_dtype=dtype,
            device=device,
            model_kwargs={"attn_implementation": "sdpa"},  # efficient attention when available
        )

        elapsed = time.monotonic() - t0
        logger.info("model.cleanup.load.ready model=%s device=%s elapsed_sec=%.1f", _VIBEVOICE_MODEL_NAME, device, elapsed)
        _status = "ready"
        _load_error = None
        return True

    except Exception as exc:
        elapsed = time.monotonic() - t0
        _load_error = str(exc)
        _status = "failed"
        logger.error(
            "VibeVoice-ASR: load failed model=%s elapsed_sec=%.1f error=%s",
            _VIBEVOICE_MODEL_NAME, elapsed, exc,
        )
        return False


def preload_vibevoice() -> bool:
    """Thread-safe model load (called at startup warmup)."""
    with _model_lock:
        if _status in ("ready", "loading"):
            return _status == "ready"
        return _load_pipeline()


def _get_pipeline():
    """Return the loaded pipeline, loading it if needed (lazy)."""
    global _pipeline
    with _model_lock:
        if _status == "ready" and _pipeline is not None:
            return _pipeline
        if _status in ("failed", "disabled"):
            return None
        # Not loaded yet — load now (lazy)
        _load_pipeline()
        return _pipeline


# ── Audio I/O helpers ─────────────────────────────────────────────────────────

def _load_wav_float32(wav_path: str) -> Tuple[np.ndarray, int]:
    """Load a WAV file and return (float32 mono audio, sample_rate)."""
    try:
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except ImportError:
        pass
    # Fallback: scipy
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        audio = data.astype(np.float32)
        if data.dtype == np.int16:
            audio /= 32768.0
        elif data.dtype == np.int32:
            audio /= 2147483648.0
        return audio, sr
    except ImportError:
        pass
    raise RuntimeError("Neither soundfile nor scipy is installed; cannot load WAV for VibeVoice cleanup.")


def _resample_to_16k(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == 16000:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=src_sr, target_sr=16000)
    except ImportError:
        pass
    # Simple linear interpolation fallback
    ratio = 16000 / src_sr
    n_new = int(len(audio) * ratio)
    return np.interp(np.linspace(0, len(audio) - 1, n_new), np.arange(len(audio)), audio).astype(np.float32)


# ── Core transcription ────────────────────────────────────────────────────────

def _transcribe_audio(
    audio: np.ndarray,
    sample_rate: int,
    source_label: str = "vibevoice_cleanup",
) -> List[Dict[str, Any]]:
    """
    Run VibeVoice ASR on audio (float32, any sr).
    Returns normalized EchoMind segment list.
    """
    pipe = _get_pipeline()
    if pipe is None:
        raise RuntimeError(
            f"VibeVoice-ASR pipeline not available (status={_status}, enabled={settings.FINAL_CLEANUP_ENABLED})"
        )

    # Resample to 16 kHz if needed
    if sample_rate != 16000:
        audio = _resample_to_16k(audio, sample_rate)

    total_sec = len(audio) / 16000
    logger.info("boardroom.cleanup.start model=%s audio_sec=%.1f source=%s", _VIBEVOICE_MODEL_NAME, total_sec, source_label)
    t0 = time.monotonic()

    segments: List[Dict[str, Any]] = []
    chunk_samples = _CHUNK_SEC * 16000
    offset_sec = 0.0

    for chunk_start in range(0, len(audio), chunk_samples):
        chunk = audio[chunk_start : chunk_start + chunk_samples]
        if len(chunk) < 160:  # < 10 ms — skip noise tails
            break

        # VibeVoice / Whisper pipeline returns {"text": ..., "chunks": [...]}
        # when return_timestamps=True
        with _model_lock:
            result = pipe(
                {"sampling_rate": 16000, "raw": chunk},
                return_timestamps=True,
                generate_kwargs={"language": "english", "task": "transcribe"},
            )

        if isinstance(result, dict):
            raw_chunks = result.get("chunks") or []
            full_text = (result.get("text") or "").strip()
            if raw_chunks:
                for rc in raw_chunks:
                    ts = rc.get("timestamp") or (None, None)
                    start = (ts[0] or 0.0) + offset_sec
                    end_t = (ts[1] or (ts[0] or 0.0) + 2.0) + offset_sec
                    text = (rc.get("text") or "").strip()
                    if not text:
                        continue
                    segments.append({
                        "speaker": None,
                        "start_time": round(start, 2),
                        "end_time": round(end_t, 2),
                        "text": text,
                        "confidence": None,
                        "source": source_label,
                    })
            elif full_text:
                # Model returned no timestamps; attach to chunk time range
                segments.append({
                    "speaker": None,
                    "start_time": round(offset_sec, 2),
                    "end_time": round(offset_sec + len(chunk) / 16000, 2),
                    "text": full_text,
                    "confidence": None,
                    "source": source_label,
                })

        offset_sec += len(chunk) / 16000

    elapsed = time.monotonic() - t0
    logger.info(
        "boardroom.cleanup.completed model=%s audio_sec=%.1f segments=%d elapsed_sec=%.1f source=%s",
        _VIBEVOICE_MODEL_NAME, total_sec, len(segments), elapsed, source_label,
    )
    return segments


def _segments_to_text(segments: List[Dict[str, Any]]) -> str:
    return " ".join(s["text"] for s in segments if s.get("text")).strip()


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe_wav_file(
    wav_path: str,
    source: Literal["vibevoice_cleanup", "vibevoice_fallback"] = "vibevoice_cleanup",
) -> Dict[str, Any]:
    """
    Transcribe a saved WAV file with VibeVoice-ASR.

    Returns:
        {
            "ok": True,
            "segments": [...],   # EchoMind normalized segments
            "text": "...",       # full cleaned text
            "source": "vibevoice_cleanup" | "vibevoice_fallback",
            "model": "microsoft/VibeVoice-ASR",
            "audio_duration_sec": 120.4,
            "elapsed_sec": 18.2,
        }

    Raises:
        RuntimeError on model load failure or audio I/O failure.
        FileNotFoundError if wav_path does not exist.
    """
    if not settings.FINAL_CLEANUP_ENABLED:
        raise RuntimeError("Final cleanup is disabled (FINAL_CLEANUP_ENABLED=false).")

    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    t0 = time.monotonic()
    audio, sr = _load_wav_float32(wav_path)
    audio_duration = len(audio) / max(sr, 1)

    segments = _transcribe_audio(audio, sr, source_label=source)
    text = _segments_to_text(segments)
    elapsed = time.monotonic() - t0

    return {
        "ok": True,
        "segments": segments,
        "text": text,
        "source": source,
        "model": _VIBEVOICE_MODEL_NAME,
        "audio_duration_sec": round(audio_duration, 2),
        "elapsed_sec": round(elapsed, 2),
    }


async def transcribe_wav_file_async(
    wav_path: str,
    source: Literal["vibevoice_cleanup", "vibevoice_fallback"] = "vibevoice_cleanup",
) -> Dict[str, Any]:
    """Async wrapper — runs transcribe_wav_file in a thread executor."""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, transcribe_wav_file, wav_path, source)
