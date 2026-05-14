"""One-shot Piper TTS over HTTP (Personal Assistant Speak Now). Independent of /ws conversation sessions."""
from __future__ import annotations

import io
import logging
import re
import wave
from functools import lru_cache
from typing import Optional

import numpy as np

from .adapters.tts_piper import PiperTTS
from .config import SETTINGS

logger = logging.getLogger(__name__)

_MAX_CHARS = 2500


def strip_for_oneoff_tts(text: str) -> str:
    """Plain text for TTS (same idea as session.strip_markdown_for_speech; avoids importing session)."""
    if not (text or "").strip():
        return (text or "").strip()
    s = (text or "").strip()
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"\*\*", "", s)
    s = re.sub(r"__", "", s)
    s = re.sub(r"\*", "", s)
    s = re.sub(r"_", " ", s)
    s = re.sub(r"`", "", s)
    s = re.sub(r"^#+\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def float32_mono_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    clip = np.clip(samples.astype(np.float32), -1.0, 1.0)
    pcm16 = (clip * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


@lru_cache(maxsize=16)
def _cached_piper(model_path: str) -> PiperTTS:
    """Reuse PiperVoice load per model path (same voices as Conversation Mode)."""
    return PiperTTS(
        model_path,
        speaker_id=SETTINGS.PIPER_SPEAKER,
        noise_scale=SETTINGS.PIPER_NOISE_SCALE,
        length_scale=SETTINGS.PIPER_LENGTH_SCALE,
        use_cuda=SETTINGS.PIPER_USE_CUDA,
    )


def resolve_model_path(voice_id: Optional[str]) -> str:
    vid = (voice_id or "").strip()
    if vid:
        if ".." in vid or "/" in vid or "\\" in vid or not re.fullmatch(r"[A-Za-z0-9_.-]+", vid):
            raise ValueError("Invalid voice_id")
        return f"/voices/{vid}.onnx"
    return SETTINGS.PIPER_MODEL


def synthesize_wav_bytes(text: str, voice_id: Optional[str] = None) -> tuple[bytes, int]:
    """
    Returns (wav_bytes, sample_rate_hz).
    Raises FileNotFoundError if model missing, ValueError for bad input.
    """
    cleaned = strip_for_oneoff_tts(text)
    if not cleaned:
        raise ValueError("text is empty after cleaning")
    if len(cleaned) > _MAX_CHARS:
        raise ValueError(f"text exceeds {_MAX_CHARS} characters")

    model_path = resolve_model_path(voice_id)
    try:
        tts = _cached_piper(model_path)
    except FileNotFoundError:
        logger.warning("Speak Now: Piper model not found: %s", model_path)
        raise

    audio = tts.synth(cleaned)
    sr = int(getattr(tts, "sr", 22050) or 22050)
    wav = float32_mono_to_wav_bytes(audio, sr)
    return wav, sr
