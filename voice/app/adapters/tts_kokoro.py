"""
Kokoro-82M TTS adapter — a more natural-sounding alternative to Piper, selectable from Settings.
Same interface as PiperTTS (``synth(text) -> np.ndarray`` at ``self.sr``) so the session can swap engines.
Runs on CPU (the Spark has no onnxruntime CUDA EP); model + voice are baked for offline use.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pipeline = None  # process-wide shared KPipeline (one model for all sessions)
_SR = 24000


def _get_pipeline(lang_code: str = "a"):
    global _pipeline
    with _lock:
        if _pipeline is None:
            from kokoro import KPipeline  # local import: heavy, optional dependency
            t = __import__("time").monotonic()
            _pipeline = KPipeline(lang_code=lang_code)
            logger.info("Kokoro TTS: pipeline ready (lang=%s) in %.1fs",
                        lang_code, __import__("time").monotonic() - t)
        return _pipeline


def ensure_kokoro_loaded() -> None:
    """Pre-warm the Kokoro pipeline so the first synth isn't slow. Non-fatal."""
    try:
        _get_pipeline()
    except Exception as e:
        logger.warning("Kokoro TTS: pre-warm failed (%s); Piper remains the fallback", e)


class KokoroTTS:
    """Drop-in for PiperTTS. ``voice`` defaults to a baked-in English voice."""

    def __init__(self, voice: str = "af_heart", lang_code: str = "a"):
        self.voice = voice or "af_heart"
        self.lang_code = lang_code or "a"
        self.sr = _SR
        _get_pipeline(self.lang_code)  # raises if Kokoro unavailable → caller keeps Piper

    def synth(self, text: str) -> np.ndarray:
        if not (text or "").strip():
            return np.zeros(0, dtype=np.float32)
        p = _get_pipeline(self.lang_code)
        chunks = []
        for _gs, _ps, aud in p(text, voice=self.voice):
            a = aud.detach().cpu().numpy() if hasattr(aud, "detach") else np.asarray(aud)
            chunks.append(np.asarray(a, dtype=np.float32).reshape(-1))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32)
