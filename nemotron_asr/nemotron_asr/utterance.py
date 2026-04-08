"""
Utterance-final Nemotron transcription: feed fixed-size chunks + final keep_all_outputs,
mirroring backend NemotronStreamContext + flush semantics (see backend transcribe/stt_streaming.py).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .adapter import ASRModelAdapter

logger = logging.getLogger(__name__)

NEMOTRON_SAMPLE_RATE = 16000


def _resample_linear_np(input_audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """CPU linear resample (same contract as backend: avoid GPU from sync paths)."""
    if src_sr == dst_sr:
        return input_audio.astype(np.float32, copy=False)
    orig_len = len(input_audio)
    new_len = int(round(orig_len * dst_sr / src_sr))
    if new_len <= 0:
        return np.zeros(0, dtype=np.float32)
    x = np.linspace(0, orig_len - 1, num=new_len, dtype=np.float64)
    xi = np.arange(orig_len, dtype=np.float64)
    return np.interp(x, xi, input_audio.astype(np.float64)).astype(np.float32)


def transcribe_utterance_float32(
    adapter: ASRModelAdapter,
    audio_float32: np.ndarray,
    *,
    sample_rate: int,
    chunk_ms: int,
) -> str:
    """
    Run Nemotron over one bounded utterance (sync). Intended for executor threads.

    Audio: mono float32, sample_rate Hz (typically 16000 from voice path).
    """
    x = np.asarray(audio_float32, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return ""
    if sample_rate != NEMOTRON_SAMPLE_RATE:
        x = _resample_linear_np(x, sample_rate, NEMOTRON_SAMPLE_RATE)

    chunk_ms_eff = max(80, int(chunk_ms))
    chunk_samples = int(NEMOTRON_SAMPLE_RATE * chunk_ms_eff / 1000)
    chunk_samples = max(chunk_samples, 1)

    state = adapter.create_session_state()
    step = 0
    last_text = ""
    pos = 0
    n = x.size

    while pos < n:
        end = min(pos + chunk_samples, n)
        chunk = x[pos:end].astype(np.float32, copy=True)
        pos = end
        is_final_chunk = pos >= n
        text, state = adapter.process_chunk(
            chunk,
            state,
            keep_all_outputs=is_final_chunk,
            step_num=step,
        )
        step += 1
        last_text = text

    return (last_text or "").strip()
