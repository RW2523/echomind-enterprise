"""Audio decoding and resampling for incoming WebSocket chunks."""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_DTYPE = np.float32


def raw_pcm_to_float(pcm: bytes, sample_width: int = 2) -> np.ndarray:
    """Convert raw PCM bytes to float32 mono. Assumes little-endian."""
    if sample_width == 2:
        arr = np.frombuffer(pcm, dtype=np.int16)
    else:
        arr = np.frombuffer(pcm, dtype=np.int8)
    return arr.astype(np.float32) / (2 ** (sample_width * 8 - 1))


def ensure_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Downmix to mono if multichannel."""
    if channels == 1:
        return samples
    samples = samples.reshape(-1, channels)
    return samples.mean(axis=1).astype(np.float32)


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample to target sample rate. Uses librosa when available for better accuracy."""
    if orig_sr == target_sr:
        return samples
    try:
        import librosa
        return librosa.resample(samples.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_fast").astype(np.float32)
    except ImportError:
        duration = len(samples) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(samples) - 1, target_len, endpoint=True)
        return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


def process_audio_chunk(
    payload: bytes,
    *,
    sample_rate: Optional[int] = None,
    channels: int = 1,
    sample_width: int = 2,
) -> Tuple[np.ndarray, int]:
    """
    Decode WebSocket audio payload to float32 mono at TARGET_SAMPLE_RATE.
    Returns (samples, actual_sample_rate). Caller can rely on actual_sr == TARGET_SAMPLE_RATE.
    """
    sr = sample_rate or TARGET_SAMPLE_RATE
    samples = raw_pcm_to_float(payload, sample_width=sample_width)
    samples = ensure_mono(samples, channels)
    samples = resample(samples, sr, TARGET_SAMPLE_RATE)
    return samples, TARGET_SAMPLE_RATE
