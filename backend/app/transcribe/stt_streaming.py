"""
Streaming STT for Live Transcript: Kyutai STT via moshi (kyutai/stt-1b-en_fr).

Uses moshi.models: loaders.CheckpointInfo.from_hf_repo, get_mimi, get_moshi, LMGen, text_tokenizer.
Frame-based: buffer -> mimi.encode -> LMGen.step -> text pieces. 24kHz, mono.
True incremental streaming: add_audio() per chunk, flush() only on EOS. No singleton; each session
gets its own instance. Audio-relative timestamps; deque buffer; efficient resampling.
"""
from __future__ import annotations
import asyncio
import os
import threading
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np
from ..core.config import settings

# Optional: disable torch compile (e.g. DGX Spark / Blackwell workaround)
for _k in ("TORCHDYNAMO_DISABLE", "TORCHINDUCTOR_DISABLE", "TORCH_COMPILE_DISABLE"):
    if _k not in os.environ:
        os.environ[_k] = "1"

KYUTAI_AVAILABLE = False
_kyutai_import_error: Optional[str] = None

try:
    from huggingface_hub import snapshot_download
    from moshi.models import loaders
    from moshi.models import LMGen
    import torch
    import torch.nn.functional as F
    KYUTAI_AVAILABLE = True
    _kyutai_import_error = None
except ImportError as e:
    _kyutai_import_error = str(e)
    F = None  # type: ignore

# Kyutai model (moshi format). Must be pre-downloaded at build time when HF_HUB_OFFLINE=1.
KYUTAI_MODEL_NAME = "kyutai/stt-1b-en_fr"
KYUTAI_SAMPLE_RATE = 24000

# Offline mode: use only local cache; never hit the network (set by docker-compose for production).
def _hub_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "yes")

# Warmup: number of real frames to run (CUDA kernels, etc.)
KYUTAI_WARMUP_FRAMES = max(4, getattr(settings, "TRANSCRIPT_STT_WARMUP_FRAMES", 8))


def _pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def _audio_rms(audio: np.ndarray) -> float:
    """Root-mean-square level of float32 mono audio."""
    if audio is None or audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _resample_fast(input_audio: np.ndarray, src_sr: int, dst_sr: int, device: str) -> np.ndarray:
    """Resample using torch F.interpolate (no torchaudio dep). GPU-capable."""
    if src_sr == dst_sr:
        return input_audio
    orig_len = len(input_audio)
    new_len = int(round(orig_len * dst_sr / src_sr))
    if new_len <= 0:
        return np.zeros(0, dtype=np.float32)
    # (1, 1, L) for 1D linear interpolation
    x = torch.from_numpy(input_audio.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    out = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return out.squeeze().cpu().numpy()


def _sliding_window_rms(audio: np.ndarray, window_samples: int, step_samples: int) -> float:
    """Max RMS over sliding windows. Used for VAD so we don't drop mixed speech+silence chunks."""
    if audio is None or audio.size == 0 or window_samples <= 0:
        return 0.0
    n = len(audio)
    if n < window_samples:
        return _audio_rms(audio)
    max_rms = 0.0
    for start in range(0, n - window_samples + 1, step_samples):
        window = audio[start : start + window_samples]
        r = _audio_rms(window)
        if r > max_rms:
            max_rms = r
    return max_rms


class KyutaiStreamingSTT:
    """
    Kyutai STT via moshi: PCM float32 (24kHz) -> frame buffer -> mimi.encode -> LMGen.step -> text pieces.
    Use add_audio(pcm_float32) per chunk and consume returned (pieces, ts_ms); call flush() only on EOS.
    Caller must call reset_streaming() when starting a new session (flush() does not reset).
    """
    def __init__(self):
        self.device = "cuda" if (KYUTAI_AVAILABLE and getattr(torch, "cuda", None) and torch.cuda.is_available()) else "cpu"
        # Offline: use only local cache; fail clearly if model missing (no silent download).
        try:
            snapshot_download(KYUTAI_MODEL_NAME, local_files_only=_hub_offline())
        except Exception as e:
            if _hub_offline():
                raise RuntimeError(
                    "Kyutai STT model not found in local cache (HF_HUB_OFFLINE=1). "
                    "Rebuild the backend image without SKIP_MODEL_DOWNLOAD=1, or run once online to populate cache."
                ) from e
            raise
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(KYUTAI_MODEL_NAME)
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        moshi_lm = checkpoint_info.get_moshi(device=self.device)
        self.lm_gen = LMGen(moshi_lm, temp=0, temp_text=0)
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.text_tokenizer = checkpoint_info.get_text_tokenizer()
        self.audio_delay_seconds = checkpoint_info.stt_config.get("audio_delay_seconds", 0.5)
        self.padding_token_id = checkpoint_info.raw_config.get("text_padding_token_id", 3)
        self.sample_rate = int(getattr(self.mimi, "sample_rate", KYUTAI_SAMPLE_RATE))
        # Deque buffer to avoid repeated concatenate/realloc
        self._buffer: deque = deque(maxlen=2 * self.frame_size * 60)  # ~2s of samples max
        # Audio-relative time: total samples fed so far
        self.audio_offset_samples = 0

        # Warmup: multiple real frames and proper CUDA sync
        with torch.no_grad():
            for _ in range(KYUTAI_WARMUP_FRAMES):
                codes = self.mimi.encode(torch.zeros(1, 1, self.frame_size, device=self.device))
                for c in range(codes.shape[-1]):
                    self.lm_gen.step(codes[:, :, c : c + 1])
            if self.device == "cuda":
                torch.cuda.synchronize()
        self.reset_streaming()

    def reset_streaming(self) -> None:
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self._buffer.clear()
        self.audio_offset_samples = 0

    def add_audio(self, pcm_float32: np.ndarray) -> Tuple[List[str], int]:
        """
        Append audio (float32, mono, sample_rate = self.sample_rate).
        Process full frames only; return (list of text pieces emitted this call, ts_ms audio-relative).
        Caller controls when to call flush() (e.g. on EOS).
        """
        pieces: List[str] = []
        ts_ms = int((self.audio_offset_samples / self.sample_rate) * 1000)
        n_in = len(pcm_float32)
        self.audio_offset_samples += n_in
        for s in pcm_float32:
            self._buffer.append(float(s))

        with torch.no_grad():
            while len(self._buffer) >= self.frame_size:
                chunk = np.array([self._buffer.popleft() for _ in range(self.frame_size)], dtype=np.float32)
                chunk_t = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(self.device)
                codes = self.mimi.encode(chunk_t)
                for c in range(codes.shape[-1]):
                    out = self.lm_gen.step(codes[:, :, c : c + 1])
                    if out is None:
                        continue
                    text_tokens = out[0] if isinstance(out, (list, tuple)) else out
                    if text_tokens is None:
                        continue
                    text_token = text_tokens[0, 0, 0].item()
                    if text_token in (0, self.padding_token_id):
                        continue
                    piece = self.text_tokenizer.id_to_piece(int(text_token))
                    if piece.startswith("▁"):
                        piece = " " + piece[1:]
                    if piece:
                        pieces.append(piece)
        return (pieces, ts_ms)

    def flush(self) -> List[str]:
        """Pad with silence, process remaining buffer, return final pieces. Does NOT reset; caller calls reset_streaming()."""
        pieces: List[str] = []
        delay_samples = int(self.audio_delay_seconds * self.sample_rate)
        for _ in range(delay_samples):
            self._buffer.append(0.0)
        self.audio_offset_samples += delay_samples

        with torch.no_grad():
            while len(self._buffer) >= self.frame_size:
                chunk = np.array([self._buffer.popleft() for _ in range(self.frame_size)], dtype=np.float32)
                chunk_t = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(self.device)
                codes = self.mimi.encode(chunk_t)
                for c in range(codes.shape[-1]):
                    out = self.lm_gen.step(codes[:, :, c : c + 1])
                    if out is None:
                        continue
                    text_tokens = out[0] if isinstance(out, (list, tuple)) else out
                    if text_tokens is None:
                        continue
                    text_token = text_tokens[0, 0, 0].item()
                    if text_token in (0, self.padding_token_id):
                        continue
                    piece = self.text_tokenizer.id_to_piece(int(text_token))
                    if piece.startswith("▁"):
                        piece = " " + piece[1:]
                    if piece:
                        pieces.append(piece)
        return pieces


# Pre-loaded instance for fast first connection (set by startup task)
_warm_stt: Optional[KyutaiStreamingSTT] = None
_warm_in_use = False
_warm_lock = threading.Lock()


def download_kyutai_model() -> bool:
    """
    Ensure Kyutai model is in local cache. In offline mode (HF_HUB_OFFLINE=1) does nothing and
    relies on build-time download; returns True only if model is already present. No runtime network.
    """
    if not KYUTAI_AVAILABLE:
        return False
    if _hub_offline():
        try:
            snapshot_download(KYUTAI_MODEL_NAME, local_files_only=True)
            return True
        except Exception:
            return False
    try:
        snapshot_download(KYUTAI_MODEL_NAME)
        return True
    except Exception:
        return False


def preload_kyutai_stt() -> bool:
    """Create one KyutaiStreamingSTT and store as warm instance. Call after download at startup. Returns True if successful."""
    global _warm_stt, _warm_in_use
    if not KYUTAI_AVAILABLE:
        return False
    try:
        with _warm_lock:
            if _warm_stt is not None:
                return True
            _warm_stt = KyutaiStreamingSTT()
            _warm_in_use = False
        return True
    except Exception:
        with _warm_lock:
            _warm_stt = None
            _warm_in_use = False
        return False


def get_or_create_kyutai_stt() -> Optional[KyutaiStreamingSTT]:
    """Return the warm instance if available (and mark in use), else create a new instance. One per WebSocket session."""
    global _warm_in_use
    if not KYUTAI_AVAILABLE:
        return None
    with _warm_lock:
        if _warm_stt is not None and not _warm_in_use:
            _warm_in_use = True
            _warm_stt.reset_streaming()
            return _warm_stt
    try:
        return KyutaiStreamingSTT()
    except Exception:
        return None


def release_kyutai_stt(stt: Optional[KyutaiStreamingSTT]) -> None:
    """If stt is the warm instance, reset and mark available. Otherwise no-op (caller may drop reference)."""
    global _warm_in_use
    if stt is None:
        return
    with _warm_lock:
        if stt is _warm_stt:
            stt.reset_streaming()
            _warm_in_use = False


def create_kyutai_stt() -> Optional[KyutaiStreamingSTT]:
    """Create a new KyutaiStreamingSTT instance. Prefer get_or_create_kyutai_stt() for WebSocket handler."""
    if not KYUTAI_AVAILABLE:
        return None
    try:
        return KyutaiStreamingSTT()
    except Exception:
        return None


def kyutai_sample_rate() -> int:
    """Model sample rate (24kHz for Kyutai)."""
    return KYUTAI_SAMPLE_RATE


async def process_audio_stream(
    pcm16_chunks: list[bytes],
    sample_rate: int,
    on_piece: Callable[[str, int], None],
    on_eos: Optional[Callable[[], None]] = None,
) -> str:
    """
    Process a stream of PCM16 chunks incrementally: add_audio per chunk, flush only at end.
    Calls on_piece(text, ts_ms) for each text piece. ts_ms is audio-relative.
    """
    stt = create_kyutai_stt()
    if stt is None:
        raise RuntimeError(
            "Kyutai STT not available. Install: pip install moshi huggingface-hub. "
            "Requires PyTorch. On ARM64: apt install libopus-dev for sphn build."
        )
    stt.reset_streaming()
    loop = asyncio.get_running_loop()
    all_pieces: List[str] = []
    for raw in pcm16_chunks:
        f32 = _pcm16_to_float32(raw)
        if sample_rate != stt.sample_rate:
            f32 = _resample_fast(f32, sample_rate, stt.sample_rate, stt.device)
        pieces, ts_ms = stt.add_audio(f32)
        for p in pieces:
            on_piece(p, ts_ms)
            all_pieces.append(p)
    # EOS: flush only at end; ts at start of flush
    ts_flush_ms = int((stt.audio_offset_samples / stt.sample_rate) * 1000)
    flush_pieces = stt.flush()
    for p in flush_pieces:
        on_piece(p, ts_flush_ms)
        all_pieces.append(p)
    stt.reset_streaming()
    if on_eos:
        on_eos()
    return " ".join(all_pieces)
