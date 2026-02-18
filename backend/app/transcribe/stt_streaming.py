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

# Kyutai model (moshi format)
KYUTAI_MODEL_NAME = "kyutai/stt-1b-en_fr"
KYUTAI_SAMPLE_RATE = 24000

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
        snapshot_download(KYUTAI_MODEL_NAME)
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


def create_kyutai_stt() -> Optional[KyutaiStreamingSTT]:
    """Create a new KyutaiStreamingSTT instance. One per WebSocket session; do not share."""
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
