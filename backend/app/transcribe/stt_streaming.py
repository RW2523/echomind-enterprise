"""
Streaming STT for Live Transcript: Kyutai STT via moshi (kyutai/stt-1b-en_fr).

Uses moshi.models: loaders.CheckpointInfo.from_hf_repo, get_mimi, get_moshi, LMGen, text_tokenizer.
Frame-based: buffer -> mimi.encode -> LMGen.step -> text pieces. 24kHz, mono.
"""
from __future__ import annotations
import asyncio
import os
import time
import numpy as np
from typing import Callable, Optional, List

from ..core.config import settings

# Optional: disable torch compile (e.g. DGX Spark / Blackwell workaround)
for _k in ("TORCHDYNAMO_DISABLE", "TORCHINDUCTOR_DISABLE", "TORCH_COMPILE_DISABLE"):
    if _k not in os.environ:
        os.environ[_k] = "1"

KYUTAI_AVAILABLE = False
_kyutai_stt = None  # KyutaiStreamingSTT singleton
_kyutai_import_error: Optional[str] = None

try:
    from huggingface_hub import snapshot_download
    from moshi.models import loaders
    from moshi.models import LMGen
    import torch
    KYUTAI_AVAILABLE = True
    _kyutai_import_error = None
except ImportError as e:
    _kyutai_import_error = str(e)

# Kyutai model (moshi format)
KYUTAI_MODEL_NAME = "kyutai/stt-1b-en_fr"
KYUTAI_SAMPLE_RATE = 24000


def _pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def _audio_rms(audio: np.ndarray) -> float:
    """Root-mean-square level of float32 mono audio. Used for VAD to skip silence/noise."""
    if audio is None or audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _resample_linear(input: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return input
    ratio = dst_sr / src_sr
    n = int(len(input) * ratio)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        x = i / ratio
        i0 = min(int(x), len(input) - 1)
        i1 = min(i0 + 1, len(input) - 1)
        w = x - i0
        out[i] = (1 - w) * input[i0] + w * input[i1]
    return out


class KyutaiStreamingSTT:
    """
    Kyutai STT via moshi: PCM float32 (24kHz) -> frame buffer -> mimi.encode -> LMGen.step -> text pieces.
    Use add_audio(pcm_float32) and consume returned pieces; call flush() on EOS.
    """
    def __init__(self):
        self.device = "cuda" if (KYUTAI_AVAILABLE and getattr(torch, "cuda", None) and torch.cuda.is_available()) else "cpu"
        # Pre-download model (required by moshi)
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
        self.buffer = np.zeros(0, dtype=np.float32)

        # Warmup
        with torch.no_grad():
            for _ in range(2):
                codes = self.mimi.encode(torch.zeros(1, 1, self.frame_size, device=self.device))
                for c in range(codes.shape[-1]):
                    self.lm_gen.step(codes[:, :, c : c + 1])
            if self.device == "cuda":
                torch.cuda.synchronize()

    def reset_streaming(self):
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.buffer = np.zeros(0, dtype=np.float32)

    def add_audio(self, pcm_float32: np.ndarray) -> List[str]:
        """
        Append audio (float32, mono, sample_rate = self.sample_rate).
        Process full frames; return list of text pieces emitted this call.
        """
        pieces: List[str] = []
        self.buffer = np.concatenate([self.buffer, pcm_float32]) if self.buffer.size else pcm_float32.copy()

        with torch.no_grad():
            while self.buffer.shape[0] >= self.frame_size:
                chunk = self.buffer[:self.frame_size].copy()
                self.buffer = self.buffer[self.frame_size:]
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
                    # SentencePiece: ▁ = word boundary. Only add space before new words, not between subwords.
                    if piece.startswith("▁"):
                        piece = " " + piece[1:]
                    if piece:
                        pieces.append(piece)
        return pieces

    def flush(self) -> List[str]:
        """Pad with silence and process remaining buffer; return final pieces."""
        pieces: List[str] = []
        delay_samples = int(self.audio_delay_seconds * self.sample_rate)
        padding = np.zeros(delay_samples, dtype=np.float32)
        self.buffer = np.concatenate([self.buffer, padding]) if self.buffer.size else padding

        with torch.no_grad():
            while self.buffer.shape[0] >= self.frame_size:
                chunk = self.buffer[:self.frame_size].copy()
                self.buffer = self.buffer[self.frame_size:]
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
                    # SentencePiece: ▁ = word boundary. Only add space before new words, not between subwords.
                    if piece.startswith("▁"):
                        piece = " " + piece[1:]
                    if piece:
                        pieces.append(piece)
        self.reset_streaming()
        return pieces


def get_kyutai_stt() -> Optional[KyutaiStreamingSTT]:
    """Load Kyutai STT once; return None if not available."""
    global _kyutai_stt
    if not KYUTAI_AVAILABLE:
        return None
    if _kyutai_stt is None:
        try:
            _kyutai_stt = KyutaiStreamingSTT()
        except Exception:
            _kyutai_stt = None
    return _kyutai_stt


def kyutai_sample_rate() -> int:
    """Model sample rate (24kHz for Kyutai)."""
    stt = get_kyutai_stt()
    return stt.sample_rate if stt else KYUTAI_SAMPLE_RATE


async def process_audio_stream(
    pcm16_chunks: list[bytes],
    sample_rate: int,
    on_piece: Callable[[str, int], None],
    on_eos: Optional[Callable[[], None]] = None,
) -> str:
    """
    Process a stream of PCM16 chunks. Calls on_piece(text, ts_ms) for each text piece.
    Kyutai STT via moshi (24kHz). Resamples if needed.
    """
    stt = get_kyutai_stt()
    if stt is None:
        raise RuntimeError(
            "Kyutai STT not available. Install: pip install moshi huggingface-hub. "
            "Requires PyTorch. On ARM64: apt install libopus-dev for sphn build."
        )
    stt.reset_streaming()
    loop = asyncio.get_running_loop()
    full = []
    for raw in pcm16_chunks:
        f32 = _pcm16_to_float32(raw)
        if sample_rate != stt.sample_rate:
            f32 = _resample_linear(f32, sample_rate, stt.sample_rate)
        full.append(f32)
    if not full:
        if on_eos:
            on_eos()
        return ""
    audio = np.concatenate(full)
    ts_ms = int(time.time() * 1000)

    def run():
        pieces = stt.add_audio(audio)
        pieces.extend(stt.flush())
        return pieces

    pieces = await loop.run_in_executor(None, run)
    for p in pieces:
        on_piece(p, ts_ms)
    if on_eos:
        on_eos()
    return " ".join(pieces)
