"""
Streaming STT for Live Transcript: Kyutai (kyutai/stt-1b-en_fr) when available, else Whisper fallback.

Kyutai stack (when moshi/huggingface_hub installed):
- huggingface_hub.snapshot_download: pulls kyutai/stt-1b-en_fr
- PyTorch: CUDA if available else CPU, torch.no_grad()
- moshi.models: loaders.CheckpointInfo.from_hf_repo, get_mimi, get_moshi, get_text_tokenizer
- Mimi: neural audio codec, mimi.encode(frame) -> discrete codes
- LMGen: autoregressive step, lm_gen.step(codes) -> text tokens
- text_tokenizer.id_to_piece() -> text pieces (▁ -> space)
- Frame-based: buffer until frame_size samples, then encode -> step -> emit pieces
- Flush: pad with silence (audio_delay_seconds) to force completion
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

try:
    from huggingface_hub import snapshot_download
    from moshi.models import loaders
    from moshi.models import LMGen
    import torch
    KYUTAI_AVAILABLE = True
except ImportError:
    pass

_whisper_model = None

# Kyutai model repo
KYUTAI_MODEL_NAME = "kyutai/stt-1b-en_fr"


def _pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(settings.WHISPER_MODEL)
    return _whisper_model


class KyutaiStreamingSTT:
    """
    Kyutai STT pipeline: PCM float32 (24kHz) -> frame buffer -> mimi.encode -> LMGen.step -> text pieces.
    Use add_audio(pcm_float32) and consume get_pieces(); call flush() on EOS.
    """
    def __init__(self):
        self.device = "cuda" if (KYUTAI_AVAILABLE and getattr(torch, "cuda", None) and torch.cuda.is_available()) else "cpu"
        snapshot_download(KYUTAI_MODEL_NAME)
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(KYUTAI_MODEL_NAME)
        self.mimi = checkpoint_info.get_mimi(device=self.device)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        moshi = checkpoint_info.get_moshi(device=self.device)
        self.lm_gen = LMGen(moshi, temp=0, temp_text=0)
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.text_tokenizer = checkpoint_info.get_text_tokenizer()
        self.audio_delay_seconds = checkpoint_info.stt_config.get("audio_delay_seconds", 0.5)
        self.padding_token_id = checkpoint_info.raw_config.get("text_padding_token_id", 3)
        self.sample_rate = int(getattr(self.mimi, "sample_rate", 24000))
        self.buffer = np.zeros(0, dtype=np.float32)
        self._pieces: List[str] = []

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
        self._pieces.clear()

    def add_audio(self, pcm_float32: np.ndarray) -> List[str]:
        """
        Append audio (float32, mono, sample_rate = self.sample_rate).
        Process full frames; return list of text pieces emitted this call.
        """
        self._pieces.clear()
        self.buffer = np.concatenate([self.buffer, pcm_float32]) if self.buffer.size else pcm_float32
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
                    piece = piece.replace("▁", " ")
                    if piece.strip():
                        self._pieces.append(piece)
        return self._pieces

    def flush(self) -> List[str]:
        """Pad with silence and process remaining buffer; return final pieces."""
        self._pieces.clear()
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
                    piece = piece.replace("▁", " ")
                    if piece.strip():
                        self._pieces.append(piece)
        self.reset_streaming()
        return self._pieces


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
    return stt.sample_rate if stt else 24000


async def process_audio_stream(
    pcm16_chunks: list[bytes],
    sample_rate: int,
    on_piece: Callable[[str, int], None],
    on_eos: Optional[Callable[[], None]] = None,
) -> str:
    """
    Process a stream of PCM16 chunks. Calls on_piece(text, ts_ms) for each text piece.
    Uses Kyutai if available (expect 24kHz), else Whisper.
    """
    if get_kyutai_stt() is not None:
        stt = get_kyutai_stt()
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
    # Whisper fallback
    loop = asyncio.get_running_loop()
    sr = sample_rate or settings.SAMPLE_RATE
    buffer = np.zeros(0, dtype=np.float32)
    for raw in pcm16_chunks:
        buffer = np.concatenate([buffer, _pcm16_to_float32(raw)]) if buffer.size else _pcm16_to_float32(raw)
    if buffer.size < int(0.25 * sr):
        if on_eos:
            on_eos()
        return ""
    def run_whisper():
        model = _load_whisper()
        return (model.transcribe(buffer, fp16=False).get("text", "") or "").strip()
    text = await loop.run_in_executor(None, run_whisper)
    if text:
        on_piece(text, int(time.time() * 1000))
    if on_eos:
        on_eos()
    return text


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


def process_audio_chunk_sync(
    pcm16: bytes,
    sample_rate: int,
    buffer: list,
    min_duration_sec: float = 2.0,
) -> Optional[str]:
    """Whisper fallback: buffer and transcribe when enough audio."""
    buffer.append(_pcm16_to_float32(pcm16))
    total_samples = sum(b.shape[0] for b in buffer)
    if total_samples < int(min_duration_sec * sample_rate):
        return None
    audio = np.concatenate(buffer)
    buffer.clear()
    if audio.size < int(0.25 * sample_rate):
        return None
    model = _load_whisper()
    text = (model.transcribe(audio, fp16=False).get("text", "") or "").strip()
    return text if text else None
