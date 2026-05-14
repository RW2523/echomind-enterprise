"""
Streaming STT for Live Transcript: NVIDIA Nemotron speech (NeMo), 16 kHz mono.
Shared ASRModelAdapter; per-WebSocket NemotronStreamContext (encoder cache + audio buffer).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..core.config import settings

logger = logging.getLogger(__name__)

for _k in ("TORCHDYNAMO_DISABLE", "TORCHINDUCTOR_DISABLE", "TORCH_COMPILE_DISABLE"):
    if _k not in os.environ:
        os.environ[_k] = "1"

NEMOTRON_AVAILABLE = False
_nemotron_import_error: Optional[str] = None

try:
    from .asr_model_adapter import ASRModelAdapter
    from nemotron_asr.adapter import _resolve_local_nemo_checkpoint

    import nemo.collections.asr  # noqa: F401 — fail fast when NeMo not installed

    NEMOTRON_AVAILABLE = True
    _nemotron_import_error = None
except ImportError as e:
    NEMOTRON_AVAILABLE = False
    _nemotron_import_error = str(e)

NEMOTRON_MODEL_NAME = os.getenv(
    "ECHOMIND_ASR_MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b"
)
BOARD_ROOM_ASR_MODEL_NAME = os.getenv(
    "BOARD_ROOM_ASR_MODEL_NAME", "nvidia/multitalker-parakeet-streaming-0.6b-v1"
)
NEMOTRON_SAMPLE_RATE = 16000
NEMOTRON_ATT_CONTEXT_RIGHT = int(os.getenv("ECHOMIND_ASR_ATT_CONTEXT_RIGHT", "6"))

_asr_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_adapter_lock = threading.Lock()
_shared_adapter: Optional["ASRModelAdapter"] = None
_board_room_adapter: Optional["ASRModelAdapter"] = None
_board_room_load_error: Optional[str] = None
# One Nemotron forward / session-state build at a time process-wide (shared weights + NeMo streaming).
_nemotron_process_lock: Optional[asyncio.Lock] = None


def get_nemotron_process_lock() -> asyncio.Lock:
    global _nemotron_process_lock
    if _nemotron_process_lock is None:
        _nemotron_process_lock = asyncio.Lock()
    return _nemotron_process_lock


def _hub_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "yes")


def _get_asr_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _asr_executor
    if _asr_executor is None:
        _asr_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, getattr(settings, "TRANSCRIPT_ASR_EXECUTOR_WORKERS", 1)),
            thread_name_prefix="nemotron_asr",
        )
    return _asr_executor


def _pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def _audio_rms(audio: np.ndarray) -> float:
    if audio is None or audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _resample_fast(input_audio: np.ndarray, src_sr: int, dst_sr: int, device: str) -> np.ndarray:
    if src_sr == dst_sr:
        return input_audio
    orig_len = len(input_audio)
    new_len = int(round(orig_len * dst_sr / src_sr))
    if new_len <= 0:
        return np.zeros(0, dtype=np.float32)
    x = torch.from_numpy(input_audio.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    out = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return out.squeeze().cpu().numpy()


def _sliding_window_rms(audio: np.ndarray, window_samples: int, step_samples: int) -> float:
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


def _hypothesis_delta(prev: str, curr: str) -> str:
    prev = (prev or "").strip()
    curr = (curr or "").strip()
    if not curr:
        return ""
    if not prev:
        return curr
    if curr.startswith(prev):
        return curr[len(prev) :]
    if len(curr) < len(prev):
        # Streaming ASR retracted unstable tail; session sync uses the full hypothesis instead.
        return ""
    n = min(len(prev), len(curr))
    i = 0
    while i < n and prev[i] == curr[i]:
        i += 1
    return curr[i:]


def _max_suffix_prefix_overlap(tail: str, incoming: str, k: int) -> int:
    if not tail or not incoming or k <= 0:
        return 0
    n = min(k, len(tail), len(incoming))
    for length in range(n, 0, -1):
        if tail[-length:] == incoming[:length]:
            return length
    return 0


def _merge_streaming_hypothesis(display: str, step_prev: str, step_curr: str) -> str:
    """Accumulate per-step ASR text into one monotonic transcript for the session."""
    step_curr = (step_curr or "").strip()
    if not step_curr:
        return display
    if not display:
        return step_curr
    if step_curr.startswith(display):
        return step_curr
    if display.startswith(step_curr):
        return display
    step_prev = (step_prev or "").strip()
    if step_prev and step_curr.startswith(step_prev):
        extension = step_curr[len(step_prev) :]
        if extension and display.rstrip().endswith(step_prev):
            return display + extension
    overlap = _max_suffix_prefix_overlap(display.rstrip(), step_curr.lstrip(), 80)
    if overlap:
        addition = step_curr[overlap:]
    else:
        addition = step_curr
        if not display.endswith(" ") and addition and not addition.startswith(" "):
            addition = " " + addition
    if not addition or display.rstrip().endswith(addition.strip()):
        return display
    return display + addition


def get_shared_asr_adapter() -> "ASRModelAdapter":
    return get_asr_adapter("default")


def get_asr_adapter(profile: str = "default") -> "ASRModelAdapter":
    global _shared_adapter, _board_room_adapter, _board_room_load_error
    if not NEMOTRON_AVAILABLE:
        raise RuntimeError(
            "Nemotron ASR not available. Install NeMo ASR toolkit (see backend Dockerfile). "
            + (_nemotron_import_error or "")
        )
    prof = (profile or "default").strip().lower()
    with _adapter_lock:
        if prof in ("board_room", "board-room", "multitalker", "parakeet"):
            if _board_room_adapter is not None:
                return _board_room_adapter
            try:
                adapter = ASRModelAdapter(
                    model_name=BOARD_ROOM_ASR_MODEL_NAME,
                    att_context_right=NEMOTRON_ATT_CONTEXT_RIGHT,
                )
                adapter.load()
                _board_room_adapter = adapter
                _board_room_load_error = None
                return adapter
            except Exception as exc:
                _board_room_load_error = str(exc)
                logger.warning(
                    "Board Room ASR %s unavailable (%s); using %s",
                    BOARD_ROOM_ASR_MODEL_NAME,
                    exc,
                    NEMOTRON_MODEL_NAME,
                )
                if _shared_adapter is None:
                    _shared_adapter = ASRModelAdapter(
                        model_name=NEMOTRON_MODEL_NAME,
                        att_context_right=NEMOTRON_ATT_CONTEXT_RIGHT,
                    )
                    _shared_adapter.load()
                _board_room_adapter = _shared_adapter
                return _board_room_adapter
        if _shared_adapter is None:
            _shared_adapter = ASRModelAdapter(
                model_name=NEMOTRON_MODEL_NAME,
                att_context_right=NEMOTRON_ATT_CONTEXT_RIGHT,
            )
            _shared_adapter.load()
        return _shared_adapter


def download_nemotron_model() -> bool:
    if not NEMOTRON_AVAILABLE:
        return False
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(NEMOTRON_MODEL_NAME, local_files_only=_hub_offline())
        return True
    except Exception:
        return not _hub_offline()


def download_board_room_model() -> bool:
    if not NEMOTRON_AVAILABLE:
        return False
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(BOARD_ROOM_ASR_MODEL_NAME, local_files_only=_hub_offline())
        return True
    except Exception:
        return not _hub_offline()


def preload_nemotron_stt() -> bool:
    try:
        get_asr_adapter("default")
        return True
    except Exception:
        return False


def preload_board_room_stt() -> bool:
    try:
        get_asr_adapter("board_room")
        return True
    except Exception:
        return False


def board_room_stt_status() -> dict:
    loaded = _board_room_adapter is not None
    cached = _resolve_local_nemo_checkpoint(BOARD_ROOM_ASR_MODEL_NAME) is not None
    using_fallback = bool(
        loaded
        and _shared_adapter is not None
        and _board_room_adapter is _shared_adapter
        and _board_room_load_error
    )
    return {
        "available": NEMOTRON_AVAILABLE,
        "model_name": BOARD_ROOM_ASR_MODEL_NAME,
        "loaded": loaded,
        "cached": cached,
        "using_fallback": using_fallback,
        "fallback_model_name": NEMOTRON_MODEL_NAME if using_fallback else None,
        "load_error": _board_room_load_error,
        "import_error": _nemotron_import_error,
    }


def _run_chunk_sync(ctx: "NemotronStreamContext", audio: np.ndarray, keep_all: bool) -> Tuple[str, int]:
    ts_ms = int((ctx.audio_offset_samples / NEMOTRON_SAMPLE_RATE) * 1000)
    ctx.audio_offset_samples += len(audio)
    step = ctx.stream_step_num
    ctx.stream_step_num += 1
    text, ctx.state = ctx.adapter.process_chunk(
        audio, ctx.state, keep_all_outputs=keep_all, step_num=step
    )
    return text.strip(), ts_ms


class NemotronStreamContext:
    """Per-session streaming buffer + NeMo cache state."""

    def __init__(self, adapter: ASRModelAdapter):
        self.adapter = adapter
        self.reset()

    def reset(self) -> None:
        self.state = self.adapter.create_session_state()
        self.buffer: deque = deque()
        self.buffer_samples = 0
        self.stream_step_num = 0
        self.audio_offset_samples = 0
        self.last_hypothesis = ""
        self.merged_hypothesis = ""

    def _chunk_samples(self) -> int:
        ms = max(80, getattr(settings, "TRANSCRIPT_NEMOTRON_CHUNK_MS", 560))
        return int(NEMOTRON_SAMPLE_RATE * ms / 1000)

    def _take_chunk(self) -> Optional[np.ndarray]:
        needed = self._chunk_samples()
        if self.buffer_samples < needed:
            return None
        chunks: List[np.ndarray] = []
        while self.buffer and sum(len(c) for c in chunks) < needed:
            c = self.buffer.popleft()
            self.buffer_samples -= len(c)
            chunks.append(c)
        if not chunks:
            return None
        combined = np.concatenate(chunks).astype(np.float32)
        if len(combined) > needed:
            audio = combined[:needed]
            remainder = combined[needed:]
            self.buffer.appendleft(remainder)
            self.buffer_samples += len(remainder)
        else:
            audio = combined
        return audio

    def append_samples(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self.buffer.append(samples.astype(np.float32, copy=False))
        self.buffer_samples += len(samples)

    async def process_pcm_chunk(
        self,
        pcm_float32: np.ndarray,
        client_sr: int,
        loop,
        device: str,
    ) -> bool:
        """Resample to 16 kHz, buffer, run ASR steps. Returns True when transcript text changed."""
        if client_sr != NEMOTRON_SAMPLE_RATE:
            # Resample on CPU only — GPU interpolate from asyncio + thread pool races the shared CUDA context.
            pcm_float32 = _resample_fast(pcm_float32, client_sr, NEMOTRON_SAMPLE_RATE, "cpu")
        self.append_samples(pcm_float32)
        before = self.merged_hypothesis
        ex = _get_asr_executor()
        while True:
            audio = self._take_chunk()
            if audio is None:
                break
            audio_copy = audio.copy()
            text, _ts_ms = await loop.run_in_executor(
                ex,
                lambda ac=audio_copy: _run_chunk_sync(self, ac, False),
            )
            if text:
                self.merged_hypothesis = _merge_streaming_hypothesis(
                    self.merged_hypothesis, self.last_hypothesis, text
                )
                self.last_hypothesis = text
        return self.merged_hypothesis != before

    async def flush(self, loop) -> bool:
        """Process remainder with keep_all_outputs=True (end of stream)."""
        before = self.merged_hypothesis
        ex = _get_asr_executor()
        if self.buffer_samples > 0:
            chunks = list(self.buffer)
            self.buffer.clear()
            self.buffer_samples = 0
            audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
            if audio.size > 0:
                audio_copy = audio.copy()
                text, _ts_ms = await loop.run_in_executor(
                    ex,
                    lambda ac=audio_copy: _run_chunk_sync(self, ac, True),
                )
                if text:
                    self.merged_hypothesis = _merge_streaming_hypothesis(
                        self.merged_hypothesis, self.last_hypothesis, text
                    )
                    self.last_hypothesis = text
        return self.merged_hypothesis != before


async def process_audio_stream(
    pcm16_chunks: list[bytes],
    sample_rate: int,
    on_piece: Callable[[str, int], None],
    on_eos: Optional[Callable[[], None]] = None,
) -> str:
    """Offline helper: run Nemotron over PCM16 chunks (optional tests / tools)."""
    import asyncio

    adapter = get_shared_asr_adapter()
    ctx = NemotronStreamContext(adapter)
    loop = asyncio.get_running_loop()
    device = adapter.device
    all_text: List[str] = []
    for raw in pcm16_chunks:
        f32 = _pcm16_to_float32(raw)
        await ctx.process_pcm_chunk(f32, sample_rate, loop, device)
    await ctx.flush(loop)
    if on_eos:
        on_eos()
    return ctx.merged_hypothesis
