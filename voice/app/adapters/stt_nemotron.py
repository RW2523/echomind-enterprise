"""
Voice path STT: NVIDIA Nemotron (NeMo) utterance-final transcription.
Uses shared nemotron_asr package (same adapter as backend live transcript).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
from typing import Optional

import numpy as np

from ..config import SETTINGS

logger = logging.getLogger(__name__)

for _k in ("TORCHDYNAMO_DISABLE", "TORCHINDUCTOR_DISABLE", "TORCH_COMPILE_DISABLE"):
    if _k not in os.environ:
        os.environ[_k] = "1"

_adapter_lock = threading.Lock()
_shared_adapter: Optional[object] = None
_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

# ── GPU/CUDA fatal-fault detection ───────────────────────────────────────────
# A CUDA error (e.g. cudaErrorUnknown) poisons the process CUDA context so that
# EVERY subsequent inference fails. There is no in-process recovery — the container
# must restart. We flag it here so /health can report unhealthy and a watchdog can
# exit the process for the orchestrator (restart: unless-stopped) to recreate it.
_stt_fatal = threading.Event()
_FATAL_HINTS = (
    "cuda", "cublas", "cudnn", "cudaerror", "device-side assert",
    "illegal memory access", "no kernel image", "misaligned address",
)


def is_fatal_gpu_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return any(h in msg for h in _FATAL_HINTS)


def note_stt_error(e: BaseException) -> bool:
    """Mark STT unrecoverable if the error indicates a poisoned CUDA context. Returns True if fatal."""
    if is_fatal_gpu_error(e):
        if not _stt_fatal.is_set():
            logger.critical(
                "Voice Nemotron: FATAL GPU/CUDA error — context poisoned, STT marked UNHEALTHY "
                "(container restart required): %s", e,
            )
            _stt_fatal.set()
        return True
    return False


def stt_healthy() -> bool:
    return not _stt_fatal.is_set()


def stt_fatal_event() -> threading.Event:
    return _stt_fatal


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        workers = max(1, int(os.getenv("VOICE_NEMOTRON_EXECUTOR_WORKERS", "1")))
        _executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="voice_nemotron_stt",
        )
    return _executor


def get_shared_asr_adapter():
    """Process-wide shared Nemotron weights (matches backend single-model assumption)."""
    global _shared_adapter
    from nemotron_asr.adapter import ASRModelAdapter

    with _adapter_lock:
        if _shared_adapter is None:
            model = SETTINGS.NEMOTRON_MODEL_NAME
            att = SETTINGS.NEMOTRON_ATT_CONTEXT_RIGHT
            logger.info("Voice Nemotron: loading model=%s att_context_right=%s", model, att)
            t0 = time.monotonic()
            ad = ASRModelAdapter(model_name=model, att_context_right=att)
            ad.load()
            logger.info(
                "Voice Nemotron: model ready device=%s load_wall_s=%.2f",
                getattr(ad, "device", "?"),
                time.monotonic() - t0,
            )
            _shared_adapter = ad
        return _shared_adapter


def ensure_nemotron_loaded_at_startup() -> None:
    """Load at process start only when VOICE_NEMOTRON_STARTUP_LOAD=1. Default 0: lazy load on first transcribe."""
    if os.getenv("VOICE_NEMOTRON_STARTUP_LOAD", "0").strip().lower() not in ("1", "true", "yes"):
        logger.warning("Voice Nemotron: startup load skipped (VOICE_NEMOTRON_STARTUP_LOAD=0)")
        return
    try:
        get_shared_asr_adapter()
    except Exception as e:
        logger.exception("Voice Nemotron: FATAL startup load failed: %s", e)
        raise RuntimeError(
            "Nemotron STT failed to load. Install NeMo ASR in the voice image, set HF_TOKEN if needed, "
            "and ensure model weights are available (build-time download or HF cache volume). "
            f"Detail: {e}"
        ) from e


# ── Accurate final-decode model (Parakeet) ───────────────────────────────────
# The streaming Nemotron-0.6B is tuned for low-latency partials, not accuracy
# (it mangles domain terms, e.g. "DoD FMR" -> "DODFMA", "B visa" -> "Pizza").
# For the FINAL transcript that feeds the LLM we use a more accurate NeMo model
# (Parakeet-TDT) when VOICE_USE_PARAKEET=1. Live on-screen partials stay on the
# streaming model. Any failure falls back to the streaming model automatically.
_parakeet_lock = threading.Lock()
_parakeet_model: Optional[object] = None
_parakeet_device: str = "cpu"


def _parakeet_fallback_to_cpu(reason: str) -> None:
    """Parakeet is optional (the streaming model can produce the final). A GPU failure in ITS
    path must not exit the whole service via the CUDA watchdog: reload it on CPU instead."""
    global _parakeet_model, _parakeet_device
    logger.warning("Voice Parakeet: falling back to CPU (%s)", reason)
    os.environ["VOICE_PARAKEET_DEVICE"] = "cpu"
    with _parakeet_lock:
        _parakeet_model = None
        _parakeet_device = "cpu"


def _parakeet_enabled() -> bool:
    return os.getenv("VOICE_USE_PARAKEET", "0").strip().lower() in ("1", "true", "yes")


def _get_parakeet():
    global _parakeet_model
    with _parakeet_lock:
        if _parakeet_model is None:
            from nemo.collections.asr.models import ASRModel
            name = os.getenv("VOICE_FINAL_ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
            logger.info("Voice Parakeet: loading final-decode model=%s", name)
            t0 = time.monotonic()
            m = ASRModel.from_pretrained(name)
            # GPU by default, with Parakeet's CUDA-graph decoder DISABLED. The historic fatal
            # "illegal memory access" was not GPU-vs-TRT-LLM: it was Parakeet's full-graph label-
            # looping decoder replaying on one executor thread while the Nemotron streaming model's
            # own CUDA-graph decoder replayed on another, in this same process. Measured on the
            # GB10 with graphs off: 100/100 clean turns while Nemotron streams and TRT-LLM generates,
            # ~170 ms per utterance versus ~1.5 s on the CPU path. A second, warm-up-only race in
            # NeMo's shared Triton autotuner is closed by fuse_triton=False plus the sequential
            # warm-up in ensure_parakeet_loaded_at_startup(). VOICE_PARAKEET_DEVICE=cpu restores the
            # old behaviour; VOICE_PARAKEET_CUDA_GRAPHS=1 re-enables graphs (not recommended).
            dev = os.getenv("VOICE_PARAKEET_DEVICE", "cuda").strip().lower()
            if dev == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA not available")
                    m = m.to("cuda")
                    if os.getenv("VOICE_PARAKEET_CUDA_GRAPHS", "0").strip().lower() not in ("1", "true", "yes"):
                        _disable_cuda_graphs(m)
                    try:
                        m.encoder.pre_encode.conv.fuse_triton = False
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning("Voice Parakeet: GPU setup failed (%s); using CPU", e)
                    dev = "cpu"
                    m = m.to("cpu")
            else:
                try:
                    m = m.to("cpu")
                except Exception:
                    pass
            m.eval()
            global _parakeet_device
            _parakeet_device = dev
            logger.info("Voice Parakeet: ready device=%s load_wall_s=%.2f", dev, time.monotonic() - t0)
            _parakeet_model = m
        return _parakeet_model


def _disable_cuda_graphs(m) -> None:
    """Switch Parakeet to the eager label-looping decoder (keeps loop_labels, drops the graph)."""
    import copy
    from omegaconf import OmegaConf
    dc = copy.deepcopy(m.cfg.decoding)
    OmegaConf.set_struct(dc, False)
    dc.greedy.use_cuda_graph_decoder = False
    m.change_decoding_strategy(dc)
    mode = getattr(getattr(getattr(m.decoding, "decoding", None), "decoding_computer", None), "cuda_graphs_mode", "?")
    logger.info("Voice Parakeet: CUDA graphs disabled (decoder cuda_graphs_mode=%s)", mode)
    if mode not in (None, "?"):
        # Never run the crash-prone configuration silently: the caller falls back to CPU.
        raise RuntimeError(f"CUDA graphs still active after disable (mode={mode})")


def _transcribe_final_parakeet(audio_f32: np.ndarray, sample_rate: int) -> str:
    try:
        return _transcribe_final_parakeet_impl(audio_f32, sample_rate)
    except Exception as e:
        if _parakeet_device == "cuda" and is_fatal_gpu_error(e) and "illegal memory" not in str(e).lower():
            # OOM / kernel / driver hiccup confined to Parakeet: retry once on CPU. An illegal
            # memory access poisons the shared context and is left to the watchdog.
            _parakeet_fallback_to_cpu(str(e)[:160])
            return _transcribe_final_parakeet_impl(audio_f32, sample_rate)
        raise


def _transcribe_final_parakeet_impl(audio_f32: np.ndarray, sample_rate: int) -> str:
    import tempfile
    import wave
    m = _get_parakeet()
    if int(sample_rate) == 16000:
        # NeMo >= 2 accepts the float32 array directly: no temp wav write/read per turn.
        try:
            out = m.transcribe([np.ascontiguousarray(audio_f32, dtype=np.float32)], batch_size=1, verbose=False)
            h = out[0] if out else ""
            if isinstance(h, (list, tuple)):
                h = h[0] if h else ""
            text = getattr(h, "text", h)
            return (text if isinstance(text, str) else str(text or "")).strip()
        except TypeError:
            pass                                   # older NeMo: fall through to the wav path
    pcm16 = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype("<i2")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sample_rate))
            w.writeframes(pcm16.tobytes())
        try:
            out = m.transcribe([path], batch_size=1, verbose=False)
        except TypeError:
            out = m.transcribe([path])
        h = out[0] if out else ""
        if isinstance(h, (list, tuple)):
            h = h[0] if h else ""
        text = getattr(h, "text", h)
        return (text if isinstance(text, str) else str(text or "")).strip()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def ensure_parakeet_loaded_at_startup() -> None:
    """Pre-warm the Parakeet final-decode model so the first utterance isn't slow. Non-fatal:
    on failure the streaming model handles the final decode (fallback in transcribe()).
    The warm-up transcribe runs here, BEFORE any session exists, so Triton's first-call autotune
    never overlaps a Nemotron streaming step."""
    if not _parakeet_enabled():
        return
    try:
        _get_parakeet()
        t0 = time.monotonic()
        warm = np.zeros(int(SETTINGS.SR * 1.2), dtype=np.float32)
        warm[::400] = 0.05                           # not pure silence: exercises the full decoder path
        _transcribe_final_parakeet(warm, SETTINGS.SR)
        logger.info("Voice Parakeet: warm-up transcribe done in %.0f ms", (time.monotonic() - t0) * 1000)
    except Exception as e:
        if os.getenv("VOICE_PARAKEET_DEVICE", "cuda").strip().lower() == "cuda":
            _parakeet_fallback_to_cpu(f"warm-up failed: {str(e)[:160]}")
            try:
                _get_parakeet()
                _transcribe_final_parakeet(np.zeros(int(SETTINGS.SR * 0.6), dtype=np.float32), SETTINGS.SR)
                logger.info("Voice Parakeet: CPU fallback ready")
                return
            except Exception as e2:
                e = e2
        logger.warning("Voice Parakeet: startup pre-warm failed (%s); will lazy-load / fall back", e)


class NemotronUtteranceSTT:
    """Async-friendly utterance STT for OmniSessionA (float32 mono, sample rate from SETTINGS.SR)."""

    def __init__(self):
        self.sample_rate = SETTINGS.SR
        self.chunk_ms = SETTINGS.NEMOTRON_CHUNK_MS

    async def transcribe(self, audio_f32: np.ndarray) -> str:
        from nemotron_asr.utterance import transcribe_utterance_float32

        adapter = get_shared_asr_adapter()
        ex = _get_executor()
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        def _run() -> str:
            if _parakeet_enabled():
                try:
                    return _transcribe_final_parakeet(audio_f32, self.sample_rate)
                except Exception as e:
                    note_stt_error(e)
                    logger.warning(
                        "Voice Parakeet final-decode failed (%s); falling back to streaming model", e
                    )
            return transcribe_utterance_float32(
                adapter,
                audio_f32,
                sample_rate=self.sample_rate,
                chunk_ms=self.chunk_ms,
            )

        try:
            text = await loop.run_in_executor(ex, _run)
        except Exception as e:
            note_stt_error(e)  # flag unhealthy if this is a poisoned-CUDA fault
            logger.exception("Voice Nemotron: transcribe failed: %s", e)
            raise
        ms = (time.monotonic() - t0) * 1000.0
        preview = (text or "")[:120]
        logger.info(
            "Voice Nemotron: transcribe done latency_ms=%.0f chars=%d text_preview=%r",
            ms,
            len(text or ""),
            preview,
        )
        return text or ""


class NemotronStreamingSTT:
    """
    Per-utterance streaming STT using Nemotron's process_chunk API.

    Feed audio frames while the user is still speaking to get a growing
    partial transcript. This lets us classify intent (knowledge-intensive vs.
    casual) before the utterance ends, so we can pre-warm RAG and pick the
    right lead phrase without any extra latency.

    Thread-safety: push_chunk() is designed to be called from a single thread
    (the executor thread in _consume_loop). reset() must be called between
    utterances.
    """

    def __init__(self):
        self.sample_rate = SETTINGS.SR
        self._lock = threading.Lock()      # push_chunk / flush / reset never overlap
        # Nemotron streaming chunk: 560ms default, same as live transcript
        frame_ms = int(os.getenv("VOICE_STREAMING_CHUNK_MS", "560"))
        self._frame_samples = int(self.sample_rate * frame_ms / 1000)
        self._buf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._state = None
        self._step: int = 0
        self._last_hyp: str = ""

    def reset(self) -> None:
        """Call at the start of each new utterance."""
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
            self._state = None
            self._step = 0
            self._last_hyp = ""

    def push_chunk(self, audio_f32: np.ndarray) -> str:
        """
        Accumulate audio and process full Nemotron frames synchronously.
        Returns the latest partial hypothesis (may be empty string).
        Safe to call from a thread-pool executor.
        """
        with self._lock:
            return self._push_chunk_locked(audio_f32)

    def _push_chunk_locked(self, audio_f32: np.ndarray) -> str:
        try:
            adapter = get_shared_asr_adapter()
            if self._state is None:
                self._state = adapter.create_session_state()

            self._buf = np.concatenate([self._buf, audio_f32])

            while len(self._buf) >= self._frame_samples:
                frame = self._buf[: self._frame_samples]
                self._buf = self._buf[self._frame_samples :]
                try:
                    hyp, self._state = adapter.process_chunk(
                        frame,
                        self._state,
                        keep_all_outputs=False,
                        step_num=self._step,
                    )
                    if hyp and hyp.strip():
                        self._last_hyp = hyp
                except Exception as e:
                    note_stt_error(e)
                    logger.debug("NemotronStreamingSTT chunk error (step %d): %s", self._step, e)
                self._step += 1

        except Exception as e:
            note_stt_error(e)
            logger.debug("NemotronStreamingSTT push_chunk error: %s", e)

        return self._last_hyp

    def latest_partial(self) -> str:
        """Return the most recent partial hypothesis without feeding new audio."""
        return self._last_hyp

    def flush(self) -> str:
        """Decode everything still held back and return the caught-up hypothesis.

        Called once at the endpoint. Two things lag the true end of speech: up to one
        unprocessed frame (< 560 ms) still in ``_buf``, and the encoder's right context
        (att_context_right) that ``keep_all_outputs=False`` withholds on every normal
        step. Zero-padding to a full frame and running one step with
        ``keep_all_outputs=True`` releases both, so the partial the speculative reply
        is built on usually contains the last words the user actually said. One GPU
        step; safe from the executor thread like push_chunk. Never raises."""
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> str:
        try:
            adapter = get_shared_asr_adapter()
            if self._state is None:
                return self._last_hyp
            pad = self._frame_samples - (len(self._buf) % self._frame_samples or self._frame_samples)
            frame = np.concatenate([self._buf, np.zeros(max(pad, 0), dtype=np.float32)])
            # Always feed at least one full frame so the right context is drained even when
            # the buffer happened to be exactly empty.
            if frame.size == 0:
                frame = np.zeros(self._frame_samples, dtype=np.float32)
            self._buf = np.zeros(0, dtype=np.float32)
            hyp, self._state = adapter.process_chunk(
                frame, self._state, keep_all_outputs=True, step_num=self._step,
            )
            self._step += 1
            if hyp and hyp.strip():
                self._last_hyp = hyp
        except Exception as e:
            note_stt_error(e)
            logger.debug("NemotronStreamingSTT flush error: %s", e)
        return self._last_hyp
