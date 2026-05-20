"""
Streaming STT for Board Room Mode: nvidia/multitalker-parakeet-streaming-0.6b-v1.

Multi-speaker (multitalker) NeMo RNNT model; per-session stream context.
Shared adapter (weights loaded once); per-connection ParakeetStreamContext.

Speaker output format: list of SpeakerSegment(speaker_id, text, ts_ms).

Key design decisions:
  - model.transcribe() CANNOT be used: it crashes in inference mode
    (ValueError: not enough values to unpack in _transcribe_forward).
    All inference goes through either conformer_stream_step (streaming)
    or _transcribe_direct (direct encoder + RNNT decoder, non-streaming).
  - set_speaker_targets() is patched at load time to NEVER accept None for
    spk_targets.  Passing None kept the model in degraded single-speaker mode.
  - VAD gating is done INSIDE process_pcm_chunk (not in ws.py) so that
    silent audio still increments the silence counter used for speaker-change
    detection.  Previously, ws.py's VAD filter swallowed silence entirely,
    which prevented speaker-change detection from ever triggering.
  - On every confirmed speaker change the RNNT decoder state
    (previous_hypotheses) is reset so the new speaker's transcript starts
    fresh rather than continuing the previous speaker's text.
  - Per-speaker spectral profiles (EMA of log-mel centroid) enable
    returning-speaker recognition, not just new-speaker detection.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.config import settings

for _k in ("TORCHDYNAMO_DISABLE", "TORCHINDUCTOR_DISABLE", "TORCH_COMPILE_DISABLE"):
    if _k not in os.environ:
        os.environ[_k] = "1"

logger = logging.getLogger(__name__)

PARAKEET_AVAILABLE = False
_parakeet_import_error: Optional[str] = None

try:
    import nemo.collections.asr as _nemo_asr_check  # noqa: F401
    PARAKEET_AVAILABLE = True
except ImportError as _e:
    _parakeet_import_error = str(_e)

PARAKEET_MODEL_NAME: str = settings.BOARDROOM_ASR_MODEL_NAME
PARAKEET_SAMPLE_RATE: int = 16000

_parakeet_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_parakeet_adapter_lock = threading.Lock()
_shared_parakeet_adapter: Optional["ParakeetMultitalkerAdapter"] = None
_parakeet_process_lock: Optional[asyncio.Lock] = None


def get_parakeet_process_lock() -> asyncio.Lock:
    global _parakeet_process_lock
    if _parakeet_process_lock is None:
        _parakeet_process_lock = asyncio.Lock()
    return _parakeet_process_lock


def _get_parakeet_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _parakeet_executor
    if _parakeet_executor is None:
        _parakeet_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, settings.BOARDROOM_GPU_CONCURRENCY),
            thread_name_prefix="parakeet_boardroom",
        )
    return _parakeet_executor


@dataclass
class SpeakerSegment:
    """One transcribed segment attributed to a speaker."""
    speaker_id: str
    text: str
    ts_ms: int
    is_final: bool = False


def _pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    new_len = int(round(len(audio) * dst_sr / src_sr))
    if new_len <= 0:
        return np.zeros(0, dtype=np.float32)
    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    return out.squeeze().numpy()


def _sliding_window_rms(audio: np.ndarray, window: int, step: int) -> float:
    if audio is None or audio.size == 0:
        return 0.0
    n = len(audio)
    if n < window:
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    best = 0.0
    for start in range(0, n - window + 1, step):
        w = audio[start : start + window]
        r = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
        if r > best:
            best = r
    return best


# ── Speaker token parsers ──────────────────────────────────────────────────────

_SPEAKER_TOKEN_PATTERNS = [
    re.compile(r'\[(?:SPEAKER|SPK|speaker|spk)_?(\d+)\]', re.IGNORECASE),
    re.compile(r'<(?:speaker|spk)(\d+)>', re.IGNORECASE),
    re.compile(r'\|(?:speaker|spk)_?(\d+)\|', re.IGNORECASE),
]
_LINE_PREFIX_RE = re.compile(r'^(?:speaker|spk)_?(\d+)\s*:\s*(.+)$', re.IGNORECASE)


def _parse_speaker_tokens(text: str) -> List[Tuple[str, str]]:
    """
    Parse speaker-labeled text from multitalker model output.
    Returns list of (speaker_id, text) pairs.
    Falls back to [("SPEAKER_00", full_text)] if no labels found.
    """
    text = (text or "").strip()
    if not text:
        return []

    for pattern in _SPEAKER_TOKEN_PATTERNS:
        if pattern.search(text):
            parts = pattern.split(text)
            results: List[Tuple[str, str]] = []
            if parts[0].strip():
                results.append(("SPEAKER_00", parts[0].strip()))
            for i in range(1, len(parts), 2):
                spk = f"SPEAKER_{int(parts[i]):02d}"
                seg_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if seg_text:
                    results.append((spk, seg_text))
            if results:
                return results

    if re.search(r'^(?:speaker|spk)', text, re.IGNORECASE | re.MULTILINE):
        results = []
        for line in text.splitlines():
            m = _LINE_PREFIX_RE.match(line.strip())
            if m:
                results.append((f"SPEAKER_{int(m.group(1)):02d}", m.group(2).strip()))
        if results:
            return results

    return [("SPEAKER_00", text)]


def _hypothesis_delta(prev: str, curr: str) -> str:
    """
    Compute new text added between two consecutive streaming RNNT hypotheses.
    For RNNT, each step returns the CUMULATIVE hypothesis from stream start.
    """
    prev = (prev or "").strip()
    curr = (curr or "").strip()
    if not curr:
        return ""
    if not prev:
        return curr
    if curr.startswith(prev):
        return curr[len(prev):]
    # Beam search revised previous tokens: emit the corrected suffix.
    # Find longest common prefix to determine the correction point.
    prefix_len = 0
    for a, b in zip(prev.split(), curr.split()):
        if a == b:
            prefix_len += len(a) + 1
        else:
            break
    if prefix_len > 0 and prefix_len < len(curr):
        return curr[prefix_len:]
    return curr


# ── Adapter ────────────────────────────────────────────────────────────────────

class ParakeetMultitalkerAdapter:
    """
    NeMo multitalker Parakeet RNNT adapter.

    Critical implementation notes:
      1. model.transcribe() cannot be used in inference mode — it crashes because
         LhotseSpeechToTextSpkBpeDataset returns (audio, lens, None, None, None, None)
         and _transcribe_forward does `spk_targets, bg_spk_targets = additional_args`
         which fails when additional_args is empty after collation strips Nones.
      2. set_speaker_targets() is patched so it never stores None for spk_targets.
         Without speaker targets the encoder hooks skip the speaker kernel network
         entirely, degrading the model to a single-speaker mode and printing
         "Mask is None" warnings for every chunk.
      3. All inference uses either conformer_stream_step (streaming) or
         _transcribe_direct (direct encoder + RNNT decoder, no dataloader).
    """

    def __init__(
        self,
        model_name: str = PARAKEET_MODEL_NAME,
        device: Optional[str] = None,
    ):
        if not PARAKEET_AVAILABLE:
            raise RuntimeError(
                f"NeMo ASR not available for Parakeet Board Room model. "
                f"Install NeMo ASR toolkit. ({_parakeet_import_error or ''})"
            )
        import nemo.collections.asr as nemo_asr
        self._nemo_asr = nemo_asr
        self.model_name = model_name
        env_dev = (os.getenv("ECHOMIND_ASR_DEVICE") or "").strip().lower()
        if env_dev in ("cpu", "cuda"):
            self.device = env_dev
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logger.warning("Parakeet: CUDA not available, using CPU")
        self._model = None
        self._preprocessor = None
        self._pre_encode_cache_size = None
        self._num_channels = None
        self._compute_dtype = torch.float32
        self._forward_lock = threading.Lock()
        self._supports_streaming = False
        self._is_multitalker = False

    # ── Device helper ──────────────────────────────────────────────────────

    def _model_device(self) -> torch.device:
        """
        Reliably determine the model's device.
        nn.Module.device is not standard in all PyTorch versions; using
        next(parameters()).device is always safe.
        """
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return torch.device(self.device)

    # ── Load ───────────────────────────────────────────────────────────────

    def _do_load(self) -> None:
        if self._model is not None:
            return
        logger.info("Board Room: loading Parakeet model %s on %s", self.model_name, self.device)
        self._model = self._nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        self._model = self._model.to(device=self.device, dtype=self._compute_dtype)
        self._model.eval()

        if self.device == "cuda":
            try:
                logger.info("Parakeet using GPU: %s", torch.cuda.get_device_name(0))
            except Exception:
                pass

        self._is_multitalker = (
            "multitalker" in self.model_name.lower() or
            "multispeaker" in self.model_name.lower() or
            hasattr(self._model, "speaker_model") or
            "multispeaker" in type(self._model).__name__.lower()
        )
        self._supports_streaming = hasattr(self._model.encoder, "streaming_cfg")

        if self._supports_streaming:
            if hasattr(self._model.encoder, "set_default_att_context_size"):
                left_val = 70
                try:
                    left = self._model.encoder.att_context_size
                    if isinstance(left, (list, tuple)) and len(left) > 0:
                        left_val = left[0][0] if isinstance(left[0], (list, tuple)) else 70
                except Exception:
                    pass
                self._model.encoder.set_default_att_context_size(
                    att_context_size=[left_val, 6]
                )
            self._pre_encode_cache_size = self._model.encoder.streaming_cfg.pre_encode_cache_size[1]

        self._num_channels = self._model.cfg.preprocessor.features

        try:
            self._preprocessor = self._model.preprocessor
        except Exception as e:
            logger.warning("Parakeet: could not get model.preprocessor (%s)", e)
            self._preprocessor = None

        # CRITICAL: patch set_speaker_targets to never allow None for spk_targets.
        # The SpeakerKernelMixin hook reads self.spk_targets; when it is None the
        # hook falls through to a 'single-speaker' fallback and skips the speaker
        # kernel network entirely.  By patching we guarantee the encoder always
        # runs in full multi-speaker mode regardless of what calls set_speaker_targets.
        self._patch_speaker_targets()

        if self.device == "cuda":
            torch.cuda.synchronize()
        logger.info(
            "Board Room Parakeet ready (streaming=%s, multitalker=%s)",
            self._supports_streaming, self._is_multitalker,
        )

    def _patch_speaker_targets(self) -> None:
        """
        Override model.set_speaker_targets so that:
          • spk_targets=None  → all-ones  (activates all speaker-kernel frames)
          • bg_spk_targets=None → all-zeros (disables background-speaker kernel
            without producing a 'Mask is None' warning or adding encoder noise)

        Without this patch:
          • spk_targets=None   → NeMo log-warns "Mask is None" every chunk and
            falls back to an ad-hoc all-ones that bypasses proper solve_length_mismatch.
          • bg_spk_targets=None → same warning; bg_spk_kernel receives zero-masked
            input *after* the warning path, which can corrupt encoder representations.
        """
        if not hasattr(self._model, "set_speaker_targets"):
            return

        model_ref = self._model
        compute_dtype = self._compute_dtype

        def _patched_set_speaker_targets(
            spk_targets: Optional[torch.Tensor] = None,
            bg_spk_targets: Optional[torch.Tensor] = None,
        ) -> None:
            dev = next(model_ref.parameters()).device
            if spk_targets is None:
                # All-ones → every frame attributed to speaker 0 (safe fallback).
                spk_targets = torch.ones(1, 3000, device=dev, dtype=compute_dtype)
            model_ref.spk_targets = spk_targets
            if getattr(model_ref, "add_bg_spk_kernel", False):
                if bg_spk_targets is None:
                    # All-zeros → no background speaker contribution.
                    # This prevents 'Mask is None' warning and avoids the bg_spk_kernel
                    # adding corrupted values (its input would be x * zeros = 0,
                    # but then the kernel bias term could add non-zero noise to x).
                    # Providing explicit zeros makes solve_length_mismatch skip the
                    # warning path entirely.
                    bg_spk_targets = torch.zeros(1, 3000, device=dev, dtype=compute_dtype)
                model_ref.bg_spk_targets = bg_spk_targets

        self._model.set_speaker_targets = _patched_set_speaker_targets
        # Immediately activate speaker kernels (spk=ones, bg_spk=zeros).
        self._model.set_speaker_targets(None, None)
        logger.info("Parakeet: speaker targets patched — encoder always in multi-speaker mode")

    def load(self) -> None:
        with self._forward_lock:
            self._do_load()

    # ── Session state ──────────────────────────────────────────────────────

    def create_session_state(self) -> Dict[str, Any]:
        """Create per-session streaming cache state."""
        with self._forward_lock:
            self._do_load()
            state: Dict[str, Any] = {"audio_offset": 0}
            if not self._supports_streaming or self._pre_encode_cache_size is None:
                state["streaming"] = False
                return state
            batch_size = 1
            cache_last_channel, cache_last_time, cache_last_channel_len = (
                self._model.encoder.get_initial_cache_state(batch_size=batch_size)
            )
            cache_pre_encode = torch.zeros(
                (1, self._num_channels, self._pre_encode_cache_size),
                device=self._model_device(),
                dtype=self._compute_dtype,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            state.update({
                "streaming": True,
                "cache_last_channel": cache_last_channel,
                "cache_last_time": cache_last_time,
                "cache_last_channel_len": cache_last_channel_len,
                "cache_pre_encode": cache_pre_encode,
                "previous_hypotheses": None,
                "pred_out_stream": None,
            })
            return state

    # ── Direct inference (replaces broken model.transcribe()) ──────────────

    def _transcribe_direct(self, audio: np.ndarray) -> List[Tuple[str, str]]:
        """
        Direct encoder + RNNT decoder inference.

        This replaces model.transcribe() which crashes in inference mode
        because LhotseSpeechToTextSpkBpeDataset._transcribe_forward tries to
        unpack spk_targets from an empty additional_args list.

        We bypass the dataloader entirely:
          preprocessor → encoder → rnnt_decoder_predictions_tensor
        The patched set_speaker_targets ensures speaker kernels are active.
        """
        if len(audio) < 160:
            return []
        try:
            device = self._model_device()
            audio_t = torch.from_numpy(audio).unsqueeze(0).to(device).to(self._compute_dtype)
            audio_len = torch.tensor([len(audio)], device=device, dtype=torch.long)

            with torch.inference_mode():
                proc, proc_len = self._model.preprocessor(
                    input_signal=audio_t, length=audio_len
                )
                # Full-audio (non-streaming) encoder forward; no cache needed.
                enc_out = self._model.encoder(audio_signal=proc, length=proc_len)
                if isinstance(enc_out, (list, tuple)):
                    encoded, encoded_len = enc_out[0], enc_out[1]
                else:
                    encoded, encoded_len = enc_out, proc_len

                best_hyp = self._model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=encoded,
                    encoded_lengths=encoded_len,
                    return_hypotheses=True,
                    partial_hypotheses=None,
                )

            if self.device == "cuda":
                torch.cuda.synchronize()
            return self._extract_speaker_pairs(best_hyp)
        except Exception as e:
            logger.warning("Parakeet _transcribe_direct error: %s", e)
            return []

    # ── Streaming inference ────────────────────────────────────────────────

    def _raw_transcribe_streaming(
        self,
        audio: np.ndarray,
        state: Dict[str, Any],
        keep_all: bool = False,
        step_num: int = 0,
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
        """
        Run one streaming inference step via conformer_stream_step.
        Returns ([(speaker_id, text), ...], new_state).
        The text is the CUMULATIVE hypothesis from stream start (caller diffs).
        """
        with self._forward_lock:
            self._do_load()
            state["audio_offset"] = state.get("audio_offset", 0) + len(audio)

            if not state.get("streaming", False):
                pairs = self._transcribe_direct(audio)
                return pairs, state

            device = self._model_device()
            audio_signal = (
                torch.from_numpy(audio).unsqueeze(0).to(device).to(self._compute_dtype)
            )
            audio_signal_len = torch.tensor([audio.shape[0]], device=device, dtype=torch.long)

            if self._preprocessor is not None:
                with torch.inference_mode():
                    processed_signal, processed_signal_length = self._preprocessor(
                        input_signal=audio_signal, length=audio_signal_len
                    )
            else:
                processed_signal = audio_signal
                processed_signal_length = audio_signal_len

            cache_pre_encode = state["cache_pre_encode"]
            processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
            processed_signal_length = processed_signal_length + cache_pre_encode.shape[-1]
            new_cache = processed_signal[:, :, -self._pre_encode_cache_size:].clone()
            state["cache_pre_encode"] = new_cache

            drop_extra = None
            if step_num > 0 and hasattr(self._model.encoder.streaming_cfg, "drop_extra_pre_encoded"):
                drop_extra = self._model.encoder.streaming_cfg.drop_extra_pre_encoded

            # Speaker targets are permanently set via the patched set_speaker_targets,
            # so no explicit call is needed here.  The hook will read self._model.spk_targets
            # which is always non-None after _patch_speaker_targets().

            with torch.inference_mode():
                result = self._model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=state["cache_last_channel"],
                    cache_last_time=state["cache_last_time"],
                    cache_last_channel_len=state["cache_last_channel_len"],
                    keep_all_outputs=keep_all,
                    previous_hypotheses=state["previous_hypotheses"],
                    previous_pred_out=state.get("pred_out_stream"),
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )

            # conformer_stream_step returns 6-tuple for RNNT:
            # (greedy_predictions, all_hyp, cache_ch, cache_tm, cache_ch_len, best_hyp)
            (
                pred_out_stream,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = result[:6]

            state["cache_last_channel"] = cache_last_channel
            state["cache_last_time"] = cache_last_time
            state["cache_last_channel_len"] = cache_last_channel_len
            state["previous_hypotheses"] = previous_hypotheses
            state["pred_out_stream"] = pred_out_stream

            pairs = self._extract_speaker_pairs(transcribed_texts)
            if self.device == "cuda":
                torch.cuda.synchronize()
            return pairs, state

    # ── Output parsing ─────────────────────────────────────────────────────

    def _extract_speaker_pairs(self, raw_output: Any) -> List[Tuple[str, str]]:
        """
        Parse model output into (speaker_id, text) pairs.
        Handles: Hypothesis objects, dicts, plain strings, inline speaker tokens.
        """
        if raw_output is None:
            return []

        pairs: List[Tuple[str, str]] = []
        try:
            items = raw_output if isinstance(raw_output, (list, tuple)) else [raw_output]
            for item in items:
                if isinstance(item, dict):
                    spk = (
                        item.get("speaker_id") or item.get("speaker") or "SPEAKER_00"
                    )
                    txt = (item.get("text") or item.get("transcription") or "").strip()
                    if txt:
                        inner = _parse_speaker_tokens(txt)
                        if len(inner) == 1 and inner[0][0] == "SPEAKER_00":
                            pairs.append((str(spk), txt))
                        else:
                            pairs.extend(inner)
                elif hasattr(item, "text"):
                    # NeMo Hypothesis object — .text is the cumulative transcript
                    txt = (item.text or "").strip()
                    # Also try decoding y_sequence if text is empty
                    if not txt and hasattr(item, "y_sequence"):
                        try:
                            ys = item.y_sequence
                            if hasattr(ys, "tolist"):
                                ys = ys.tolist()
                            if ys and hasattr(self._model, "tokenizer"):
                                txt = self._model.tokenizer.ids_to_text(ys).strip()
                        except Exception:
                            pass
                    spk = str(
                        getattr(item, "speaker_id", None)
                        or getattr(item, "speaker", None)
                        or "SPEAKER_00"
                    )
                    if txt:
                        inner = _parse_speaker_tokens(txt)
                        if len(inner) == 1 and inner[0][0] == "SPEAKER_00":
                            pairs.append((spk, txt))
                        else:
                            pairs.extend(inner)
                elif isinstance(item, str) and item.strip():
                    pairs.extend(_parse_speaker_tokens(item.strip()))
        except Exception as e:
            logger.debug("Parakeet speaker extraction error (non-fatal): %s", e)

        return [(spk, txt) for spk, txt in pairs if txt.strip()]

    def transcribe_full(self, audio: np.ndarray) -> List[SpeakerSegment]:
        """
        Full offline transcription for final board room session processing.
        Uses _transcribe_direct (not model.transcribe() which is broken).
        """
        with self._forward_lock:
            self._do_load()
            if len(audio) < 160:
                return []
            try:
                pairs = self._transcribe_direct(audio)
                if not pairs:
                    return []
                return [
                    SpeakerSegment(speaker_id=spk, text=txt, ts_ms=0, is_final=True)
                    for spk, txt in pairs if txt.strip()
                ]
            except Exception as e:
                logger.error("Parakeet full transcription failed: %s", e)
                return []


# ── Stream context ─────────────────────────────────────────────────────────────

class ParakeetStreamContext:
    """
    Per-session streaming state for the Parakeet multitalker model.

    Speaker change detection strategy
    ──────────────────────────────────
    The multitalker RNNT emits ONE hypothesis per chunk (dominant speaker).
    To detect and label multiple speakers we use a lightweight spectral
    approach:

      1. Every audio chunk is checked for silence (RMS threshold).
         Silent chunks increment _silent_chunk_count WITHOUT running ASR.
         This is critical: ws.py must NOT filter silent audio before it
         reaches process_pcm_chunk, otherwise silence is invisible here.

      2. After a silence gap of >= _SILENCE_GAP_CHUNKS consecutive silent
         800ms chunks (~800ms silence), the next speech chunk triggers a
         speaker identity check.

      3. We maintain a per-speaker spectral profile (EMA of 32-band log-mel
         centroid).  The incoming chunk centroid is compared against all
         known profiles:
           - If closest profile distance < _RECOGNITION_THRESH → same speaker
           - If closest profile distance > _NEW_SPEAKER_THRESH  → new speaker
           - Between the two thresholds                         → unchanged

      4. On confirmed speaker change the RNNT decoder state
         (previous_hypotheses) is reset so the new speaker's text starts
         fresh rather than continuing the previous speaker's beam search.
    """

    # ── Silence / VAD ─────────────────────────────────────────────────────────
    # 0.004 was too low for real rooms (ambient noise ~0.01-0.05 RMS).
    # 0.012 is a good middle ground: catches quiet speech pauses without
    # triggering on constant background hum.
    _SILENCE_RMS_THRESH: float = 0.012
    # One silent chunk (~400 ms) is enough to mark a potential speaker boundary.
    _SILENCE_GAP_CHUNKS: int = 1

    # ── Intra-turn check ───────────────────────────────────────────────────────
    # Check speaker identity every N consecutive speech chunks even without
    # a silence gap.  Interval = 2 chunks ≈ 800 ms — fast enough to catch
    # natural speaker alternation in conversation.
    # (Was 6 = 2.4 s which missed most turn-taking.)
    _INTRA_TURN_CHECK_INTERVAL: int = 2

    # ── Spectral speaker profiling ─────────────────────────────────────────────
    # 40-band mel gives more resolution than 32 for distinguishing voices.
    _N_MEL_BANDS: int = 40

    # Cosine distance thresholds:
    #   < _RECOGNITION_THRESH  → definitely same speaker (profile updated)
    #   > _NEW_SPEAKER_THRESH  → new speaker confirmed
    #   between the two        → ambiguous; resolved by consecutive-chunk voting
    #
    # Tighter thresholds (was 0.15/0.26) reduce the ambiguous dead-zone where
    # a second voice was silently absorbed into Speaker 1's profile.
    _RECOGNITION_THRESH: float = 0.08   # very close → same speaker
    _NEW_SPEAKER_THRESH: float = 0.16   # clearly different → new speaker

    # EMA weight for profile updates WITHIN a confirmed speaker turn.
    # 0.05 (was 0.30): slow adaptation keeps the profile stable across the full
    # utterance and prevents it from drifting toward a different speaker's voice.
    _EMA_ALPHA: float = 0.05

    # ── Centroid stability window ──────────────────────────────────────────────
    # Average this many recent chunk centroids before comparing to profiles.
    # A 3-chunk (1.2 s) running average smooths out frame-level noise.
    _CENTROID_WINDOW: int = 3

    # ── Consecutive-chunk voting ───────────────────────────────────────────────
    # To confirm a speaker change, the new centroid must stay outside the
    # recognition threshold for this many consecutive check intervals.
    # 2 votes (≈ 1.6 s) prevents single-chunk acoustic artifacts from firing
    # a false speaker change while still being responsive to real turns.
    _CHANGE_VOTES_NEEDED: int = 2

    def __init__(self, adapter: ParakeetMultitalkerAdapter):
        self.adapter = adapter
        self.reset()

    def reset(self) -> None:
        self.state: Dict[str, Any] = self.adapter.create_session_state()
        self.state["audio_offset"] = 0
        self.buffer: deque = deque()
        self.buffer_samples: int = 0
        self.stream_step_num: int = 0
        # Per-speaker cumulative hypothesis for delta computation
        self._last_hyp: Dict[str, str] = {}
        # Full audio for final EOS flush
        self._all_chunks: List[np.ndarray] = []
        # Speaker tracking
        self._current_speaker: str = "SPEAKER_00"
        self._speaker_counter: int = 0
        self._silent_chunk_count: int = 0
        self._speech_chunk_count: int = 0   # consecutive non-silent chunks
        # Per-speaker spectral profiles: speaker_id -> normalized centroid
        self._speaker_profiles: Dict[str, np.ndarray] = {}
        # Sliding window of recent chunk centroids for stable comparison
        self._centroid_history: deque = deque(maxlen=self._CENTROID_WINDOW)
        # Pending speaker-change votes: (candidate_speaker, vote_count)
        self._pending_change: Optional[Tuple[str, int]] = None
        # Speaker timeline: list of (sample_offset, speaker_id) transitions.
        # Used in flush() to re-attribute the full-session transcript with
        # correct speakers even when the streaming RNNT doesn't produce output.
        self._speaker_timeline: List[Tuple[int, str]] = [(0, "SPEAKER_00")]
        # Running count of samples processed (for timeline bookkeeping)
        self._sample_cursor: int = 0

    # ── Spectral speaker profiling ─────────────────────────────────────────

    def _log_mel_centroid(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute a normalized N-band log-mel energy vector for speaker profiling.
        Uses Hanning window + FFT + triangular mel filterbank.
        """
        n = len(audio)
        if n < 64:
            return np.zeros(self._N_MEL_BANDS, dtype=np.float32)

        fft_size = max(512, 1 << (n - 1).bit_length())
        win = np.hanning(n)
        spec = np.abs(np.fft.rfft(audio * win, n=fft_size)) ** 2
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / PARAKEET_SAMPLE_RATE)

        # Mel filterbank — evenly spaced in mel scale
        f_min, f_max = 80.0, min(8000.0, PARAKEET_SAMPLE_RATE / 2.0)
        mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
        mel_pts = np.linspace(mel_min, mel_max, self._N_MEL_BANDS + 2)
        hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)

        centroid = np.zeros(self._N_MEL_BANDS, dtype=np.float64)
        for b in range(self._N_MEL_BANDS):
            lo, hi = hz_pts[b], hz_pts[b + 2]
            mask = (freqs >= lo) & (freqs <= hi)
            if mask.any():
                centroid[b] = spec[mask].sum()

        # Log-compress and L2 normalize
        centroid = np.log1p(centroid)
        norm = np.linalg.norm(centroid)
        return (centroid / (norm + 1e-9)).astype(np.float32)

    def _smoothed_centroid(self, raw: np.ndarray) -> np.ndarray:
        """
        Push raw centroid into a sliding window and return the window average.
        A multi-chunk average is more stable than a single 400 ms estimate.
        """
        self._centroid_history.append(raw)
        stacked = np.stack(list(self._centroid_history), axis=0)
        avg = stacked.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        return avg / (norm + 1e-9)

    def _cosine_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))

    def _detect_speaker(self, centroid: np.ndarray, trigger: str) -> str:
        """
        Determine the speaker ID for an audio chunk.

        Uses consecutive-chunk voting to avoid single-frame false switches:
        - A speaker change is *proposed* when the smoothed centroid exceeds
          _RECOGNITION_THRESH against the current speaker's profile.
        - The change is *confirmed* after _CHANGE_VOTES_NEEDED consecutive
          check intervals all agree on the same new candidate.

        Always logs cosine distances at DEBUG level for diagnosability.
        """
        smoothed = self._smoothed_centroid(centroid)

        if not self._speaker_profiles:
            self._speaker_profiles[self._current_speaker] = smoothed.copy()
            return self._current_speaker

        # Distance to every known speaker profile
        distances: Dict[str, float] = {
            spk: self._cosine_dist(smoothed, prof)
            for spk, prof in self._speaker_profiles.items()
        }
        best_spk = min(distances, key=lambda s: distances[s])
        best_dist = distances[best_spk]
        dist_str = "  ".join(f"{s}:{d:.3f}" for s, d in sorted(distances.items()))

        logger.debug(
            "Board Room: speaker check (%s) current=%s  distances=[%s]  best=%s(%.3f)",
            trigger, self._current_speaker, dist_str, best_spk, best_dist,
        )

        if best_dist <= self._RECOGNITION_THRESH:
            # Clearly the same speaker — clear any pending vote
            self._pending_change = None
            candidate = best_spk

        elif best_dist >= self._NEW_SPEAKER_THRESH:
            # Clearly different from current speaker.
            if best_spk != self._current_speaker:
                # Matches a different KNOWN speaker's profile already
                candidate = best_spk
            else:
                # All known profiles are far away → a new voice is speaking.
                # CRITICAL: reuse the same provisional ID that we already started
                # voting for, rather than minting a fresh SPEAKER_XX every check.
                # Without this, each check creates a new ID and votes never
                # accumulate past 1/N (the bug that kept everything as SPEAKER_00).
                if self._pending_change is not None:
                    candidate = self._pending_change[0]
                else:
                    self._speaker_counter += 1
                    candidate = f"SPEAKER_{self._speaker_counter:02d}"
                    logger.info(
                        "Board Room: new speaker candidate %s "
                        "(min_dist=%.3f to %s via %s)",
                        candidate, best_dist, best_spk, trigger,
                    )

        else:
            # Ambiguous zone — do not switch, but keep any in-progress vote alive
            # so a single "recovery" chunk can't erase accumulated evidence.
            candidate = self._current_speaker

        # ── Voting: require _CHANGE_VOTES_NEEDED consecutive agreements ────────
        if candidate != self._current_speaker:
            if self._pending_change and self._pending_change[0] == candidate:
                votes = self._pending_change[1] + 1
            else:
                votes = 1
            self._pending_change = (candidate, votes)

            if votes >= self._CHANGE_VOTES_NEEDED:
                logger.info(
                    "Board Room: speaker change CONFIRMED %s → %s "
                    "(votes=%d/%d trigger=%s dist=%.3f)",
                    self._current_speaker, candidate,
                    votes, self._CHANGE_VOTES_NEEDED, trigger, best_dist,
                )
                new_spk = candidate
                self._pending_change = None
            else:
                logger.debug(
                    "Board Room: speaker change vote %d/%d for %s (trigger=%s)",
                    votes, self._CHANGE_VOTES_NEEDED, candidate, trigger,
                )
                new_spk = self._current_speaker   # hold until confirmed
        else:
            new_spk = self._current_speaker

        # ── Update speaker profile with EMA ───────────────────────────────────
        # Do NOT update the profile while a speaker-change vote is in progress.
        # If we updated SPEAKER_00's profile with the new voice's audio during
        # the voting window, the distances would drift downward and the second
        # speaker's voice would be absorbed into Speaker 1's profile before the
        # vote confirms — exactly the bug seen in the first session.
        if self._pending_change is None:
            if new_spk in self._speaker_profiles:
                self._speaker_profiles[new_spk] = (
                    self._EMA_ALPHA * smoothed
                    + (1.0 - self._EMA_ALPHA) * self._speaker_profiles[new_spk]
                )
                norm = np.linalg.norm(self._speaker_profiles[new_spk])
                self._speaker_profiles[new_spk] /= (norm + 1e-9)
            else:
                self._speaker_profiles[new_spk] = smoothed.copy()

        return new_spk

    # ── Audio buffering ────────────────────────────────────────────────────

    def _chunk_samples(self) -> int:
        # Minimum 200 ms gives enough frames for the streaming encoder to produce
        # meaningful output; 400 ms (default) balances latency vs accuracy.
        ms = max(200, settings.BOARDROOM_CHUNK_MS)
        return int(PARAKEET_SAMPLE_RATE * ms / 1000)

    def _take_chunk(self) -> Optional[np.ndarray]:
        needed = self._chunk_samples()
        if self.buffer_samples < needed:
            return None
        chunks: List[np.ndarray] = []
        collected = 0
        while self.buffer and collected < needed:
            c = self.buffer.popleft()
            self.buffer_samples -= len(c)
            chunks.append(c)
            collected += len(c)
        if not chunks:
            return None
        combined = np.concatenate(chunks).astype(np.float32)
        if len(combined) > needed:
            remainder = combined[needed:]
            self.buffer.appendleft(remainder)
            self.buffer_samples += len(remainder)
            return combined[:needed]
        return combined

    def append_samples(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        arr = samples.astype(np.float32, copy=False)
        self.buffer.append(arr)
        self.buffer_samples += len(arr)
        self._all_chunks.append(arr)

    # ── Delta computation ──────────────────────────────────────────────────

    def _compute_deltas(self, pairs: List[Tuple[str, str]], ts_ms: int) -> List[SpeakerSegment]:
        """
        Convert (speaker_id, cumulative_hypothesis) pairs into delta SpeakerSegments.
        """
        result: List[SpeakerSegment] = []
        for speaker_id, hyp in pairs:
            prev = self._last_hyp.get(speaker_id, "")
            delta = _hypothesis_delta(prev, hyp)
            if delta.strip():
                result.append(SpeakerSegment(speaker_id=speaker_id, text=delta, ts_ms=ts_ms))
            if hyp.strip():
                self._last_hyp[speaker_id] = hyp
        return result

    # ── Main processing ────────────────────────────────────────────────────

    async def process_pcm_chunk(
        self,
        pcm_float32: np.ndarray,
        client_sr: int,
        loop: asyncio.AbstractEventLoop,
    ) -> List[SpeakerSegment]:
        """
        Resample → buffer → per-chunk: silence detection, speaker detection,
        streaming inference, delta computation → return new SpeakerSegments.

        IMPORTANT: Callers must NOT apply a VAD gate before calling this method.
        Silent audio must reach here so _silent_chunk_count increments correctly
        for speaker-change detection.
        """
        if client_sr != PARAKEET_SAMPLE_RATE:
            pcm_float32 = _resample(pcm_float32, client_sr, PARAKEET_SAMPLE_RATE)
        self.append_samples(pcm_float32)

        all_segments: List[SpeakerSegment] = []
        ex = _get_parakeet_executor()

        while True:
            audio = self._take_chunk()
            if audio is None:
                break

            # Track sample cursor for speaker timeline (used in flush())
            chunk_start_sample = self._sample_cursor
            self._sample_cursor += len(audio)

            # ── Silence detection ──────────────────────────────────────────
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            if rms < self._SILENCE_RMS_THRESH:
                self._silent_chunk_count += 1
                self._speech_chunk_count = 0
                continue

            # ── Speaker detection ──────────────────────────────────────────
            raw_centroid = self._log_mel_centroid(audio)
            self._speech_chunk_count += 1

            # Trigger a speaker identity check when:
            #   a) we just came out of a silence gap (classic turn-taking pause), OR
            #   b) every _INTRA_TURN_CHECK_INTERVAL speech chunks without a pause
            after_silence = self._silent_chunk_count >= self._SILENCE_GAP_CHUNKS
            intra_turn_check = (
                self._speech_chunk_count > 1
                and self._speech_chunk_count % self._INTRA_TURN_CHECK_INTERVAL == 0
            )

            if self._current_speaker not in self._speaker_profiles:
                # Very first speech chunk: seed profile via _detect_speaker
                self._detect_speaker(raw_centroid, trigger="init")
            elif after_silence or intra_turn_check:
                trigger_reason = "silence" if after_silence else "intra"
                new_spk = self._detect_speaker(raw_centroid, trigger=trigger_reason)
                if new_spk != self._current_speaker:
                    self._current_speaker = new_spk
                    # Record the speaker switch in the timeline for flush() attribution
                    self._speaker_timeline.append((chunk_start_sample, new_spk))
                    logger.debug(
                        "Board Room: speaker timeline updated → %s at sample %d",
                        new_spk, chunk_start_sample,
                    )
                    # Reset RNNT decoder so new speaker starts fresh.
                    self.state["previous_hypotheses"] = None
                    self.state["pred_out_stream"] = None
                    self._last_hyp.pop(new_spk, None)
                    self._speech_chunk_count = 0
            else:
                # Within a confirmed speaker turn: push centroid to history window
                # so the next check uses a smoothed average — no EMA profile
                # update mid-turn to keep the profile stable.
                self._centroid_history.append(raw_centroid)

            self._silent_chunk_count = 0
            attributed_speaker = self._current_speaker

            # ── Streaming ASR inference ────────────────────────────────────
            audio_copy = audio.copy()
            step = self.stream_step_num
            self.stream_step_num += 1
            cur_state = self.state

            def _run(ac=audio_copy, s=cur_state, st=step):
                return self.adapter._raw_transcribe_streaming(
                    ac, s, keep_all=False, step_num=st
                )

            pairs, new_state = await loop.run_in_executor(ex, _run)
            self.state = new_state

            ts_ms = int((self.state.get("audio_offset", 0) / PARAKEET_SAMPLE_RATE) * 1000)

            if pairs:
                # Override model's speaker_id with our detected speaker.
                # The multitalker RNNT always emits SPEAKER_00 in streaming mode;
                # our detection layer maps it to the actual speaker.
                attributed_pairs = [
                    (attributed_speaker if spk == "SPEAKER_00" else spk, txt)
                    for spk, txt in pairs
                ]
                deltas = self._compute_deltas(attributed_pairs, ts_ms)
                all_segments.extend(deltas)

        return all_segments

    def _apply_speaker_timeline(
        self, segs: List[SpeakerSegment], total_samples: int
    ) -> List[SpeakerSegment]:
        """
        Re-attribute full-session transcription segments using the speaker timeline
        that was tracked during real-time processing.

        The streaming ASR often produces no text in real time (RNNT needs warm-up),
        but we DID detect speaker changes.  This method merges the full-session
        text (from transcribe_direct) with those detected speaker boundaries so
        the final transcript gets the right speaker labels.

        Words are split proportionally by the fraction of audio each speaker occupied.
        """
        if len(self._speaker_timeline) <= 1:
            # Only one speaker — nothing to re-attribute
            return segs

        full_text = " ".join(s.text.strip() for s in segs if s.text.strip())
        words = full_text.split()
        if not words:
            return segs

        timeline = sorted(self._speaker_timeline, key=lambda x: x[0])
        total = max(total_samples, 1)

        # Build (start_frac, end_frac, speaker_id) intervals
        intervals: List[Tuple[float, float, str]] = []
        for i, (offset, spk) in enumerate(timeline):
            start_frac = offset / total
            end_frac = timeline[i + 1][0] / total if i + 1 < len(timeline) else 1.0
            duration_frac = end_frac - start_frac
            if duration_frac > 0:
                intervals.append((start_frac, end_frac, spk))

        result: List[SpeakerSegment] = []
        word_idx = 0
        for i, (start_frac, end_frac, spk) in enumerate(intervals):
            n_words = round(len(words) * (end_frac - start_frac))
            # Last interval gets all remaining words
            if i == len(intervals) - 1:
                n_words = len(words) - word_idx
            n_words = max(1, n_words)
            chunk_words = words[word_idx: word_idx + n_words]
            if chunk_words:
                ts_ms = int(start_frac * total / PARAKEET_SAMPLE_RATE * 1000)
                result.append(SpeakerSegment(
                    speaker_id=spk,
                    text=" ".join(chunk_words),
                    ts_ms=ts_ms,
                    is_final=True,
                ))
            word_idx += n_words
            if word_idx >= len(words):
                break

        # Safety: any leftover words go to the last speaker
        if word_idx < len(words):
            last_spk = timeline[-1][1]
            last_ts_ms = int(timeline[-1][0] / PARAKEET_SAMPLE_RATE * 1000)
            result.append(SpeakerSegment(
                speaker_id=last_spk,
                text=" ".join(words[word_idx:]),
                ts_ms=last_ts_ms,
                is_final=True,
            ))

        logger.info(
            "Parakeet: speaker timeline applied — %d speakers × %d word-chunks "
            "(timeline=%s)",
            len(set(s.speaker_id for s in result)),
            len(result),
            [(spk, f"{off/PARAKEET_SAMPLE_RATE:.1f}s") for off, spk in timeline],
        )
        return result

    async def flush(self, loop: asyncio.AbstractEventLoop) -> List[SpeakerSegment]:
        """
        Flush remaining audio buffer, then run full-session offline transcription
        for the highest-accuracy final transcript.

        Speaker attribution is preserved by applying the real-time speaker timeline
        to the full-session output.  This is critical because the streaming RNNT
        often produces little to no text in real time (it needs warm-up context),
        but speaker changes ARE reliably detected in real time via spectral profiling.

        Note: model.transcribe() is broken for this model (crashes in inference
        mode).  Full-session transcription uses _transcribe_direct instead.
        """
        ex = _get_parakeet_executor()

        # Flush remaining streaming buffer
        streaming_segs: List[SpeakerSegment] = []
        if self.buffer_samples > 0:
            chunks = list(self.buffer)
            self.buffer.clear()
            self.buffer_samples = 0
            if chunks:
                audio = np.concatenate(chunks).astype(np.float32)
                audio_copy = audio.copy()
                step = self.stream_step_num
                self.stream_step_num += 1

                def _flush_run(ac=audio_copy, s=self.state, st=step):
                    return self.adapter._raw_transcribe_streaming(
                        ac, s, keep_all=True, step_num=st
                    )

                pairs, new_state = await loop.run_in_executor(ex, _flush_run)
                self.state = new_state
                ts_ms = int((self.state.get("audio_offset", 0) / PARAKEET_SAMPLE_RATE) * 1000)
                if pairs:
                    attributed_pairs = [
                        (self._current_speaker if spk == "SPEAKER_00" else spk, txt)
                        for spk, txt in pairs
                    ]
                    streaming_segs.extend(self._compute_deltas(attributed_pairs, ts_ms))

        # Full-session offline transcription for best accuracy
        if self._all_chunks:
            full_audio = np.concatenate(self._all_chunks).astype(np.float32)
            total_samples = len(full_audio)

            def _full(fa=full_audio):
                return self.adapter.transcribe_full(fa)

            full_segs = await loop.run_in_executor(ex, _full)
            if full_segs:
                logger.info(
                    "Parakeet: full-session transcription returned %d raw segments; "
                    "speaker timeline has %d entries",
                    len(full_segs), len(self._speaker_timeline),
                )
                # Apply our real-time speaker timeline to the full transcript so
                # each portion of the text gets the correct speaker label.
                attributed = self._apply_speaker_timeline(full_segs, total_samples)
                return attributed if attributed else full_segs

        return streaming_segs


# ── Singleton helpers ──────────────────────────────────────────────────────────

def get_shared_parakeet_adapter() -> ParakeetMultitalkerAdapter:
    global _shared_parakeet_adapter
    if not PARAKEET_AVAILABLE:
        raise RuntimeError(
            f"NeMo ASR not available for Parakeet Board Room mode. "
            f"Install NeMo ASR (see backend Dockerfile). "
            + (_parakeet_import_error or "")
        )
    with _parakeet_adapter_lock:
        if _shared_parakeet_adapter is None:
            _shared_parakeet_adapter = ParakeetMultitalkerAdapter(model_name=PARAKEET_MODEL_NAME)
            _shared_parakeet_adapter.load()
        return _shared_parakeet_adapter


def download_parakeet_model() -> bool:
    if not PARAKEET_AVAILABLE:
        return False
    try:
        from huggingface_hub import snapshot_download
        offline = os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "yes")
        snapshot_download(PARAKEET_MODEL_NAME, local_files_only=offline)
        return True
    except Exception:
        return False


def preload_parakeet() -> bool:
    try:
        get_shared_parakeet_adapter()
        return True
    except Exception:
        return False


# Alias used by multitalker_transcription.py to access the loaded adapter.
get_parakeet_adapter = get_shared_parakeet_adapter
