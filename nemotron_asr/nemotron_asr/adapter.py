"""
ASR model adapter for nvidia/nemotron-speech-streaming-en-0.6b.
NeMo cache-aware streaming: conformer_stream_step with encoder cache + decoder state.
"""
import copy
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_nemo_asr = None
_EncDecCTCModelBPE = None
_Hypothesis = None


def _import_nemo():
    global _nemo_asr, _EncDecCTCModelBPE, _Hypothesis
    if _nemo_asr is not None:
        return
    try:
        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

        _nemo_asr = nemo_asr
        _EncDecCTCModelBPE = EncDecCTCModelBPE
        _Hypothesis = Hypothesis
    except ImportError as e:
        raise ImportError(
            "NeMo ASR is required for Nemotron. Install: "
            "pip install Cython packaging && "
            'pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"'
        ) from e


def extract_transcriptions(hyps: List[Any]) -> List[str]:
    _import_nemo()
    if not hyps:
        return []
    if isinstance(hyps[0], _Hypothesis):
        return [h.text for h in hyps]
    return list(hyps)


class ASRModelAdapter:
    """NeMo Nemotron streaming ASR: chunk-by-chunk inference with per-session state."""

    def __init__(
        self,
        model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        att_context_right: int = 6,
        device: Optional[str] = None,
    ):
        _import_nemo()
        self.model_name = model_name
        self.att_context_right = att_context_right
        env_dev = (os.getenv("ECHOMIND_ASR_DEVICE") or "").strip().lower()
        if env_dev in ("cpu", "cuda"):
            self.device = env_dev
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            if os.getenv("ECHOMIND_ASR_REQUIRE_CUDA", "").strip().lower() in ("1", "true", "yes"):
                raise RuntimeError(
                    "ECHOMIND_ASR_DEVICE=cuda but torch.cuda.is_available() is False "
                    "(check NVIDIA Container Toolkit, deploy GPU reservation, and driver)."
                )
            self.device = "cpu"
            logger.warning("CUDA not available, falling back to CPU (transcription will be slower)")
        self._model = None
        self._preprocessor = None
        self._pre_encode_cache_size = None
        self._num_channels = None
        self._compute_dtype = torch.float32
        self._forward_lock = threading.Lock()

    def _do_load_body(self) -> None:
        if self._model is not None:
            return
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = os.getenv("TRANSCRIPT_CUDNN_BENCHMARK", "0").lower() in (
                "1",
                "true",
                "yes",
            )
        logger.info("Loading Nemotron ASR model: %s on device=%s", self.model_name, self.device)
        self._model = _nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        self._model = self._model.to(device=self.device, dtype=self._compute_dtype)
        self._model.eval()
        if self.device == "cuda":
            try:
                logger.info("ASR using GPU: %s", torch.cuda.get_device_name(0))
            except Exception:
                pass

        if hasattr(self._model.encoder, "set_default_att_context_size"):
            left = getattr(self._model.encoder, "att_context_size", [[70, 13]])
            if isinstance(left, (list, tuple)) and len(left) > 0:
                left_val = left[0][0] if isinstance(left[0], (list, tuple)) else 70
            else:
                left_val = 70
            self._model.encoder.set_default_att_context_size(
                att_context_size=[left_val, self.att_context_right]
            )
        if hasattr(self._model, "change_decoding_strategy"):
            import inspect

            sig = inspect.signature(self._model.change_decoding_strategy)
            if "decoder_type" in sig.parameters:
                self._model.change_decoding_strategy(decoder_type="rnnt")

        self._pre_encode_cache_size = self._model.encoder.streaming_cfg.pre_encode_cache_size[1]
        self._num_channels = self._model.cfg.preprocessor.features
        self._preprocessor = self._build_preprocessor()
        if self.device == "cuda":
            torch.cuda.synchronize()
        logger.info("Nemotron ASR model loaded on %s", self.device)

    def load(self) -> None:
        with self._forward_lock:
            self._do_load_body()

    def _build_preprocessor(self):
        cfg = copy.deepcopy(self._model._cfg)
        from omegaconf import OmegaConf

        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        preprocessor = _EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        preprocessor.to(self._model.device)
        return preprocessor

    def create_session_state(self) -> Dict[str, Any]:
        with self._forward_lock:
            self._do_load_body()
            batch_size = 1
            cache_last_channel, cache_last_time, cache_last_channel_len = (
                self._model.encoder.get_initial_cache_state(batch_size=batch_size)
            )
            cache_pre_encode = torch.zeros(
                (1, self._num_channels, self._pre_encode_cache_size),
                device=self._model.device,
                dtype=self._compute_dtype,
            )
            if self.device == "cuda":
                torch.cuda.synchronize()
            return {
                "cache_last_channel": cache_last_channel,
                "cache_last_time": cache_last_time,
                "cache_last_channel_len": cache_last_channel_len,
                "cache_pre_encode": cache_pre_encode,
                "previous_hypotheses": None,
                "pred_out_stream": None,
            }

    def process_chunk(
        self,
        audio_float32: np.ndarray,
        state: Dict[str, Any],
        *,
        keep_all_outputs: bool = False,
        step_num: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        with self._forward_lock:
            self._do_load_body()
            device = self._model.device
            audio_signal = (
                torch.from_numpy(audio_float32)
                .unsqueeze(0)
                .to(device, non_blocking=False)
                .to(self._compute_dtype)
            )
            audio_signal_len = torch.tensor([audio_float32.shape[0]], device=device, dtype=torch.long)
            with torch.inference_mode():
                processed_signal, processed_signal_length = self._preprocessor(
                    input_signal=audio_signal, length=audio_signal_len
                )
            cache_pre_encode = state["cache_pre_encode"]
            processed_signal = torch.cat([cache_pre_encode, processed_signal], dim=-1)
            processed_signal_length = processed_signal_length + cache_pre_encode.shape[-1]
            new_cache_pre_encode = processed_signal[:, :, -self._pre_encode_cache_size :].clone()
            state["cache_pre_encode"] = new_cache_pre_encode

            drop_extra = None
            if step_num > 0 and hasattr(self._model.encoder.streaming_cfg, "drop_extra_pre_encoded"):
                drop_extra = self._model.encoder.streaming_cfg.drop_extra_pre_encoded
            with torch.inference_mode():
                (
                    pred_out_stream,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = self._model.conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=state["cache_last_channel"],
                    cache_last_time=state["cache_last_time"],
                    cache_last_channel_len=state["cache_last_channel_len"],
                    keep_all_outputs=keep_all_outputs,
                    previous_hypotheses=state["previous_hypotheses"],
                    previous_pred_out=state["pred_out_stream"],
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )
            state["cache_last_channel"] = cache_last_channel
            state["cache_last_time"] = cache_last_time
            state["cache_last_channel_len"] = cache_last_channel_len
            state["previous_hypotheses"] = previous_hypotheses
            state["pred_out_stream"] = pred_out_stream
            texts = extract_transcriptions(transcribed_texts)
            text = texts[0] if texts else ""
            if self.device == "cuda":
                torch.cuda.synchronize()
            return text.strip(), state
