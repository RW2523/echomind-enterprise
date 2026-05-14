"""
Docker build: load Nemotron ASR and run one short utterance so NeMo/HF caches and
lazy imports are exercised before runtime. Default device is CPU (no GPU during
`docker build`); runtime still loads on CUDA when ECHOMIND_ASR_DEVICE=cuda.

Run: python -m nemotron_asr.warmup_build --model <hf_id> [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

for _k in ("TORCHDYNAMO_DISABLE", "TORCHINDUCTOR_DISABLE", "TORCH_COMPILE_DISABLE"):
    if _k not in os.environ:
        os.environ[_k] = "1"

from nemotron_asr.adapter import ASRModelAdapter
from nemotron_asr.utterance import transcribe_utterance_float32

logger = logging.getLogger("nemotron_asr.warmup_build")


def run_warmup(
    *,
    model_name: str,
    device: str,
    att_context_right: int,
    chunk_ms: int,
    audio_seconds: float = 1.2,
) -> str:
    os.environ["ECHOMIND_ASR_DEVICE"] = device
    os.environ["ECHOMIND_ASR_REQUIRE_CUDA"] = "0"

    adapter = ASRModelAdapter(model_name=model_name, att_context_right=att_context_right)
    adapter.load()

    n = max(int(16000 * audio_seconds), 1600)
    rng = np.random.default_rng(0)
    audio = (rng.random(n, dtype=np.float64).astype(np.float32) - 0.5) * 0.02

    # NeMo 2.2.x build warmup: keep the utterance in one streaming step so RNNT
    # decoding does not require partial_hypotheses across chunks.
    warmup_chunk_ms = max(chunk_ms, int((n / 16000.0) * 1000) + 100)

    text = transcribe_utterance_float32(
        adapter,
        audio,
        sample_rate=16000,
        chunk_ms=warmup_chunk_ms,
    )
    return text or ""


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Nemotron ASR build warmup (load + one forward pass).")
    p.add_argument(
        "--model",
        default=os.environ.get("ECHOMIND_ASR_MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b"),
        help="Hugging Face model id (same as ECHOMIND_ASR_MODEL_NAME).",
    )
    p.add_argument(
        "--device",
        default=os.environ.get("WARMUP_ASR_DEVICE", "cpu"),
        choices=("cpu", "cuda"),
        help="cpu for docker build; cuda only if build has GPU (BuildKit + NVIDIA).",
    )
    p.add_argument(
        "--att-context-right",
        type=int,
        default=int(os.environ.get("ECHOMIND_ASR_ATT_CONTEXT_RIGHT", "6")),
    )
    p.add_argument(
        "--chunk-ms",
        type=int,
        default=int(
            os.environ.get(
                "VOICE_NEMOTRON_CHUNK_MS",
                os.environ.get("TRANSCRIPT_NEMOTRON_CHUNK_MS", "560"),
            )
        ),
    )
    p.add_argument("--audio-seconds", type=float, default=1.2)
    args = p.parse_args(argv)

    try:
        text = run_warmup(
            model_name=args.model,
            device=args.device,
            att_context_right=args.att_context_right,
            chunk_ms=args.chunk_ms,
            audio_seconds=args.audio_seconds,
        )
    except Exception as e:
        logger.exception("Nemotron warmup failed: %s", e)
        return 1

    logger.info(
        "Nemotron build warmup OK model=%s device=%s chars=%d preview=%r",
        args.model,
        args.device,
        len(text),
        text[:80],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
