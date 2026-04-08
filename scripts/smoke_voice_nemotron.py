#!/usr/bin/env python3
"""
Quick smoke test: nemotron_asr package layout + optional NeMo import (no full model load).

Usage (from repo root):
  python scripts/smoke_voice_nemotron.py
  PYTHONPATH=voice:nemotron_asr python scripts/smoke_voice_nemotron.py --voice-adapter

Exit 0 = layout/imports OK. Does not benchmark latency or throughput.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NEMOTRON_PKG = REPO / "nemotron_asr"
VOICE_APP = REPO / "voice"


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke test Nemotron ASR package wiring.")
    p.add_argument(
        "--voice-adapter",
        action="store_true",
        help="Also try importing voice Nemotron STT adapter (needs voice deps on PYTHONPATH).",
    )
    p.add_argument(
        "--try-nemo",
        action="store_true",
        help="Try importing nemo.collections.asr (fails if NeMo not installed).",
    )
    args = p.parse_args()

    sys.path.insert(0, str(NEMOTRON_PKG))

    try:
        from nemotron_asr import transcribe_utterance_float32  # noqa: F401
        from nemotron_asr.adapter import ASRModelAdapter  # noqa: F401
    except ImportError as e:
        print("FAIL: nemotron_asr import:", e, file=sys.stderr)
        return 1

    print("OK: nemotron_asr package (adapter + utterance exports)")

    if args.try_nemo:
        try:
            import nemo.collections.asr  # noqa: F401
        except ImportError as e:
            print("WARN: NeMo not installed (expected outside voice/backend image):", e)
        else:
            print("OK: NeMo ASR import")

    if args.voice_adapter:
        sys.path.insert(0, str(VOICE_APP))
        try:
            from app.adapters.stt_nemotron import NemotronUtteranceSTT  # noqa: F401
        except ImportError as e:
            print("WARN: voice adapter import:", e, file=sys.stderr)
            return 0
        print("OK: voice stt_nemotron adapter import")

    return 0


if __name__ == "__main__":
    sys.exit(main())
