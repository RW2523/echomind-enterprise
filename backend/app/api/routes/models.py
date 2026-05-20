"""
Model status endpoint.

GET /api/models/status

Returns readiness state for all three ASR pipelines:
  - live_transcript  (Nemotron)
  - boardroom        (Parakeet multitalker)
  - final_cleanup    (VibeVoice-ASR)
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["models"])


@router.get("/models/status")
def model_status():
    """Return readiness state for all speech model pipelines."""

    # ── Live Transcript (Nemotron) ────────────────────────────────────────────
    try:
        from ...transcribe.stt_streaming import (
            NEMOTRON_AVAILABLE,
            NEMOTRON_MODEL_NAME,
            _nemotron_import_error,
            _shared_adapter,
        )
        if not NEMOTRON_AVAILABLE:
            lt_status = "unavailable"
            lt_error = _nemotron_import_error
        elif _shared_adapter is not None:
            lt_status = "ready"
            lt_error = None
        else:
            lt_status = "not_loaded"
            lt_error = None
    except Exception as e:
        lt_status = "failed"
        lt_error = str(e)
        NEMOTRON_MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"

    # ── Board Room (Parakeet multitalker) ─────────────────────────────────────
    try:
        from ...boardroom.stt_parakeet import (
            PARAKEET_AVAILABLE,
            PARAKEET_MODEL_NAME,
            _parakeet_import_error,
            _shared_parakeet_adapter,
        )
        from ...core.config import settings as _cfg
        if not PARAKEET_AVAILABLE:
            br_status = "unavailable"
            br_error = _parakeet_import_error
        elif _shared_parakeet_adapter is not None:
            br_status = "ready"
            br_error = None
        else:
            br_status = "not_loaded"
            br_error = None
        diar_model = _cfg.BOARDROOM_DIAR_MODEL
    except Exception as e:
        br_status = "failed"
        br_error = str(e)
        PARAKEET_MODEL_NAME = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
        diar_model = "nvidia/diar_streaming_sortformer_4spk-v2.1"

    # ── Final Cleanup (VibeVoice-ASR) ─────────────────────────────────────────
    try:
        from ...cleanup.stt_vibevoice import get_status as _vv_status
        vv = _vv_status()
        fc_status = vv["status"]
        fc_error = vv.get("error")
        fc_model = vv["model"]
    except Exception as e:
        fc_status = "failed"
        fc_error = str(e)
        fc_model = "microsoft/VibeVoice-ASR"

    return {
        "live_transcript": {
            "model": NEMOTRON_MODEL_NAME,
            "status": lt_status,
            "error": lt_error,
        },
        "boardroom": {
            "asr_model": PARAKEET_MODEL_NAME,
            "diar_model": diar_model,
            "status": br_status,
            "error": br_error,
        },
        "final_cleanup": {
            "model": fc_model,
            "status": fc_status,
            "error": fc_error,
        },
    }
