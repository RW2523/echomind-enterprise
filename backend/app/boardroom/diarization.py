"""
BoardRoomDiarizationService — real speaker diarization using NVIDIA Sortformer.

Model: nvidia/diar_streaming_sortformer_4spk-v2.1
  - Loaded lazily (first call to diarize()) to avoid consuming GPU RAM at startup.
  - Loaded once, then kept in memory for subsequent sessions.

Output format:
    {
        "speaker_count": 2,
        "segments": [
            {"speaker": "Speaker 1", "start_time": 0.0, "end_time": 3.5},
            {"speaker": "Speaker 2", "start_time": 3.5, "end_time": 7.2},
        ],
    }

The diarize() call is blocking (GPU inference). Call it from a thread pool via
asyncio.get_event_loop().run_in_executor() so the WS handler stays non-blocking.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Lazy-load globals ──────────────────────────────────────────────────────────
_model: Optional[Any] = None
_model_lock = threading.Lock()
_load_error: Optional[str] = None   # set once if load fails permanently


def _resolve_nemo_file(model_name: str) -> Optional[str]:
    """
    Return the local .nemo file path for a HuggingFace model name, if cached.
    Looks in the standard HuggingFace snapshot cache.
    """
    import os
    hf_cache = os.path.expanduser(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    )
    hub_dir = os.path.join(hf_cache, "hub")
    # model id → directory name: replace / with --
    dir_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = os.path.join(hub_dir, dir_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    for snap in sorted(os.listdir(snapshots_dir), reverse=True):
        snap_path = os.path.join(snapshots_dir, snap)
        for f in os.listdir(snap_path):
            if f.endswith(".nemo"):
                return os.path.join(snap_path, f)
    return None


def _load_model(model_name: str) -> None:
    """
    Load the Sortformer model (blocking, call once).

    Loading strategy (in order):
      1. Try to restore directly from the local .nemo file (HF cache).
         This avoids HF_HUB_OFFLINE issues and is the fastest path.
      2. Fall back to from_pretrained() with HF_HUB_OFFLINE=0 if the local
         .nemo file isn't found (first-ever run before the Dockerfile cached it).

    Device strategy:
      We load on CPU to avoid competing with Parakeet + Nemotron for GPU RAM.
      Diarization is purely a softmax over ~90s of audio — CPU is fast enough.
    """
    global _model, _load_error
    import os
    import torch

    try:
        logger.info("boardroom.diarization.loading model=%s", model_name)
        import nemo.collections.asr as nemo_asr  # type: ignore

        # Attempt 1: restore from local .nemo file (no HF network needed)
        nemo_path = _resolve_nemo_file(model_name)
        if nemo_path:
            logger.info("boardroom.diarization.local_nemo path=%s", nemo_path)
            m = nemo_asr.models.SortformerEncLabelModel.restore_from(
                restore_path=nemo_path,
                map_location="cpu",  # diarization runs fine on CPU
            )
        else:
            # Attempt 2: from_pretrained, temporarily allowing online access
            original_offline = os.environ.get("HF_HUB_OFFLINE", "")
            if original_offline in ("1", "true", "yes"):
                logger.warning(
                    "boardroom.diarization.offline_miss model=%s — "
                    "no local .nemo found; temporarily enabling HF hub to download",
                    model_name,
                )
                os.environ["HF_HUB_OFFLINE"] = "0"
            try:
                m = nemo_asr.models.SortformerEncLabelModel.from_pretrained(
                    model_name, map_location="cpu"
                )
            finally:
                if original_offline in ("1", "true", "yes"):
                    os.environ["HF_HUB_OFFLINE"] = original_offline

        m.eval()
        _model = m
        logger.info(
            "boardroom.diarization.loaded model=%s device=cpu", model_name
        )
    except Exception as exc:
        _load_error = str(exc)
        logger.error("boardroom.diarization.load_failed model=%s error=%s", model_name, exc)


def _ensure_model(model_name: str) -> None:
    global _model
    if _model is not None:
        return
    with _model_lock:
        if _model is None and _load_error is None:
            _load_model(model_name)
    if _model is None:
        raise RuntimeError(
            f"Sortformer model could not be loaded: {_load_error or 'unknown error'}"
        )


# ── RTTM / output parsing ──────────────────────────────────────────────────────

def _parse_diar_output(raw: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Convert Sortformer diarize() output to a list of segment dicts.

    Sortformer returns:  [["start end speaker_label", ...], ...]
    where the outer list is per-audio-file, inner list is per-segment.
    """
    segments: List[Dict[str, Any]] = []
    # Map internal speaker labels (speaker_0, speaker_1 …) to friendly names
    label_map: Dict[str, str] = {}
    counter = 1

    for session in raw:
        for line in session:
            line = line.strip()
            if not line:
                continue
            # Expected format: "start_sec end_sec speaker_label"
            parts = line.split()
            if len(parts) < 3:
                logger.debug("boardroom.diarization.skip_line line=%r", line)
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                raw_label = parts[2]
            except ValueError:
                logger.debug("boardroom.diarization.bad_line line=%r", line)
                continue

            if raw_label not in label_map:
                label_map[raw_label] = f"Speaker {counter}"
                counter += 1
            friendly = label_map[raw_label]
            segments.append({
                "speaker": friendly,
                "raw_speaker": raw_label,
                "start_time": round(start, 3),
                "end_time":   round(end,   3),
            })

    # Sort by start time
    segments.sort(key=lambda s: s["start_time"])
    return segments


def _merge_adjacent_segments(
    segments: List[Dict[str, Any]],
    gap_tolerance_sec: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments from the same speaker if the gap between
    them is <= gap_tolerance_sec.
    """
    if not segments:
        return segments
    merged: List[Dict[str, Any]] = [dict(segments[0])]
    for seg in segments[1:]:
        last = merged[-1]
        gap = seg["start_time"] - last["end_time"]
        if seg["speaker"] == last["speaker"] and gap <= gap_tolerance_sec:
            last["end_time"] = seg["end_time"]
        else:
            merged.append(dict(seg))
    return merged


# ── Public API ─────────────────────────────────────────────────────────────────

def diarize_wav_sync(
    wav_path: str,
    model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1",
) -> Dict[str, Any]:
    """
    Run Sortformer diarization on a WAV file (blocking).

    Returns:
        {
            "speaker_count": int,
            "segments": [{"speaker": str, "start_time": float, "end_time": float}, ...],
            "model": str,
            "elapsed_sec": float,
        }

    Raises RuntimeError if the model cannot be loaded.
    """
    t0 = time.monotonic()
    _ensure_model(model_name)

    logger.info("boardroom.diarization.start wav=%s model=%s", wav_path, model_name)

    try:
        raw = _model.diarize(
            audio=wav_path,
            sample_rate=16000,
            batch_size=1,
            verbose=False,
        )
    except Exception as exc:
        logger.error("boardroom.diarization.error wav=%s error=%s", wav_path, exc)
        raise

    elapsed = time.monotonic() - t0
    segments = _parse_diar_output(raw)
    segments = _merge_adjacent_segments(segments)
    speaker_count = len({s["speaker"] for s in segments})

    logger.info(
        "boardroom.diarization.completed wav=%s elapsed_sec=%.2f "
        "speaker_count=%d segment_count=%d",
        wav_path, elapsed, speaker_count, len(segments),
    )
    logger.info(
        "boardroom.diarization.speaker_count speaker_count=%d", speaker_count,
    )

    # Preview first 5 segments
    preview = [
        {"speaker": s["speaker"], "start_time": s["start_time"], "end_time": s["end_time"]}
        for s in segments[:5]
    ]
    logger.info("boardroom.diarization.segments_preview preview=%s", preview)

    return {
        "speaker_count": speaker_count,
        "segments": [
            {
                "speaker":    s["speaker"],
                "start_time": s["start_time"],
                "end_time":   s["end_time"],
            }
            for s in segments
        ],
        "model": model_name,
        "elapsed_sec": round(elapsed, 2),
    }


async def diarize_wav_async(
    wav_path: str,
    model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2.1",
) -> Dict[str, Any]:
    """Async wrapper — runs diarize_wav_sync in a thread-pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, diarize_wav_sync, wav_path, model_name
    )


def get_status() -> Dict[str, Any]:
    """Return the loading status of the Sortformer model."""
    if _model is not None:
        return {"status": "ready", "model": "nvidia/diar_streaming_sortformer_4spk-v2.1"}
    if _load_error:
        return {"status": "error", "error": _load_error}
    return {"status": "not_loaded"}
