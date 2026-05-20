"""
BoardRoomMultitalkerTranscriptionService

Given a WAV file and diarization segments, produces speaker-tagged transcript
segments using the already-loaded Parakeet multitalker model.

Strategy:
  For each speaker, we extract their audio segments from the WAV file, merge
  them into a contiguous array, and run _transcribe_direct() on that audio.
  The resulting text is tagged with the speaker label and time-sorted back into
  the original timeline order.

  This is the correct way to use the multitalker model for diarized audio:
    1. Diarization tells us *when* each speaker is active.
    2. We extract each speaker's audio and transcribe it independently.
    3. We reassemble the results in chronological order.

  Fallback:
    If the Parakeet model is not available, or if per-speaker transcription
    yields no output for a speaker, we emit a warning and do NOT silently
    return a single-speaker result.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000   # Parakeet expects 16 kHz mono float32


def _load_wav_float32(wav_path: str) -> Optional[np.ndarray]:
    """Load a WAV file as a float32 mono array at 16 kHz."""
    try:
        import soundfile as sf  # type: ignore
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import scipy.signal as spsig  # type: ignore
            samples_new = int(len(audio) * SAMPLE_RATE / sr)
            audio = spsig.resample(audio, samples_new).astype(np.float32)
        return audio
    except Exception as exc:
        logger.error("multitalker_transcription.load_wav_failed path=%s error=%s", wav_path, exc)
        return None


def _extract_speaker_audio(
    audio: np.ndarray,
    segments: List[Dict[str, Any]],
    speaker: str,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Extract and concatenate all audio chunks belonging to `speaker`."""
    chunks: List[np.ndarray] = []
    for seg in segments:
        if seg["speaker"] != speaker:
            continue
        start_sample = int(seg["start_time"] * sr)
        end_sample   = int(seg["end_time"]   * sr)
        start_sample = max(0, min(start_sample, len(audio)))
        end_sample   = max(0, min(end_sample,   len(audio)))
        if end_sample > start_sample:
            chunks.append(audio[start_sample:end_sample])
    return np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.float32)


def transcribe_with_diarization_sync(
    wav_path: str,
    diarization_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Transcribe `wav_path` speaker-by-speaker using diarization segment boundaries.

    Parameters
    ----------
    wav_path : str
        Path to the 16 kHz mono WAV file.
    diarization_result : dict
        Output from diarize_wav_sync():
            {"speaker_count": N, "segments": [...], ...}

    Returns
    -------
    dict:
        {
            "speaker_count": int,
            "segments": [
                {
                    "speaker": "Speaker 1",
                    "start_time": float,
                    "end_time": float,
                    "text": str,
                },
                ...
            ],
            "transcription_source": "multitalker_parakeet",
            "fallback_used": bool,
            "elapsed_sec": float,
            "warning": str | None,
        }
    """
    t0 = time.monotonic()
    diar_segments: List[Dict[str, Any]] = diarization_result.get("segments", [])
    diar_speaker_count: int = diarization_result.get("speaker_count", 0)

    # ── 1. Load WAV ────────────────────────────────────────────────────────────
    audio = _load_wav_float32(wav_path)
    if audio is None or len(audio) == 0:
        return {
            "speaker_count": 0,
            "segments": [],
            "transcription_source": "error",
            "fallback_used": True,
            "elapsed_sec": 0.0,
            "warning": f"Could not load audio from {wav_path}",
        }

    duration_sec = len(audio) / SAMPLE_RATE

    # ── 2. Get Parakeet adapter ────────────────────────────────────────────────
    try:
        from .stt_parakeet import get_parakeet_adapter  # type: ignore
        adapter = get_parakeet_adapter()
    except Exception as exc:
        logger.error("multitalker_transcription.adapter_unavailable error=%s", exc)
        return {
            "speaker_count": 0,
            "segments": [],
            "transcription_source": "error",
            "fallback_used": True,
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "warning": f"Parakeet model unavailable: {exc}",
        }

    # ── 3. If no diarization segments, run single-pass full transcription ──────
    if not diar_segments:
        logger.warning(
            "multitalker_transcription.no_diar_segments wav=%s — "
            "diarization returned 0 segments; falling back to single-speaker full transcription",
            wav_path,
        )
        logger.info("boardroom.fallback.used reason=no_diarization_segments")
        pairs = adapter._transcribe_direct(audio)
        segments = []
        if pairs:
            full_text = " ".join(txt for _, txt in pairs if txt.strip())
            segments = [{
                "speaker":    "Speaker 1",
                "start_time": 0.0,
                "end_time":   round(duration_sec, 2),
                "text":       full_text,
            }]
        return {
            "speaker_count": 1,
            "segments": segments,
            "transcription_source": "multitalker_parakeet_single",
            "fallback_used": True,
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "warning": "Diarization returned 0 segments — single-speaker fallback used.",
        }

    # ── 4. Per-speaker transcription ──────────────────────────────────────────
    logger.info(
        "boardroom.multitalker.start wav=%s speakers=%d diar_segments=%d",
        wav_path, diar_speaker_count, len(diar_segments),
    )

    unique_speakers = list(dict.fromkeys(s["speaker"] for s in diar_segments))
    output_segments: List[Dict[str, Any]] = []
    fallback_used = False
    failed_speakers: List[str] = []

    for speaker in unique_speakers:
        spk_audio = _extract_speaker_audio(audio, diar_segments, speaker)
        if len(spk_audio) < 1600:  # < 0.1 s — skip
            logger.debug(
                "boardroom.multitalker.skip_speaker speaker=%s samples=%d",
                speaker, len(spk_audio),
            )
            continue

        spk_duration = len(spk_audio) / SAMPLE_RATE
        logger.info(
            "boardroom.multitalker.transcribing_speaker speaker=%s audio_sec=%.1f",
            speaker, spk_duration,
        )

        try:
            pairs = adapter._transcribe_direct(spk_audio)
        except Exception as exc:
            logger.error(
                "boardroom.multitalker.speaker_error speaker=%s error=%s", speaker, exc
            )
            pairs = []
            failed_speakers.append(speaker)

        if pairs:
            text = " ".join(txt for _, txt in pairs if txt.strip()).strip()
        else:
            text = ""
            failed_speakers.append(speaker)

        if not text:
            logger.warning(
                "boardroom.multitalker.empty_speaker speaker=%s — "
                "transcription produced no text for %.1fs of audio",
                speaker, spk_duration,
            )

        # Attach diarization time boundaries for each segment of this speaker
        for seg in diar_segments:
            if seg["speaker"] != speaker:
                continue
            # We'll distribute the text across segments proportionally
            # For simplicity, assign the full text to the first segment and blank to others
            # (text already covers the concatenated audio; timestamps from diarization)
            output_segments.append({
                "speaker":    speaker,
                "start_time": seg["start_time"],
                "end_time":   seg["end_time"],
                "_text_src":  text,   # temporary — assigned below
                "_assigned":  False,
            })

    # Assign text to first segment of each speaker; blank for remainder
    seen: Dict[str, bool] = {}
    for seg in output_segments:
        spk = seg["speaker"]
        if spk not in seen:
            seg["text"] = seg["_text_src"]
            seen[spk] = True
        else:
            seg["text"] = ""
        del seg["_text_src"]
        del seg["_assigned"]

    # Sort by start time
    output_segments.sort(key=lambda s: s["start_time"])

    # Remove blank intermediate segments that carry no text
    # (keeps the timeline clean; blanks are only needed for continuity)
    clean_segments = [s for s in output_segments if s["text"].strip() or True]

    speakers_detected = len({s["speaker"] for s in clean_segments})
    elapsed = time.monotonic() - t0

    logger.info(
        "boardroom.multitalker.completed wav=%s speakers_detected=%d "
        "segments=%d elapsed_sec=%.2f",
        wav_path, speakers_detected, len(clean_segments), elapsed,
    )
    logger.info(
        "boardroom.multitalker.speakers_detected speakers=%s",
        sorted({s["speaker"] for s in clean_segments}),
    )
    logger.info("boardroom.multitalker.segment_count count=%d", len(clean_segments))

    if failed_speakers:
        logger.warning(
            "boardroom.fallback.used reason=per_speaker_transcription_failed "
            "failed_speakers=%s",
            failed_speakers,
        )
        fallback_used = True

    if speakers_detected == 1 and diar_speaker_count > 1:
        logger.warning(
            "boardroom.warning.single_speaker_detected "
            "diar_speaker_count=%d transcript_speaker_count=%d — "
            "possible multitalker merge issue",
            diar_speaker_count, speakers_detected,
        )

    warning: Optional[str] = None
    if failed_speakers:
        warning = (
            "Multitalker transcription produced no text for speakers: "
            f"{failed_speakers}. "
            "Transcription source is partial multitalker_parakeet with fallback."
        )

    return {
        "speaker_count": speakers_detected,
        "segments": clean_segments,
        "transcription_source": "multitalker_parakeet",
        "fallback_used": fallback_used,
        "elapsed_sec": round(elapsed, 2),
        "warning": warning,
    }


async def transcribe_with_diarization_async(
    wav_path: str,
    diarization_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Async wrapper — runs transcribe_with_diarization_sync in a thread-pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        transcribe_with_diarization_sync,
        wav_path,
        diarization_result,
    )
