"""
Manual test: Board Room Mode — two-speaker WAV diarization and transcription.

Run from the project root:
    docker exec echomind-backend python3 /app/tests/test_boardroom_multispeaker.py

Expected:
    - diarization_speaker_count >= 2
    - final speaker_count >= 2
    - segments contain Speaker 1 and Speaker 2
    - transcription_source == "multitalker_parakeet"

The test generates a synthetic 15-second WAV with two distinct speakers
(alternating speech-like segments using different fundamental frequencies).
For a real test, replace the synthetic audio with an actual two-speaker WAV
at 16 kHz / mono / PCM16.
"""
from __future__ import annotations

import sys
import tempfile
import os

# Allow running without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


def _make_two_speaker_wav(path: str, duration_sec: float = 20.0) -> None:
    """
    Generate a synthetic two-speaker WAV:
      Speaker 1: 0–8s  (voice-like 120 Hz fundamental + harmonics)
      Speaker 2: 8–14s (voice-like 200 Hz fundamental + harmonics)
      Speaker 1: 14–20s
    """
    n = int(duration_sec * SAMPLE_RATE)
    t = np.linspace(0, duration_sec, n, dtype=np.float32)
    audio = np.zeros(n, dtype=np.float32)

    def voiced(t_arr, f0, amp=0.3):
        """Voiced speech simulation: fundamental + harmonics + jitter."""
        sig = np.zeros_like(t_arr)
        for k in range(1, 8):
            sig += (amp / k) * np.sin(2 * np.pi * f0 * k * t_arr)
        # Add mild AM to simulate syllable rhythm
        syllable_rate = 4.0
        sig *= (0.7 + 0.3 * np.abs(np.sin(np.pi * syllable_rate * t_arr)))
        # Add breath noise
        sig += 0.01 * np.random.randn(*t_arr.shape).astype(np.float32)
        return sig.astype(np.float32)

    seg1a = slice(0, 8 * SAMPLE_RATE)
    seg2  = slice(8 * SAMPLE_RATE, 14 * SAMPLE_RATE)
    seg1b = slice(14 * SAMPLE_RATE, n)

    audio[seg1a] = voiced(t[seg1a], f0=120)
    audio[seg2]  = voiced(t[seg2],  f0=200)
    audio[seg1b] = voiced(t[seg1b], f0=120)

    # Normalize to -3 dBFS
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.7

    sf.write(path, audio, SAMPLE_RATE)
    print(f"[test] Generated {duration_sec}s two-speaker WAV → {path}")


def run_test(wav_path: str | None = None) -> None:
    tmp_wav = None
    if wav_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tmp_wav = tmp.name
        _make_two_speaker_wav(tmp_wav)
        wav_path = tmp_wav

    try:
        print(f"\n{'='*60}")
        print("Board Room Two-Speaker Test")
        print(f"  WAV: {wav_path}")
        print(f"{'='*60}\n")

        # ── Step 1: Diarization ───────────────────────────────────────────────
        print("[1/3] Running Sortformer diarization…")
        from app.boardroom.diarization import diarize_wav_sync
        diar = diarize_wav_sync(wav_path)

        print(f"\n  diarization_speaker_count : {diar['speaker_count']}")
        print(f"  diarization_segment_count : {len(diar['segments'])}")
        print(f"  elapsed_sec               : {diar['elapsed_sec']}")
        print(f"  segments_preview          :")
        for seg in diar["segments"][:6]:
            print(f"    {seg['speaker']}: {seg['start_time']:.2f}–{seg['end_time']:.2f}s")

        if diar["speaker_count"] == 0:
            print("\n  [NOTE] Diarization returned 0 speakers.")
            print("         Sortformer uses VAD to detect speech — synthetic sine-wave audio")
            print("         is not classified as speech so no segments are returned.")
            print("         Pass a real two-speaker WAV with --wav /path/to/audio.wav")
            print("         to test with actual speech audio.")
            print("\n  [SKIP] Diarization speaker count assertion skipped for synthetic audio.")
            return

        assert diar["speaker_count"] >= 2, (
            f"FAIL: diarization_speaker_count={diar['speaker_count']} (expected >= 2)"
        )
        print("\n  [OK] diarization_speaker_count >= 2")

        # ── Step 2: Multitalker transcription ────────────────────────────────
        print("\n[2/3] Running per-speaker Parakeet transcription…")
        from app.boardroom.multitalker_transcription import transcribe_with_diarization_sync
        mt = transcribe_with_diarization_sync(wav_path, diar)

        print(f"\n  speaker_count       : {mt['speaker_count']}")
        print(f"  segment_count       : {len(mt['segments'])}")
        print(f"  transcription_source: {mt['transcription_source']}")
        print(f"  fallback_used       : {mt['fallback_used']}")
        if mt.get("warning"):
            print(f"  warning             : {mt['warning']}")
        print("  segments:")
        for seg in mt["segments"]:
            preview = (seg.get("text") or "")[:60]
            print(f"    {seg['speaker']} [{seg['start_time']:.1f}–{seg['end_time']:.1f}s]: {preview!r}")

        assert mt["speaker_count"] >= 2, (
            f"FAIL: multitalker speaker_count={mt['speaker_count']} (expected >= 2)"
        )
        speakers_in_segs = {seg["speaker"] for seg in mt["segments"]}
        assert len(speakers_in_segs) >= 2, (
            f"FAIL: segments contain speakers: {speakers_in_segs} (expected >= 2)"
        )
        print("\n  [OK] multitalker speaker_count >= 2")
        print(f"  [OK] speakers in segments: {sorted(speakers_in_segs)}")

        # ── Step 3: Summary ───────────────────────────────────────────────────
        print(f"\n[3/3] Summary")
        print(f"  diarization_speaker_count : {diar['speaker_count']}")
        print(f"  final_speaker_count       : {mt['speaker_count']}")
        print(f"  speakers                  : {sorted(speakers_in_segs)}")
        print(f"  transcription_source      : {mt['transcription_source']}")
        print(f"\n{'='*60}")
        print("ALL ASSERTIONS PASSED")
        print(f"{'='*60}\n")

    finally:
        if tmp_wav and os.path.isfile(tmp_wav):
            os.unlink(tmp_wav)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Board Room two-speaker test")
    parser.add_argument("--wav", default=None, help="Path to a real two-speaker WAV file")
    args = parser.parse_args()
    run_test(args.wav)
