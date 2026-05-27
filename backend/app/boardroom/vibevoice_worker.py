#!/usr/bin/env python3
"""
Standalone VibeVoice ASR worker.

Runs as an isolated subprocess so it gets its own CUDA context and can
never corrupt the Nemotron model that is already loaded in the main process.

Usage:
    python vibevoice_worker.py <wav_path> <cache_dir>

Prints a JSON array of diarised segment dicts to stdout, then exits.
stderr is used for logging and is captured separately by the caller.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import wave

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

_VIBEVOICE_SR   = 24_000
_VIBEVOICE_MAX_S = 60 * 60          # 60-minute batch ceiling


def _ts_to_sec(ts: str) -> float:
    ts = ts.strip().rstrip("s").strip()
    try:
        return float(ts)
    except ValueError:
        pass
    parts = ts.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        pass
    return 0.0


def _parse_vibevoice_output(raw_text: str, time_offset_s: float = 0.0) -> list:
    pattern = re.compile(
        r"\[([^\]\-–]+?)\s*[-–]\s*([^\]]+?)\]\s*([^:\[\n]+?):\s*(.+?)(?=\[|\Z)",
        re.DOTALL,
    )
    segments: list = []
    for m in pattern.finditer(raw_text):
        start_s = _ts_to_sec(m.group(1)) + time_offset_s
        end_s   = _ts_to_sec(m.group(2)) + time_offset_s
        speaker = m.group(3).strip()
        text    = re.sub(r"\s+", " ", m.group(4)).strip()
        if text:
            segments.append({"speaker": speaker, "text": text,
                              "start_time": start_s, "end_time": end_s})

    if not segments and raw_text.strip():
        clean = re.sub(r"\s+", " ", raw_text).strip()
        segments = [{"speaker": "Speaker 1", "text": clean,
                     "start_time": time_offset_s, "end_time": time_offset_s}]
    return segments


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: vibevoice_worker.py <wav_path> <cache_dir>", file=sys.stderr)
        sys.exit(1)

    wav_path  = sys.argv[1]
    cache_dir = sys.argv[2]
    model_id  = "microsoft/VibeVoice-ASR-HF"

    print(f"[vv-worker] wav={wav_path}  cache={cache_dir}", file=sys.stderr, flush=True)

    # ── Load WAV → float32 @ 24 kHz mono ─────────────────────────────────────
    try:
        import scipy.signal
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            raw        = wf.readframes(wf.getnframes())
    except Exception as e:
        print(f"[vv-worker] WAV read error: {e}", file=sys.stderr)
        sys.exit(2)

    if sampwidth != 2:
        print(f"[vv-worker] Unsupported sample width {sampwidth}", file=sys.stderr)
        sys.exit(3)

    pcm = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    audio = pcm.astype(np.float32) / 32768.0

    if framerate != _VIBEVOICE_SR:
        n_out = int(len(audio) * _VIBEVOICE_SR / framerate)
        audio = scipy.signal.resample(audio, n_out).astype(np.float32)

    total_s = len(audio) / _VIBEVOICE_SR
    print(f"[vv-worker] {total_s:.1f}s of audio @ {_VIBEVOICE_SR} Hz", file=sys.stderr, flush=True)

    # ── Load VibeVoice ────────────────────────────────────────────────────────
    try:
        import torch
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

        # Use CUDA if available; this subprocess owns the GPU exclusively
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[vv-worker] device={device_str}", file=sys.stderr, flush=True)

        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=True,
        )
        model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        print("[vv-worker] model loaded", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[vv-worker] model load error: {e}", file=sys.stderr)
        sys.exit(4)

    # ── Process in ≤ 60-minute batches ───────────────────────────────────────
    max_samples   = int(_VIBEVOICE_MAX_S * _VIBEVOICE_SR)
    all_segments: list = []
    offset        = 0
    batch_idx     = 0

    try:
        import torch
        while offset < len(audio):
            batch       = audio[offset: offset + max_samples]
            time_off_s  = offset / _VIBEVOICE_SR
            batch_dur_s = len(batch) / _VIBEVOICE_SR

            print(
                f"[vv-worker] batch {batch_idx+1}: "
                f"{time_off_s/60:.1f}–{(time_off_s+batch_dur_s)/60:.1f} min",
                file=sys.stderr, flush=True,
            )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                with wave.open(tmp_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(_VIBEVOICE_SR)
                    pcm16 = (batch * 32768.0).clip(-32768, 32767).astype(np.int16)
                    wf.writeframes(pcm16.tobytes())

                # Determine which device the first parameter lives on
                try:
                    target_device = next(
                        p.device for p in model.parameters() if p.device.type != "meta"
                    )
                except StopIteration:
                    target_device = torch.device("cpu")

                inputs = processor.apply_transcription_request(
                    audio=tmp_path,
                ).to(target_device, torch.bfloat16)

                with torch.no_grad():
                    output_ids = model.generate(**inputs)

                generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
                raw_text      = processor.decode(generated_ids[0])

                batch_segs = _parse_vibevoice_output(raw_text, time_offset_s=time_off_s)
                all_segments.extend(batch_segs)
                print(
                    f"[vv-worker] batch {batch_idx+1}: "
                    f"{len(batch_segs)} segments, "
                    f"{len({s['speaker'] for s in batch_segs})} speaker(s)",
                    file=sys.stderr, flush=True,
                )
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            offset    += max_samples
            batch_idx += 1

    except Exception as e:
        print(f"[vv-worker] inference error: {e}", file=sys.stderr)
        import traceback; traceback.print_exc(file=sys.stderr)
        sys.exit(5)

    if not all_segments:
        all_segments = [{"speaker": "Speaker 1",
                         "text": "[No speech detected]",
                         "start_time": 0.0, "end_time": total_s}]

    print(
        f"[vv-worker] done: {len(all_segments)} segments, "
        f"{len({s['speaker'] for s in all_segments})} unique speaker(s)",
        file=sys.stderr, flush=True,
    )

    # ── Output JSON to stdout for the caller ──────────────────────────────────
    print(json.dumps(all_segments))


if __name__ == "__main__":
    main()
