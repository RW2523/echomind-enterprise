"""
Boardroom Mode service.

Flow:
  1. Frontend streams audio chunks → POST /api/boardroom/{id}/chunks (multipart bytes)
  2. Backend stores chunks as raw files on disk under DATA_DIR/boardroom/{session_id}/
  3. POST /api/boardroom/{id}/finalize → concatenate + transcribe with ASR
     Primary: microsoft/VibeVoice-ASR via HuggingFace transformers
     Fallback: existing Nemotron ASR adapter (no speaker labels)
  4. POST /api/boardroom/{id}/analyse → RAG + LLM meeting report
  5. GET /api/boardroom/{id}/export?format=pdf|pptx → file download
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import struct
import subprocess
import tempfile
import wave
from typing import Optional

import numpy as np

from ..core.config import settings
from ..core.db import get_conn
from ..rag.index import index as faiss_index
from ..rag.llm import OpenAICompatChat
from ..utils.ids import new_id, now_iso

logger = logging.getLogger(__name__)

_BOARDROOM_DIR = os.path.join(settings.DATA_DIR, "boardroom")

# VibeVoice-ASR-HF is stored in the echomind_data volume (populated once via docker-compose.prepare.yml).
# It is NOT baked into the Docker image because the model is ~18 GB.
_VIBEVOICE_CACHE_DIR = os.path.join(settings.DATA_DIR, "hf_cache")
_VIBEVOICE_MODEL_ID  = "microsoft/VibeVoice-ASR-HF"
_VIBEVOICE_SR        = 24_000   # model native sample rate (Hz)
_VIBEVOICE_MAX_S     = 60 * 60  # 60-minute batch limit (seconds)

_llm: Optional[OpenAICompatChat] = None


def _get_llm() -> OpenAICompatChat:
    global _llm
    if _llm is None:
        _llm = OpenAICompatChat(settings.LLM_BASE_URL, settings.LLM_MODEL)
    return _llm


# ── DB helpers ────────────────────────────────────────────────────────────────

def create_session(transcript_id: Optional[str] = None) -> str:
    """Insert a new boardroom_sessions row and return the session id."""
    sid = new_id("brm")
    os.makedirs(os.path.join(_BOARDROOM_DIR, sid), exist_ok=True)
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO boardroom_sessions
               (id, transcript_id, status, chunk_count, audio_format, created_at, updated_at)
               VALUES (?, ?, 'recording', 0, 'webm', ?, ?)""",
            (sid, transcript_id, now_iso(), now_iso()),
        )
        conn.commit()
    return sid


def get_session(session_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT id, transcript_id, status, chunk_count, audio_format,
                      diarized_json, report_json, created_at, updated_at
               FROM boardroom_sessions WHERE id = ?""",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    sid, tid, status, chunk_count, afmt, diar_json, report_json, created_at, updated_at = row
    return {
        "id": sid,
        "transcript_id": tid,
        "status": status,
        "chunk_count": chunk_count or 0,
        "audio_format": afmt or "webm",
        "diarized_segments": json.loads(diar_json) if diar_json else None,
        "report": json.loads(report_json) if report_json else None,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def list_sessions(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, transcript_id, status, chunk_count, created_at, updated_at
               FROM boardroom_sessions ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r[0],
            "transcript_id": r[1],
            "status": r[2],
            "chunk_count": r[3] or 0,
            "created_at": r[4],
            "updated_at": r[5],
        }
        for r in rows
    ]


def _update_status(session_id: str, status: str, **extra) -> None:
    sets = ["status = ?", "updated_at = ?"]
    vals: list = [status, now_iso()]
    for k, v in extra.items():
        sets.append(f"{k} = ?")
        vals.append(v)
    vals.append(session_id)
    with get_conn() as conn:
        conn.execute(
            f"UPDATE boardroom_sessions SET {', '.join(sets)} WHERE id = ?",
            vals,
        )
        conn.commit()


# ── Audio chunk storage ───────────────────────────────────────────────────────

def store_chunk(session_id: str, chunk_bytes: bytes, chunk_index: int, audio_format: str = "webm") -> int:
    """Write a chunk to disk. Returns updated chunk count."""
    chunk_dir = os.path.join(_BOARDROOM_DIR, session_id)
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:06d}.{audio_format}")
    with open(chunk_path, "wb") as f:
        f.write(chunk_bytes)
    with get_conn() as conn:
        conn.execute(
            "UPDATE boardroom_sessions SET chunk_count = ?, audio_format = ?, updated_at = ? WHERE id = ?",
            (chunk_index + 1, audio_format, now_iso(), session_id),
        )
        conn.commit()
    return chunk_index + 1


# ── ASR ───────────────────────────────────────────────────────────────────────

def _ffmpeg_to_wav16k(src_path: str, dst_path: str) -> bool:
    """Convert any audio file to 16 kHz mono PCM WAV using ffmpeg. Returns True on success."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-ar", "16000",
                "-ac", "1",
                "-sample_fmt", "s16",
                "-f", "wav",
                dst_path,
            ],
            capture_output=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning("ffmpeg conversion failed for %s: %s", src_path, e)
        return False


def _concat_chunks_to_wav(session_id: str, audio_format: str) -> Optional[bytes]:
    """
    Concatenate all stored audio chunks into a single 16 kHz mono WAV file.

    MediaRecorder emits a WebM stream split across many chunks: chunk 0 contains
    the EBML header + Tracks, subsequent chunks contain only cluster data and are
    NOT individually valid WebM files.  The correct approach is to concatenate
    all chunk bytes in order into one binary file and then convert that combined
    file with ffmpeg in a single pass.
    """
    chunk_dir = os.path.join(_BOARDROOM_DIR, session_id)
    if not os.path.isdir(chunk_dir):
        return None

    chunk_files = sorted(
        f for f in os.listdir(chunk_dir) if f.startswith("chunk_")
    )
    if not chunk_files:
        return None

    ext = audio_format or "webm"
    # Strip any existing extension from the stored filenames to detect WAV-only sessions
    non_wav = [f for f in chunk_files if not f.endswith(".wav")]

    with tempfile.TemporaryDirectory() as tmpdir:
        if non_wav:
            # Concatenate ALL chunk bytes in index order into one container file,
            # then run ffmpeg once on the combined stream.
            combined_path = os.path.join(tmpdir, f"combined.{ext}")
            total_bytes = 0
            with open(combined_path, "wb") as out:
                for fname in chunk_files:
                    with open(os.path.join(chunk_dir, fname), "rb") as src:
                        data = src.read()
                        out.write(data)
                        total_bytes += len(data)

            logger.info(
                "Combined %d chunks (%d bytes) for session %s — converting via ffmpeg",
                len(chunk_files), total_bytes, session_id,
            )

            wav_path = os.path.join(tmpdir, "output.wav")
            if not _ffmpeg_to_wav16k(combined_path, wav_path):
                logger.error(
                    "ffmpeg failed to convert combined audio for session %s", session_id
                )
                return None

            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
            logger.info(
                "ffmpeg conversion ok → %d bytes WAV for session %s",
                len(wav_bytes), session_id,
            )
            return wav_bytes

        else:
            # All chunks are already WAV — concatenate raw PCM frames.
            all_frames: list[bytes] = []
            params = None
            for fname in chunk_files:
                try:
                    with wave.open(os.path.join(chunk_dir, fname), "rb") as wf:
                        if params is None:
                            params = wf.getparams()
                        all_frames.append(wf.readframes(wf.getnframes()))
                except Exception as e:
                    logger.warning("Failed to read WAV chunk %s: %s", fname, e)

            if not all_frames or params is None:
                return None

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setparams(params)
                for frames in all_frames:
                    wf.writeframes(frames)
            return buf.getvalue()


# ── VibeVoice output parser ───────────────────────────────────────────────────

def _parse_vibevoice_output(raw_text: str, time_offset_s: float = 0.0) -> list:
    """
    Parse VibeVoice structured output into our standard segment list.

    VibeVoice emits lines like:
        [00:00 - 00:05] Speaker 1: Hello, welcome.
        [0.00s - 5.23s] SPEAKER_1: Hi there.
        [00:00:00 - 00:00:05] Speaker 1: ...

    Multiple timestamp/speaker formats are handled; raw text is the fallback.
    """
    import re

    def _ts_to_sec(ts: str) -> float:
        ts = ts.strip().rstrip("s").strip()
        try:
            return float(ts)          # "5.23"
        except ValueError:
            pass
        parts = ts.split(":")
        try:
            if len(parts) == 2:       # MM:SS
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:       # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except (ValueError, IndexError):
            pass
        return 0.0

    # Greedy: match everything up to the next '[' or end-of-string
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
            segments.append({
                "speaker":    speaker,
                "text":       text,
                "start_time": start_s,
                "end_time":   end_s,
            })

    # If the regex found nothing, treat the entire output as one speaker
    if not segments and raw_text.strip():
        clean = re.sub(r"\s+", " ", raw_text).strip()
        segments = [{
            "speaker":    "Speaker 1",
            "text":       clean,
            "start_time": time_offset_s,
            "end_time":   time_offset_s,
        }]

    return segments


# ── Speaker diarisation helpers ───────────────────────────────────────────────

def _vad_segment(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = 20,
    min_speech_ms: int = 300,
    min_silence_ms: int = 700,
    energy_pct: float = 12.0,
) -> list:
    """Energy-based VAD: returns list of {start, end, audio} dicts."""
    frame_sz = int(sr * frame_ms / 1000)
    if frame_sz < 1 or len(audio) < frame_sz:
        return [{"start": 0.0, "end": len(audio) / sr, "audio": audio.copy()}]

    n_frames = len(audio) // frame_sz
    energies = np.array([
        float(np.sqrt(np.mean(audio[i * frame_sz:(i + 1) * frame_sz].astype(np.float64) ** 2)))
        for i in range(n_frames)
    ])

    # Adaptive threshold: percentile × multiplier, with a hard floor
    thresh = max(np.percentile(energies, energy_pct) * 3.0, 0.004)
    is_sp = energies > thresh

    min_sp_fr = max(1, int(min_speech_ms / frame_ms))
    min_sil_fr = max(1, int(min_silence_ms / frame_ms))

    utterances: list = []
    in_utt, utt_start, sil_cnt = False, 0, 0

    for i, s in enumerate(is_sp):
        if s:
            if not in_utt:
                utt_start, in_utt = i, True
            sil_cnt = 0
        else:
            if in_utt:
                sil_cnt += 1
                if sil_cnt >= min_sil_fr:
                    end = i - sil_cnt
                    if (end - utt_start) >= min_sp_fr:
                        utterances.append({
                            "start": utt_start * frame_sz / sr,
                            "end": end * frame_sz / sr,
                            "audio": audio[utt_start * frame_sz: end * frame_sz].copy(),
                        })
                    in_utt, sil_cnt = False, 0

    if in_utt and (n_frames - utt_start) >= min_sp_fr:
        utterances.append({
            "start": utt_start * frame_sz / sr,
            "end": n_frames * frame_sz / sr,
            "audio": audio[utt_start * frame_sz:].copy(),
        })

    # Safety: if VAD found nothing, treat the whole audio as one utterance
    if not utterances:
        utterances = [{"start": 0.0, "end": len(audio) / sr, "audio": audio.copy()}]

    return utterances


def _spectral_voice_features(audio: np.ndarray, sr: int, n_bands: int = 32) -> np.ndarray:
    """
    Log-spaced spectral energy bands (80 Hz – 7 600 Hz).
    Captures pitch range and formant structure well enough for speaker separation.
    """
    if len(audio) < 256:
        return np.zeros(n_bands)

    n_fft = min(4096, len(audio))
    # Use the middle portion of the utterance — more stable than edges
    mid = max(0, (len(audio) - n_fft) // 2)
    frame = audio[mid: mid + n_fft]
    if len(frame) < n_fft:
        frame = np.pad(frame, (0, n_fft - len(frame)))

    win = np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(frame * win)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    edges = np.logspace(np.log10(80), np.log10(7_600), n_bands + 1)
    feats = np.zeros(n_bands)
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if mask.any():
            feats[i] = float(spectrum[mask].mean())

    norm = np.linalg.norm(feats)
    return feats / norm if norm > 1e-8 else feats


def _cluster_speakers(
    utterances: list,
    sr: int,
    similarity_thresh: float = 0.88,
    max_speakers: int = 8,
) -> list:
    """
    Online cosine-similarity clustering.

    Each utterance is compared to the running mean prototype of every known
    speaker. If cosine similarity ≥ similarity_thresh it is assigned to that
    speaker; otherwise a new speaker is created (up to max_speakers).
    """
    feats = [_spectral_voice_features(u["audio"], sr) for u in utterances]

    # prototypes: list of [speaker_id, mean_feat_vector, count]
    prototypes: list = []

    for i, u in enumerate(utterances):
        f = feats[i]
        f_norm = np.linalg.norm(f)

        if f_norm < 1e-8:
            # Effectively silent — inherit last speaker
            u["speaker"] = f"Speaker {prototypes[-1][0]}" if prototypes else "Speaker 1"
            continue

        best_id, best_sim = None, -1.0
        for spk_id, proto, _cnt in prototypes:
            p_norm = np.linalg.norm(proto)
            if p_norm < 1e-8:
                continue
            sim = float(np.dot(f, proto) / (f_norm * p_norm))
            if sim > best_sim:
                best_sim, best_id = sim, spk_id

        if best_id is None or (best_sim < similarity_thresh and len(prototypes) < max_speakers):
            new_id = len(prototypes) + 1
            prototypes.append([new_id, f.copy(), 1])
            u["speaker"] = f"Speaker {new_id}"
        else:
            u["speaker"] = f"Speaker {best_id}"
            # Update prototype (running mean, re-normalised)
            idx = next(j for j, p in enumerate(prototypes) if p[0] == best_id)
            old_id, old_proto, old_cnt = prototypes[idx]
            new_cnt = old_cnt + 1
            new_proto = (old_proto * old_cnt + f) / new_cnt
            n = np.linalg.norm(new_proto)
            prototypes[idx] = [old_id, new_proto / n if n > 1e-8 else new_proto, new_cnt]

    return utterances


def _merge_speaker_segments(utterances: list, max_gap_s: float = 2.5) -> list:
    """Merge consecutive utterances from the same speaker when the gap is small."""
    if not utterances:
        return []

    result: list = []
    cur = {
        "speaker":    utterances[0]["speaker"],
        "text":       utterances[0].get("text", ""),
        "start_time": utterances[0]["start"],
        "end_time":   utterances[0]["end"],
    }

    for u in utterances[1:]:
        gap = u["start"] - cur["end_time"]
        if u["speaker"] == cur["speaker"] and gap <= max_gap_s:
            cur["text"] = (cur["text"] + " " + u.get("text", "")).strip()
            cur["end_time"] = u["end"]
        else:
            if cur["text"].strip():
                result.append(cur)
            cur = {
                "speaker":    u["speaker"],
                "text":       u.get("text", ""),
                "start_time": u["start"],
                "end_time":   u["end"],
            }

    if cur["text"].strip():
        result.append(cur)

    return result


def _is_vibevoice_cached() -> bool:
    model_dir = os.path.join(_VIBEVOICE_CACHE_DIR, "models--microsoft--VibeVoice-ASR-HF")
    return os.path.isdir(model_dir)


async def _transcribe_with_vibevoice(wav_bytes: bytes) -> list[dict]:
    """
    Primary boardroom transcription using microsoft/VibeVoice-ASR-HF.

    VibeVoice runs in a **completely isolated subprocess** so it gets its own
    CUDA context and can never corrupt the Nemotron model that is already loaded
    in the main process.  The worker script (vibevoice_worker.py) loads the
    model, processes audio, prints JSON to stdout, and exits — releasing all GPU
    memory automatically when the process terminates.

    Falls back to the Nemotron pipeline if the model is not yet downloaded.
    """
    if not _is_vibevoice_cached():
        logger.info(
            "VibeVoice-ASR-HF not found in %s — using Nemotron. "
            "Run the prepare step to download the model.",
            _VIBEVOICE_CACHE_DIR,
        )
        return await _transcribe_with_nemotron(wav_bytes)

    # Write audio to a temp file the subprocess can read
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        wav_path = tmp.name

    worker_script = os.path.join(os.path.dirname(__file__), "vibevoice_worker.py")

    try:
        logger.info("VibeVoice: launching isolated subprocess worker")
        loop   = asyncio.get_running_loop()

        def _run_worker():
            return subprocess.run(
                ["python", worker_script, wav_path, _VIBEVOICE_CACHE_DIR],
                capture_output=True,
                text=True,
                timeout=1800,   # 30-minute ceiling for very long meetings
            )

        proc = await loop.run_in_executor(None, _run_worker)

        # Forward worker stderr to our logger for visibility
        if proc.stderr:
            for line in proc.stderr.strip().splitlines():
                logger.info("VibeVoice worker: %s", line)

        if proc.returncode != 0:
            logger.error(
                "VibeVoice worker exited with code %d — falling back to Nemotron",
                proc.returncode,
            )
            return await _transcribe_with_nemotron(wav_bytes)

        segments: list = json.loads(proc.stdout.strip())
        unique_spk = len({s["speaker"] for s in segments})
        logger.info(
            "VibeVoice complete: %d segment(s), %d unique speaker(s)",
            len(segments), unique_spk,
        )
        return segments

    except subprocess.TimeoutExpired:
        logger.error("VibeVoice worker timed out — falling back to Nemotron")
        return await _transcribe_with_nemotron(wav_bytes)
    except Exception as e:
        logger.error("VibeVoice worker error: %s — falling back to Nemotron", e, exc_info=True)
        return await _transcribe_with_nemotron(wav_bytes)
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass


async def _transcribe_with_nemotron(wav_bytes: bytes) -> list[dict]:
    """
    Boardroom transcription pipeline:
      1. Load WAV → normalise to float32 @ 16 kHz mono
      2. VAD  — energy-based utterance segmentation (real speech boundaries)
      3. ASR  — Nemotron streaming model on each utterance (fresh state per utterance)
      4. Diarisation — spectral feature clustering assigns speaker IDs based on voice
      5. Merge — consecutive same-speaker utterances within a short gap are joined
    """
    loop = asyncio.get_running_loop()

    def _run_nemotron():
        try:
            from ..transcribe.stt_streaming import (
                NEMOTRON_AVAILABLE,
                get_shared_asr_adapter,
                NEMOTRON_SAMPLE_RATE,
            )
            import scipy.signal

            if not NEMOTRON_AVAILABLE:
                return [{"speaker": "Speaker 1", "text": "[ASR unavailable]",
                         "start_time": 0, "end_time": 0}]

            adapter = get_shared_asr_adapter()

            # ── Load WAV ──────────────────────────────────────────────────────
            with io.BytesIO(wav_bytes) as buf:
                with wave.open(buf, "rb") as wf:
                    n_channels = wf.getnchannels()
                    sampwidth  = wf.getsampwidth()
                    framerate  = wf.getframerate()
                    raw        = wf.readframes(wf.getnframes())

            if sampwidth != 2:
                return [{"speaker": "Speaker 1", "text": "[Unsupported PCM width]",
                         "start_time": 0, "end_time": 0}]

            pcm = np.frombuffer(raw, dtype=np.int16)
            if n_channels > 1:
                pcm = pcm.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

            audio = pcm.astype(np.float32) / 32768.0
            if framerate != NEMOTRON_SAMPLE_RATE:
                n_out = int(len(audio) * NEMOTRON_SAMPLE_RATE / framerate)
                audio = scipy.signal.resample(audio, n_out).astype(np.float32)

            total_dur = len(audio) / NEMOTRON_SAMPLE_RATE
            logger.info("Boardroom audio loaded: %.1f s @ %d Hz", total_dur, NEMOTRON_SAMPLE_RATE)

            # ── Step 1: VAD ───────────────────────────────────────────────────
            utterances = _vad_segment(audio, NEMOTRON_SAMPLE_RATE)
            logger.info("VAD: %d utterances found in %.1f s of audio", len(utterances), total_dur)

            # ── Step 2: Per-utterance ASR ─────────────────────────────────────
            FRAME_SAMPLES = int(NEMOTRON_SAMPLE_RATE * 0.560)  # 560 ms = Nemotron chunk size

            for utt in utterances:
                seg = utt["audio"]
                if len(seg) < FRAME_SAMPLES // 4:
                    utt["text"] = ""
                    continue

                state    = adapter.create_session_state()
                step_num = 0
                last_hyp = ""

                for fi in range(0, len(seg), FRAME_SAMPLES):
                    frame   = seg[fi: fi + FRAME_SAMPLES]
                    is_last = (fi + FRAME_SAMPLES) >= len(seg)
                    try:
                        hyp, state = adapter.process_chunk(
                            frame, state,
                            keep_all_outputs=is_last,
                            step_num=step_num,
                        )
                        if hyp and hyp.strip():
                            last_hyp = hyp
                    except Exception as e:
                        logger.warning("Nemotron frame error (step %d): %s", step_num, e)
                    step_num += 1

                utt["text"] = last_hyp.strip()

            # ── Step 3: Speaker clustering ────────────────────────────────────
            utterances = _cluster_speakers(utterances, NEMOTRON_SAMPLE_RATE)

            # ── Step 4: Merge consecutive same-speaker turns ──────────────────
            segments = _merge_speaker_segments(utterances)

            unique_spk = len(set(s["speaker"] for s in segments))
            logger.info(
                "Boardroom diarisation complete: %d segments, %d unique speaker(s)",
                len(segments), unique_spk,
            )

            return segments or [{"speaker": "Speaker 1", "text": "[No speech detected]",
                                  "start_time": 0, "end_time": total_dur}]

        except Exception as e:
            logger.error("Nemotron boardroom transcription error: %s", e, exc_info=True)
            return [{"speaker": "Speaker 1", "text": f"[Transcription error: {e}]",
                     "start_time": 0, "end_time": 0}]

    return await loop.run_in_executor(None, _run_nemotron)


async def finalize_and_transcribe(session_id: str) -> list[dict]:
    """Concatenate audio chunks and run ASR. Updates DB with diarized_json."""
    _update_status(session_id, "processing")
    session = get_session(session_id)
    if not session:
        raise ValueError(f"Boardroom session not found: {session_id}")

    audio_format = session.get("audio_format") or "webm"
    wav_bytes = await asyncio.get_event_loop().run_in_executor(
        None, _concat_chunks_to_wav, session_id, audio_format
    )

    if not wav_bytes:
        _update_status(session_id, "error")
        raise RuntimeError("No audio chunks found for this boardroom session")

    segments = await _transcribe_with_vibevoice(wav_bytes)

    diarized_json = json.dumps(segments)
    _update_status(session_id, "transcribed", diarized_json=diarized_json)
    return segments


# ── Report generation ─────────────────────────────────────────────────────────

_REPORT_SYSTEM = (
    "You are an expert meeting analyst. You receive a full diarized meeting transcript "
    "and relevant reference document excerpts. Produce a structured JSON meeting report.\n\n"
    "Return ONLY a JSON object with no markdown:\n"
    "{\n"
    '  "executive_summary": "<2-3 sentence summary>",\n'
    '  "speakers": [{"speaker": "Speaker 1", "summary": "...", "key_points": ["..."]}],\n'
    '  "key_topics": ["topic1", "topic2"],\n'
    '  "rag_verified_facts": ["Fact verified against documents: ..."],\n'
    '  "contradictions": ["Claim X was made but document Y states Z"],\n'
    '  "recommendations": ["Action item 1"],\n'
    '  "overall_sentiment": "positive|neutral|mixed|negative"\n'
    "}"
)


async def analyse_meeting(session_id: str) -> dict:
    """Run RAG + LLM analysis on the diarized transcript and store a meeting report."""
    session = get_session(session_id)
    if not session:
        raise ValueError(f"Session not found: {session_id}")
    segments = session.get("diarized_segments") or []
    if not segments:
        raise ValueError("No diarized transcript available. Run finalize first.")

    full_transcript = "\n".join(
        f"{s['speaker']}: {s['text']}" for s in segments
    )

    # RAG lookup for the meeting content
    rag_context = ""
    try:
        hits = await faiss_index.search(full_transcript[:500], k=6)
        doc_hits = [h for h in hits if not (h.get("source") or {}).get("filename", "").startswith("transcript_")]
        if doc_hits:
            parts = []
            for i, h in enumerate(doc_hits[:5]):
                txt = (h.get("text") or "")[:400]
                doc = (h.get("source") or {}).get("filename") or ""
                parts.append(f"[Doc {i+1}, {doc}]: {txt}")
            rag_context = "\n\n".join(parts)
    except Exception as e:
        logger.warning("RAG lookup for boardroom report failed: %s", e)

    user_msg = (
        f"Meeting transcript ({len(segments)} turns):\n\n{full_transcript[:6000]}"
        + (f"\n\nRelevant reference documents:\n{rag_context}" if rag_context else "")
    )

    llm = _get_llm()
    raw = await llm.chat(
        [
            {"role": "system", "content": _REPORT_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    import re
    report: dict = {}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            report = json.loads(match.group())
        except Exception:
            report = {"executive_summary": raw, "raw_llm": raw}
    else:
        report = {"executive_summary": raw}

    report["raw_transcript"] = full_transcript

    report_json = json.dumps(report)
    _update_status(session_id, "analysed", report_json=report_json)
    return report


# ── Export ────────────────────────────────────────────────────────────────────

def export_pdf(session_id: str) -> bytes:
    """Generate a PDF report from the boardroom session analysis."""
    session = get_session(session_id)
    if not session:
        raise ValueError("Session not found")
    report = session.get("report") or {}
    segments = session.get("diarized_segments") or []

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        story = []

        h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=12, textColor=colors.HexColor("#1e293b"))
        h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceAfter=8, textColor=colors.HexColor("#334155"))
        body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11, spaceAfter=6, leading=16)
        tag_style = ParagraphStyle("Tag", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#64748b"))

        story.append(Paragraph("EchoMind Boardroom Report", h1))
        story.append(Paragraph(f"Session ID: {session_id} | Generated: {now_iso()}", tag_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
        story.append(Spacer(1, 12))

        if report.get("executive_summary"):
            story.append(Paragraph("Executive Summary", h2))
            story.append(Paragraph(str(report["executive_summary"]), body))
            story.append(Spacer(1, 8))

        if report.get("key_topics"):
            story.append(Paragraph("Key Topics", h2))
            for topic in report["key_topics"]:
                story.append(Paragraph(f"• {topic}", body))
            story.append(Spacer(1, 8))

        if report.get("speakers"):
            story.append(Paragraph("Speaker Breakdown", h2))
            for spk in report["speakers"]:
                story.append(Paragraph(f"<b>{spk.get('speaker', 'Speaker')}</b>: {spk.get('summary', '')}", body))
                for pt in (spk.get("key_points") or []):
                    story.append(Paragraph(f"  — {pt}", body))
            story.append(Spacer(1, 8))

        if report.get("rag_verified_facts"):
            story.append(Paragraph("RAG-Verified Facts", h2))
            for fact in report["rag_verified_facts"]:
                story.append(Paragraph(f"✓ {fact}", body))
            story.append(Spacer(1, 8))

        if report.get("contradictions"):
            story.append(Paragraph("Contradictions / Risks", h2))
            for c in report["contradictions"]:
                story.append(Paragraph(f"⚠ {c}", body))
            story.append(Spacer(1, 8))

        if report.get("recommendations"):
            story.append(Paragraph("Recommendations", h2))
            for rec in report["recommendations"]:
                story.append(Paragraph(f"→ {rec}", body))
            story.append(Spacer(1, 12))

        if segments:
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
            story.append(Spacer(1, 8))
            story.append(Paragraph("Full Transcript", h2))
            for seg in segments:
                spk = seg.get("speaker", "Speaker")
                txt = seg.get("text", "")
                ts = seg.get("start_time")
                ts_label = f" [{ts:.1f}s]" if ts is not None else ""
                story.append(Paragraph(f"<b>{spk}{ts_label}:</b> {txt}", body))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        # Fallback: simple plain-text PDF using fpdf2 or raw bytes
        return _export_plain_pdf(session_id, report, segments)


def _export_plain_pdf(session_id: str, report: dict, segments: list) -> bytes:
    """Minimal plain-text PDF without reportlab."""
    lines = [
        b"%PDF-1.4",
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj",
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj",
    ]
    content = "EchoMind Boardroom Report\n"
    content += f"Session: {session_id}\n\n"
    if report.get("executive_summary"):
        content += f"Summary: {report['executive_summary']}\n\n"
    for seg in segments:
        content += f"{seg.get('speaker')}: {seg.get('text')}\n"
    c_bytes = content.encode("latin-1", errors="replace")
    lines.append(b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj")
    lines.append(b"xref\n0 4")
    lines.append(b"trailer<</Size 4/Root 1 0 R>>")
    lines.append(b"startxref\n0")
    lines.append(b"%%EOF")
    return b"\n".join(lines)


def export_pptx(session_id: str) -> bytes:
    """Generate a PowerPoint presentation from the boardroom session analysis."""
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    session = get_session(session_id)
    if not session:
        raise ValueError("Session not found")
    report = session.get("report") or {}
    segments = session.get("diarized_segments") or []

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    DARK = RGBColor(0x0F, 0x17, 0x2A)
    CYAN = RGBColor(0x22, 0xD3, 0xEE)
    WHITE = RGBColor(0xFF, 0xFF, 0xFF)
    MUTED = RGBColor(0x94, 0xA3, 0xB8)

    def _add_slide(layout_idx=6):
        layout = prs.slide_layouts[layout_idx] if len(prs.slide_layouts) > layout_idx else prs.slide_layouts[0]
        slide = prs.slides.add_slide(layout)
        fill = slide.background.fill
        fill.solid()
        fill.fore_color.rgb = DARK
        for ph in slide.placeholders:
            ph.element.getparent().remove(ph.element)
        return slide

    def _add_textbox(slide, text, left, top, width, height, bold=False, size=18, color=WHITE, align=PP_ALIGN.LEFT):
        txb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = txb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = align
        run = p.add_run()
        run.text = text
        run.font.bold = bold
        run.font.size = Pt(size)
        run.font.color.rgb = color
        return txb

    # Slide 1: Title
    slide = _add_slide()
    _add_textbox(slide, "EchoMind", 0.5, 0.5, 12, 1.2, bold=True, size=40, color=CYAN, align=PP_ALIGN.CENTER)
    _add_textbox(slide, "Boardroom Intelligence Report", 0.5, 1.8, 12, 1, bold=False, size=24, color=WHITE, align=PP_ALIGN.CENTER)
    _add_textbox(slide, f"Session: {session_id[:16]}…  |  {now_iso()[:10]}", 0.5, 3, 12, 0.6, size=13, color=MUTED, align=PP_ALIGN.CENTER)

    # Slide 2: Executive Summary
    if report.get("executive_summary"):
        slide = _add_slide()
        _add_textbox(slide, "Executive Summary", 0.5, 0.4, 12, 0.8, bold=True, size=28, color=CYAN)
        _add_textbox(slide, str(report["executive_summary"])[:800], 0.5, 1.4, 12.3, 5.5, size=16)

    # Slide 3: Key Topics
    if report.get("key_topics"):
        slide = _add_slide()
        _add_textbox(slide, "Key Topics", 0.5, 0.4, 12, 0.8, bold=True, size=28, color=CYAN)
        topics_text = "\n".join(f"• {t}" for t in report["key_topics"][:12])
        _add_textbox(slide, topics_text, 0.5, 1.4, 12.3, 5.5, size=16)

    # Slide 4: Speaker Breakdown
    if report.get("speakers"):
        slide = _add_slide()
        _add_textbox(slide, "Speaker Breakdown", 0.5, 0.4, 12, 0.8, bold=True, size=28, color=CYAN)
        y = 1.4
        for spk_data in report["speakers"][:4]:
            name = spk_data.get("speaker", "Speaker")
            summary = spk_data.get("summary", "")
            _add_textbox(slide, f"{name}: {summary}"[:200], 0.5, y, 12.3, 1.1, size=14)
            y += 1.2

    # Slide 5: RAG-Verified Facts
    if report.get("rag_verified_facts"):
        slide = _add_slide()
        _add_textbox(slide, "RAG-Verified Facts", 0.5, 0.4, 12, 0.8, bold=True, size=28, color=CYAN)
        facts_text = "\n".join(f"✓ {f}" for f in report["rag_verified_facts"][:8])
        _add_textbox(slide, facts_text, 0.5, 1.4, 12.3, 5.5, size=15, color=RGBColor(0x34, 0xD3, 0x99))

    # Slide 6: Recommendations
    if report.get("recommendations"):
        slide = _add_slide()
        _add_textbox(slide, "Recommendations", 0.5, 0.4, 12, 0.8, bold=True, size=28, color=CYAN)
        recs_text = "\n".join(f"→ {r}" for r in report["recommendations"][:8])
        _add_textbox(slide, recs_text, 0.5, 1.4, 12.3, 5.5, size=15)

    # Slide 7+: Transcript snippet
    if segments:
        slide = _add_slide()
        _add_textbox(slide, "Meeting Transcript", 0.5, 0.4, 12, 0.8, bold=True, size=28, color=CYAN)
        transcript_lines = [f"{s['speaker']}: {s['text']}" for s in segments[:15]]
        _add_textbox(slide, "\n".join(transcript_lines)[:900], 0.5, 1.4, 12.3, 5.5, size=12, color=MUTED)

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()
