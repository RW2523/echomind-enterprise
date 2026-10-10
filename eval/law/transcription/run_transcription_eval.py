#!/usr/bin/env python3
"""EchoMind Law transcription evaluation (S2.4, LIVE-01; decision D-040).

Scores transcripts of the scripted legal conversations in scenarios.json:
- word error rate, verbatim and normalised;
- entities (names, amounts, dates, references), counting spoken forms;
- negations kept;
- speaker accuracy.

The scoring is `app.legal.meetings.accuracy`.

Where the transcripts come from (`--asr`):
- **capture** (the real measurement):
  1. Each turn is spoken by a local voice: `--voice` is a Piper model file, run with the `piper`
     program on this host. The voice model's licence must be checked before results are published.
  2. The audio is streamed through the Law capture socket, in-process, with the speaker marked per
     turn, exactly as a person would.
  3. The live speech model transcribes it.

  This needs the speech model installed (NeMo and nvidia/nemotron-speech-streaming-en-0.6b). On the
  development host it is not: running it means the Law speech stack (the first `compose.sh up`, an
  owner decision), so this mode stops and says so.
- **file:** score transcripts produced elsewhere (`--hypotheses file.json`, in the form
  {scenario_id: [{"speaker": name, "text": ...}]}).
- **script:** the scripts themselves, as a self-check of the harness: everything must score
  perfectly.

    cd backend && python3 ../eval/law/transcription/run_transcription_eval.py --asr script
    cd backend && python3 ../eval/law/transcription/run_transcription_eval.py --asr capture --voice /path/en_US-voice.onnx

Reports go to eval/law/reports/<time>-transcription-<asr>.json and .md. Nothing leaves this host.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAW = HERE.parent
ROOT = LAW.parent.parent
sys.path.insert(0, str(LAW))


def _speech_available() -> bool:
    try:
        from app.transcribe import stt_streaming
        return bool(getattr(stt_streaming, "NEMOTRON_AVAILABLE", False))
    except Exception:
        return False


def _synthesise(text: str, voice: str, out: Path) -> bytes:
    """16 kHz mono PCM16 of `text`, spoken by a local Piper voice."""
    exe = shutil.which("piper")
    if not exe:
        raise SystemExit("The `piper` program is not installed on this host.")
    subprocess.run([exe, "--model", voice, "--output_file", str(out)], input=text.encode(), check=True, capture_output=True)
    with wave.open(str(out), "rb") as w:
        rate, frames = w.getframerate(), w.readframes(w.getnframes())
    if rate != 16000:
        import numpy as np
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        idx = np.arange(0, len(pcm), rate / 16000.0)
        frames = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.int16).tobytes()
    return frames


def _capture(scenario: dict, voice: str, work: Path) -> tuple[list[dict], float]:
    """Run one scenario through the real capture socket; returns its transcript and the time it took."""
    from fastapi.testclient import TestClient
    import app.main as main_mod
    from app.legal.matters import repo
    from app.legal.orgs import default_org_id
    from conftest import csrf_headers, make_user, sign_in
    org = default_org_id()
    user = make_user(f"eval{int(time.time() * 1000) % 10**9}", mfa=True, firm_roles=["attorney"])
    client = repo.create_client(org, f"Client {scenario['id']}", scenario["id"][:8].upper(), user["id"])
    matter = repo.create_matter(org, client["id"], f"T-{scenario['id'][:10]}", "Transcription evaluation", user["id"])
    c = TestClient(main_mod.app, base_url="https://testserver")
    sign_in(c, user)
    base = f"/api/legal/matters/{matter['id']}/meetings"
    m = c.post(base, headers=csrf_headers(c), json={"title": scenario["id"], "kind": scenario["kind"]}).json()["meeting"]
    base = f"{base}/{m['id']}"
    ids = {}
    for s in scenario["speakers"]:
        p = c.post(f"{base}/participants", headers=csrf_headers(c), json=s).json()["participant"]
        c.post(f"{base}/participants/{p['id']}/consent", headers=csrf_headers(c), json={"method": "written", "statement_read": True})
        ids[s["name"]] = p["id"]
    started = time.monotonic()
    with c.websocket_connect(f"wss://testserver{base}/capture?sample_rate=16000", headers={"origin": "https://testserver"}) as ws:
        ws.receive_json()
        for k, turn in enumerate(scenario["turns"]):
            ws.send_json({"type": "speaker", "participant_id": ids[turn["speaker"]]})
            while ws.receive_json()["type"] != "speaker":
                pass
            pcm = _synthesise(turn["text"], voice, work / f"{scenario['id']}-{k}.wav")
            for i in range(0, len(pcm), 8192):                     # 256 ms frames
                ws.send_bytes(pcm[i:i + 8192])
        ws.send_json({"type": "stop"})
        while ws.receive_json()["type"] != "stopped":
            pass
    elapsed = time.monotonic() - started
    segs = c.get(f"{base}/transcript").json()["segments"]
    return [{"speaker": s["speaker"], "text": s["raw_text"]} for s in segs], elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asr", choices=["capture", "file", "script"], default="script")
    ap.add_argument("--voice", help="Piper voice model (.onnx) for --asr capture")
    ap.add_argument("--hypotheses", help="JSON transcripts for --asr file")
    args = ap.parse_args()
    from run_law_eval import setup_env
    args.llm, args.model, args.embed = "http://localhost:8355/v1", "none", "http://localhost:11435/api/embeddings"
    data = setup_env(args)
    try:
        return run(args)
    finally:
        shutil.rmtree(data, ignore_errors=True)
        (data.parent / f"{data.name}.key").unlink(missing_ok=True)


def run(args) -> int:
    import logging
    logging.disable(logging.INFO)
    from app.core.db import init_db
    init_db()
    from app.legal.meetings import accuracy as acc
    scenarios = json.loads((HERE / "scenarios.json").read_text())["scenarios"]
    if args.asr == "capture":
        if not _speech_available():
            print("Speech recognition is not installed on this host (NeMo and the Nemotron model are needed).\n"
                  "Run this mode on the Law speech stack; starting it on the shared host is an owner decision (first compose.sh up).")
            return 2
        if not args.voice:
            raise SystemExit("--voice is required for --asr capture")
    given = json.loads(Path(args.hypotheses).read_text()) if args.asr == "file" else {}
    rows, work = [], Path(tempfile.mkdtemp(prefix="emlaw-tts-"))
    try:
        for sc in scenarios:
            seconds = None
            if args.asr == "script":
                hyp = [{"speaker": t["speaker"], "text": t["text"]} for t in sc["turns"]]
            elif args.asr == "file":
                hyp = given.get(sc["id"], [])
            else:
                hyp, seconds = _capture(sc, args.voice, work)
            s = acc.score(sc["turns"], hyp, [acc.Entity(e["kind"], e["text"]) for e in sc["entities"]], sc["negations"]).as_dict()
            rows.append({"id": sc["id"], "kind": sc["kind"], "speakers": len(sc["speakers"]), "seconds": seconds, **s})
            print(f"{sc['id']}: WER {s['wer_normalised']} (verbatim {s['wer_verbatim']}), entities {s['entity_accuracy']}, "
                  f"negations {s['negation_accuracy']}, speakers {s['speaker_accuracy']}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    mean = lambda k: round(statistics.mean(r[k] for r in rows if r[k] is not None), 4) if any(r[k] is not None for r in rows) else None  # noqa: E731
    summary = {"scenarios": len(rows), "wer_normalised": mean("wer_normalised"), "wer_verbatim": mean("wer_verbatim"),
               "entity_accuracy": mean("entity_accuracy"), "negation_accuracy": mean("negation_accuracy"),
               "speaker_accuracy": mean("speaker_accuracy"),
               "missed_entities": sum(len(r["missed_entities"]) for r in rows), "lost_negations": sum(len(r["lost_negations"]) for r in rows)}
    out = LAW / "reports"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / f"{stamp}-transcription-{args.asr}.json").write_text(json.dumps({"meta": {"asr": args.asr, "commit": commit, "time": stamp,
                                                                                     "voice": Path(args.voice).name if args.voice else None},
                                                                            "summary": summary, "scenarios": rows}, indent=1))
    lines = [f"# EchoMind Law transcription evaluation — {args.asr}", "", f"- Commit: `{commit}`", f"- Run: {stamp}", "",
             "| Measure | Result |", "|---|---|"] + [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    lines += ["", "| Scenario | WER (normalised) | WER (verbatim) | Entities | Negations | Speakers | Missed | Lost negations |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['wer_normalised']} | {r['wer_verbatim']} | {r['entity_accuracy']} | {r['negation_accuracy']} | "
              f"{r['speaker_accuracy']} | {', '.join(r['missed_entities']) or '—'} | {', '.join(r['lost_negations']) or '—'} |" for r in rows]
    (out / f"{stamp}-transcription-{args.asr}.md").write_text("\n".join(lines) + "\n")
    print(f"report: eval/law/reports/{stamp}-transcription-{args.asr}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
