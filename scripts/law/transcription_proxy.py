#!/usr/bin/env python3
"""A proxy measure of live transcription on the pilot (D-040, D-072): synthetic voices, not real people.

    python3 scripts/law/transcription_proxy.py          # then scores with eval/law/transcription/run_transcription_eval.py

Each turn of the FICTIONAL scripted conversations in eval/law/transcription/scenarios.json is spoken by a local Piper
voice (fixed, without random variation, so a run can be repeated) and streamed in real time through the pilot's capture
socket on M-101, with the speaker marked per turn as a person would. The pilot's speech model transcribes it; the
transcript is scored for word error rate, entities, negations and speakers. Synthetic speech is cleaner than a real
meeting, so this is an upper bound: real voices, accents, cross-talk and room sound need the firm's own recordings.
Touches only the Law pilot (its demo matter M-101), never production.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "law"))
from demo_check import BASE, Pilot, _synthesise, q  # noqa: E402

VOICES = ["en_US-lessac-medium", "en_US-joe-medium", "en_US-amy-medium", "en_US-john-medium", "en_US-kusal-medium"]


def transcribe(p: Pilot, sc: dict, audio: list[tuple[str, bytes, int]]) -> list[dict]:
    import websockets
    mid = p.matter("alice", "M-101")
    m = p.call("alice", "POST", f"/legal/matters/{q(mid)}/meetings", {"title": f"Transcription check: {sc['id']}", "kind": sc["kind"]})
    base = f"/legal/matters/{q(mid)}/meetings/{q(m['meeting']['id'])}"
    ids = {}
    for s in sc["speakers"]:
        r = p.call("alice", "POST", base + "/participants", {"name": s["name"], "role": s["role"]})
        ids[s["name"]] = r["participant"]["id"]
        p.call("alice", "POST", f"{base}/participants/{q(ids[s['name']])}/consent", {"method": "verbal_on_record", "statement_read": True})

    async def run():
        url = f"ws://localhost:3110/api{base}/capture?sample_rate=16000"
        async with websockets.connect(url, extra_headers=[("Cookie", p.cookie_header("alice")), ("Origin", BASE)], max_size=2 ** 22,
                                      open_timeout=300) as ws:
            assert json.loads(await asyncio.wait_for(ws.recv(), 300))["type"] == "ready"

            async def drain():
                async for msg in ws:
                    if isinstance(msg, str) and json.loads(msg).get("type") == "stopped":
                        return
            task = asyncio.create_task(drain())
            frame, silence = 3200, b"\x00\x00" * 12800                       # 100 ms frames; 0.8 s between turns
            for (speaker, pcm, _) in audio:
                await ws.send(json.dumps({"type": "speaker", "participant_id": ids[speaker]}))
                for i in range(0, len(pcm + silence), frame):
                    await ws.send((pcm + silence)[i:i + frame])
                    await asyncio.sleep(0.1)
            await asyncio.sleep(2)
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.wait_for(task, 400)
    asyncio.run(run())
    p.call("alice", "POST", base + "/end")
    segs = p.call("alice", "GET", base + "/transcript")["segments"]
    return [{"speaker": s.get("speaker"), "text": s.get("raw_text") or s.get("text") or ""} for s in segs]


def main() -> int:
    scenarios = json.loads((ROOT / "eval/law/transcription/scenarios.json").read_text())["scenarios"]
    p = Pilot()
    hyps, t0 = {}, time.time()
    for sc in scenarios:
        voice = {s["name"]: VOICES[i % len(VOICES)] for i, s in enumerate(sc["speakers"])}
        audio = _synthesise([(t["speaker"], voice[t["speaker"]], t["text"]) for t in sc["turns"]])
        hyps[sc["id"]] = transcribe(p, sc, audio)
        print(f"{sc['id']}: {len(hyps[sc['id']])} segments", flush=True)
    out = ROOT / "eval/law/reports" / time.strftime("%Y%m%d-%H%M%S-transcription-proxy-hypotheses.json")
    out.write_text(json.dumps({"note": "Piper voices (fixed), streamed through the pilot's capture socket; a proxy, not real voices.",
                               "voices": VOICES, "seconds": round(time.time() - t0), "hypotheses": hyps}, indent=1))
    file = out.with_suffix(".plain.json")
    file.write_text(json.dumps(hyps))
    return subprocess.run([sys.executable, str(ROOT / "eval/law/transcription/run_transcription_eval.py"), "--asr", "file",
                           "--hypotheses", str(file)], cwd=ROOT / "backend").returncode


if __name__ == "__main__":
    sys.exit(main())
