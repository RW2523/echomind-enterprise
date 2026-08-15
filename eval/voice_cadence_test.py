"""Voice cadence measurement: quantifies the "chunk...speak" feel of the voice loop.

Feeds synthesized speech into the live voice WebSocket (like voice_e2e_test.py) and
measures, per reply:
  - n_phrases            : assistant_phrase events (each = one independent Piper synth)
  - pct_non_sentence     : fraction of phrases NOT ending in sentence punctuation
                           (each one is a prosody break mid-sentence — the choppiness proxy)
  - ttfa_ms              : end of user speech -> first audio_out
  - audio_s              : total synthesized audio duration (from pcm sample counts)
  - starve_ms            : worst client-underrun deficit — max over time of
                           (wall-clock since first audio) - (cumulative audio received);
                           > 0 means the client ran out of audio and playback gapped
  - rates                : distinct playback_rate values seen (tempo jumps within a reply)

Usage: python3 eval/voice_cadence_test.py [ws_url] [out_json]
Defaults: ws://127.0.0.1:8002/ws  /tmp/voice_cadence.json
"""
import asyncio, base64, io, json, sys, time
import numpy as np, soundfile as sf, urllib.request, websockets

WS_URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8002/ws"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/voice_cadence.json"
SPEAK_URL = WS_URL.replace("ws://", "http://").replace("/ws", "/speak")

TURNS = [
    ("casual", "Hello there! How are you doing today?"),
    ("long", "Please tell me in detail what you can do for me and how you can help with my work."),
    ("knowledge", "What do the documents say about travel expense reimbursement policy?"),
]


def synth(text: str) -> np.ndarray:
    body = json.dumps({"text": text}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        SPEAK_URL, body, {"Content-Type": "application/json"}), timeout=60)
    wav, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    n = int(len(wav) * 16000 / sr)
    return np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)


def frames(audio: np.ndarray):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    for i in range(0, len(pcm) - 639, 640):
        yield pcm[i:i + 640]


async def drain_until_quiet(ws, quiet_s=2.0, max_s=30.0):
    last_audio = time.time(); t0 = time.time()
    while time.time() - t0 < max_s and time.time() - last_audio < quiet_s:
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
            if msg.get("type") == "audio_out":
                last_audio = time.time()
        except asyncio.TimeoutError:
            pass


def ends_sentence(s: str) -> bool:
    s = (s or "").rstrip()
    return bool(s) and s[-1] in ".!?" or (len(s) > 1 and s[-1] in "\"'" and s[-2] in ".!?")


async def measure_turn(ws, label, text):
    audio = synth(text)
    for f in frames(audio):
        await ws.send(json.dumps({"type": "audio_frame", "ts": time.time(),
                                  "pcm16_b64": base64.b64encode(f).decode()}))
        await asyncio.sleep(0.018)
    silence = b"\x00" * 640
    for _ in range(70):
        await ws.send(json.dumps({"type": "audio_frame", "ts": time.time(),
                                  "pcm16_b64": base64.b64encode(silence).decode()}))
        await asyncio.sleep(0.018)
    t_end_speech = time.time()

    phrases = []            # (t, text, loop)
    t_first_audio = None
    cum_audio_s = 0.0
    starve_ms = 0.0
    rates = set()
    reply_text = None
    t0 = time.time(); last_evt = time.time()
    while time.time() - t0 < 150:
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
        except asyncio.TimeoutError:
            if phrases and time.time() - last_evt > 6:
                break
            continue
        mt = msg.get("type"); last_evt = time.time()
        now = time.time()
        if mt == "assistant_phrase":
            phrases.append((now, msg.get("text") or "", msg.get("loop") or "?"))
        elif mt == "assistant_text":
            reply_text = msg.get("text") or ""
        elif mt == "audio_out":
            if t_first_audio is None:
                t_first_audio = now
            sr = msg.get("sample_rate") or 24000
            n_samp = len(base64.b64decode(msg["pcm16_b64"])) // 2 if msg.get("pcm16_b64") else \
                     len(msg.get("pcm16_raw") or b"") // 2
            # deficit BEFORE crediting this chunk = playback starvation at its arrival
            deficit = (now - t_first_audio) - cum_audio_s
            if deficit > starve_ms / 1000.0:
                starve_ms = deficit * 1000.0
            cum_audio_s += n_samp / sr
            rates.add(round(float(msg.get("playback_rate") or 1.0), 3))
        elif mt == "event" and msg.get("name") == "BACK_TO_LISTENING":
            if phrases:
                break
    lg = [(t, x) for t, x, lp in phrases if lp != "L_I"]
    non_sent = [x for _, x in lg if not ends_sentence(x)]
    res = {
        "label": label,
        "n_phrases": len(lg),
        "pct_non_sentence": round(100.0 * len(non_sent) / max(1, len(lg)), 1),
        "non_sentence_examples": [x[-60:] for x in non_sent[:5]],
        "ttfa_ms": round((t_first_audio - t_end_speech) * 1000.0, 0) if t_first_audio else None,
        "audio_s": round(cum_audio_s, 2),
        "starve_ms": round(starve_ms, 0),
        "rates": sorted(rates),
        "reply_chars": len(reply_text or ""),
        "phrase_lens": [len(x) for _, x in lg],
    }
    print(json.dumps(res, indent=1))
    return res


async def main():
    results = []
    async with websockets.connect(WS_URL, max_size=2 ** 23) as ws:
        json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        await drain_until_quiet(ws)
        for label, text in TURNS:
            print(f"\n=== {label}: {text!r}")
            results.append(await measure_turn(ws, label, text))
            await drain_until_quiet(ws, quiet_s=2.5)
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")

asyncio.run(main())
