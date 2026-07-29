"""End-to-end voice-bot test: synthesize speech with Piper, feed it into the live
voice WebSocket as real mic frames, measure the full loop with the new 30B LLM."""
import asyncio, base64, io, json, time
import numpy as np, soundfile as sf, urllib.request, websockets

def synth(text: str) -> np.ndarray:
    """Piper TTS via /speak → 16 kHz mono float32."""
    body = json.dumps({"text": text}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/speak", body, {"Content-Type": "application/json"}), timeout=60)
    wav, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(axis=1)
    # linear resample to 16k
    n = int(len(wav) * 16000 / sr)
    return np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav).astype(np.float32)

def frames(audio: np.ndarray):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    step = 640  # 320 samples * 2 bytes = 20 ms
    for i in range(0, len(pcm) - step + 1, step):
        yield pcm[i:i + step]

async def drain_until_quiet(ws, label, quiet_s=2.0, max_s=25.0):
    """Consume messages (intro/greeting audio) until no audio for quiet_s."""
    last_audio = time.time(); t0 = time.time()
    while time.time() - t0 < max_s and time.time() - last_audio < quiet_s:
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
            if msg.get("type") == "audio_out": last_audio = time.time()
        except asyncio.TimeoutError:
            pass
    print(f"[{label}] intro/turn drained")

async def speak_turn(ws, text, label):
    audio = synth(text)
    print(f"\n=== TURN [{label}]: feeding {len(audio)/16000:.1f}s of synthesized speech: {text!r}")
    t_send0 = time.time()
    for f in frames(audio):
        await ws.send(json.dumps({"type": "audio_frame", "ts": time.time(),
                                  "pcm16_b64": base64.b64encode(f).decode()}))
        await asyncio.sleep(0.018)  # ~real-time pacing
    # trailing silence to trigger the endpoint (~1.4 s)
    silence = b"\x00" * 640
    for _ in range(70):
        await ws.send(json.dumps({"type": "audio_frame", "ts": time.time(),
                                  "pcm16_b64": base64.b64encode(silence).decode()}))
        await asyncio.sleep(0.018)
    t_end_speech = time.time()

    partials, final_text, reply_parts = 0, None, []
    t_final = t_first_reply_text = t_first_audio = None
    audio_chunks = 0
    t0 = time.time(); last_evt = time.time()
    while time.time() - t0 < 120:
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
        except asyncio.TimeoutError:
            if reply_parts and time.time() - last_evt > 6: break
            continue
        mt = msg.get("type"); last_evt = time.time()
        if mt == "partial_transcript": partials += 1
        elif mt == "asr_final":
            final_text = msg.get("text"); t_final = time.time()
        elif mt in ("assistant_text_partial", "assistant_text"):
            if t_first_reply_text is None: t_first_reply_text = time.time()
            if mt == "assistant_text": reply_parts = [msg.get("text") or ""]
        elif mt == "assistant_phrase":
            if t_first_reply_text is None: t_first_reply_text = time.time()
            reply_parts.append(msg.get("text") or "")
        elif mt == "audio_out":
            audio_chunks += 1
            if t_first_audio is None: t_first_audio = time.time()
        elif mt == "event" and msg.get("name") in ("BACK_TO_LISTENING",):
            if reply_parts: break
    reply = " ".join(p for p in reply_parts if p).strip()
    def d(t): return f"{t - t_end_speech:+.2f}s" if t else "—"
    print(f"  heard (final STT): {final_text!r}")
    print(f"  partials={partials}  audio_out_chunks={audio_chunks}")
    print(f"  timing after end-of-speech: asr_final={d(t_final)}  first_reply_text={d(t_first_reply_text)}  FIRST AUDIO={d(t_first_audio)}")
    print(f"  reply: {reply[:300]}")

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws", max_size=2**23) as ws:
        hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        print("connected:", hello.get("type"))
        await drain_until_quiet(ws, "intro")
        await speak_turn(ws, "Hello there! How are you doing today?", "casual")
        await drain_until_quiet(ws, "turn1-tail", quiet_s=2.5)
        await speak_turn(ws, "What do the documents say about travel expense reimbursement?", "knowledge/RAG")

asyncio.run(main())
