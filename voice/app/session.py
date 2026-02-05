import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Optional, Deque, List, Dict
from collections import deque

import numpy as np
import webrtcvad

from .config import SETTINGS
from .adapters.stt_whisper import WhisperSTT
from .adapters.llm_openai_stream import OpenAICompatLLMStream
from .adapters.tts_piper import PiperTTS
from .adapters.moshi_ws import MoshiWsAdapter

def pcm16_bytes_to_float32(pcm16: bytes) -> np.ndarray:
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    return x / 32768.0

def float32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()

def rms_energy(pcm16: bytes) -> float:
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((x / 32768.0) ** 2)))

def detect_emotion_playback_rate(text: str) -> float:
    t = (text or "").lower()
    if any(w in t for w in ["great", "awesome", "perfect", "nice", "congrats", "yay", "happy"]):
        return 1.06
    if any(w in t for w in ["sorry", "unfortunately", "sad", "issue", "problem", "can't", "cannot"]):
        return 0.96
    if any(w in t for w in ["warning", "important", "careful", "critical"]):
        return 1.02
    return 1.00

def ends_sentence(buf: str) -> bool:
    return bool(re.search(r"[\.!\?]\s*$", buf.strip()))

def approx_token_count(text: str) -> int:
    # crude but safe: ~4 chars per token in English
    return max(1, int(len(text) / 4))

@dataclass
class Frame:
    ts: float
    pcm16: bytes  # 16kHz, 20ms PCM16 mono

class UtteranceBuffer:
    def __init__(self, max_ms: int):
        self.max_frames = max(1, int(max_ms / SETTINGS.FRAME_MS))
        self.frames: Deque[Frame] = deque()

    def reset(self):
        self.frames.clear()

    def push(self, fr: Frame):
        self.frames.append(fr)
        while len(self.frames) > self.max_frames:
            self.frames.popleft()

    def to_audio_f32(self) -> np.ndarray:
        if not self.frames:
            return np.zeros(0, dtype=np.float32)
        pcm = b"".join(fr.pcm16 for fr in self.frames)
        return pcm16_bytes_to_float32(pcm)

class OmniSessionA:
    """Unmute-like session:
    - VAD endpointing -> Whisper final ASR
    - LLM streaming with conversation memory
    - Phrase commit -> Piper TTS streaming
    - Barge-in cancel + client fade smoothing
    """

    def __init__(self, ws):
        self.ws = ws

        self.sr = SETTINGS.SR
        self.frame_ms = SETTINGS.FRAME_MS
        self.frame_bytes = int(self.sr * (self.frame_ms/1000.0) * 2)

        self.vad = webrtcvad.Vad(SETTINGS.VAD_AGGR)
        self.endpoint_silence_frames = max(1, int(SETTINGS.ENDPOINT_SILENCE_MS / self.frame_ms))
        self.min_speech_frames = max(1, int(SETTINGS.MIN_SPEECH_MS / self.frame_ms))
        self.tail_frames = max(0, int(SETTINGS.END_TAIL_MS / self.frame_ms))

        self.generation_id = 0
        self.turn_id = 0

        self.in_q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.out_q: asyncio.Queue = asyncio.Queue(maxsize=1800)

        self._closed = False
        self._tasks = []
        self._finalize_task: Optional[asyncio.Task] = None
        self._reply_task: Optional[asyncio.Task] = None
        self._llm_prod_task: Optional[asyncio.Task] = None
        self._kickoff_task: Optional[asyncio.Task] = None
        self._cancel_lock = asyncio.Lock()

        self.in_speech = False
        self.silence_count = 0
        self.speech_count = 0
        self.utt = UtteranceBuffer(max_ms=15000)

        self.stt = WhisperSTT(SETTINGS.WHISPER_MODEL)
        self.llm = OpenAICompatLLMStream(
            SETTINGS.LLM_URL,
            SETTINGS.LLM_MODEL,
            temperature=SETTINGS.LLM_TEMPERATURE,
            max_tokens=SETTINGS.LLM_MAX_TOKENS,
        )
        self.tts = PiperTTS(
            SETTINGS.PIPER_MODEL,
            speaker_id=SETTINGS.PIPER_SPEAKER,
            noise_scale=SETTINGS.PIPER_NOISE_SCALE,
            length_scale=SETTINGS.PIPER_LENGTH_SCALE,
        )

        self.moshi = MoshiWsAdapter(SETTINGS.MOSHI_URL) if SETTINGS.USE_MOSHI_CORE else None

        # ---- Conversation memory ----
        self.system_prompt: str = "You are a realtime voice assistant. Be concise, helpful, and conversational."
        self.history: List[Dict] = []  # [{"role":"user"/"assistant","content":...}, ...]
        self.max_history_turns: int = 12
        self.max_history_tokens: int = 1400  # keep prompt reasonable

    async def start(self, session_id: str):
        if self.moshi:
            await self.moshi.connect()
            self._tasks.append(asyncio.create_task(self._moshi_recv_loop()))

        self._tasks += [
            asyncio.create_task(self._sender_loop()),
            asyncio.create_task(self._consume_loop()),
        ]

        await self.send({
            "type": "hello",
            "session_id": session_id,
            "note": "v5.1: adds Context box + conversation memory. Set context then speak; it remembers prior turns."
        })
        await self.send({"type": "context_ack", "system_prompt": self.system_prompt})

    async def close(self):
        self._closed = True
        if self._finalize_task and not self._finalize_task.done():
            self._finalize_task.cancel()
        for t in self._tasks:
            t.cancel()
        if self.moshi:
            await self.moshi.close()

    async def send(self, msg: dict):
        await self.out_q.put(msg)


    async def _cancel_assistant_pipeline(self, keep_listening: bool, send_cancel: bool = True):
        """Cancel assistant output (LLM/TTS/kickoff) immediately.
        If keep_listening=True, do NOT reset the current utterance buffer (so barge-in keeps capturing your speech).
        """
        async with self._cancel_lock:
            self.generation_id += 1

            for t in [self._reply_task, self._llm_prod_task, self._kickoff_task, self._finalize_task]:
                if t and (not t.done()):
                    t.cancel()

            self._reply_task = None
            self._llm_prod_task = None
            self._kickoff_task = None
            self._finalize_task = None

            self._assistant_is_speaking = False
            self._assistant_active_gen = None

            if not keep_listening:
                self.in_speech = False
                self.silence_count = 0
                self.speech_count = 0
                self._speech_run = 0
                self._nonspeech_run = 0
                self.utt.reset()

            if send_cancel:
                await self.send({"type": "cancel", "generation_id": self.generation_id})

            if self.moshi:
                asyncio.create_task(self.moshi.cancel(self.generation_id))

    def _assistant_active(self) -> bool:
        if self._assistant_is_speaking:
            return True
        if self._assistant_active_gen is not None:
            return True
        if self._reply_task is not None or self._llm_prod_task is not None or self._kickoff_task is not None or self._finalize_task is not None:
            return True
        return False

    async def on_audio_frame(self, ts: float, pcm16: bytes):
        if self._closed:
            return
        if len(pcm16) != self.frame_bytes:
            return
        try:
            self.in_q.put_nowait(Frame(ts=ts, pcm16=pcm16))
        except asyncio.QueueFull:
            pass

    async def on_control(self, data: dict):
        """Handle control messages from the browser."""
        t = data.get("type")
        if t == "set_context":
            self.system_prompt = (data.get("system_prompt") or "").strip() or self.system_prompt
            # If user checks "clear memory"
            if bool(data.get("clear_memory")):
                self.history = []
            await self.send({"type": "context_ack", "system_prompt": self.system_prompt, "cleared": bool(data.get('clear_memory'))})

        if t == "clear_memory":
            self.history = []
            await self.send({"type": "context_ack", "system_prompt": self.system_prompt, "cleared": True})

    def _barge_in(self):
        self.generation_id += 1

    def _trim_history(self):
        # keep last N turns and under approx token budget
        # turns = pairs user+assistant, so 2 messages per turn
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-self.max_history_turns*2:]
        # token budget
        total = approx_token_count(self.system_prompt)
        for m in self.history[::-1]:
            total += approx_token_count(m.get("content",""))
            if total > self.max_history_tokens:
                # drop older messages until under budget
                # remove from front
                while self.history and total > self.max_history_tokens:
                    dropped = self.history.pop(0)
                    total -= approx_token_count(dropped.get("content",""))

    def _build_messages(self, user_text: str) -> List[Dict]:
        self._trim_history()
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(self.history)
        msgs.append({"role": "user", "content": user_text})
        return msgs

    async def _sender_loop(self):
        while not self._closed:
            msg = await self.out_q.get()
            if msg.get("type") == "audio_out" and isinstance(msg.get("pcm16_raw"), (bytes, bytearray)):
                msg["pcm16_b64"] = base64.b64encode(msg.pop("pcm16_raw")).decode("utf-8")
            await self.ws.send_text(json.dumps(msg))

    async def _consume_loop(self):
        while not self._closed:
            fr = await self.in_q.get()

            if self.moshi:
                asyncio.create_task(self.moshi.send_audio(fr.pcm16, self.sr))

            e = rms_energy(fr.pcm16)
            is_speech = False if e < 0.004 else self.vad.is_speech(fr.pcm16, self.sr)

            if is_speech:
                self.silence_count = 0
                self.speech_count += 1

                if not self.in_speech:
                    self.in_speech = True
                    self.utt.reset()
                    self._barge_in()
                    await self.send({"type": "cancel", "generation_id": self.generation_id})
                    await self.send({"type": "event", "event": "USER_SPEECH_START", "generation_id": self.generation_id})
                    if self.moshi:
                        asyncio.create_task(self.moshi.cancel(self.generation_id))

                self.utt.push(fr)

            else:
                if self.in_speech:
                    self.silence_count += 1
                    if self.tail_frames > 0:
                        self.utt.push(fr)

                    if self.silence_count >= self.endpoint_silence_frames:
                        self.in_speech = False
                        await self.send({"type": "event", "event": "USER_SPEECH_END", "generation_id": self.generation_id})

                        if self.speech_count < self.min_speech_frames:
                            self.speech_count = 0
                            self.silence_count = 0
                            continue

                        if self._finalize_task and not self._finalize_task.done():
                            self._finalize_task.cancel()
                        self._finalize_task = asyncio.create_task(self._finalize_and_reply(self.generation_id))

                        self.speech_count = 0
                        self.silence_count = 0

    async def _finalize_and_reply(self, my_gen: int):
        await self.send({"type": "event", "event": "THINKING", "generation_id": my_gen})

        audio = self.utt.to_audio_f32()
        if audio.size < int(0.25 * self.sr):
            return

        try:
            user_text = self.stt.transcribe(audio)
        except Exception as e:
            await self.send({"type": "error", "where": "stt", "message": str(e), "generation_id": my_gen})
            return

        user_text = (user_text or "").strip()
        if not user_text:
            return

        self.turn_id += 1
        await self.send({"type": "asr_final", "turn_id": self.turn_id, "generation_id": my_gen, "text": user_text})
        await self.send({"type": "event", "event": "SPEAKING", "generation_id": my_gen})

        # Build messages with memory
        messages = self._build_messages(user_text)

        phrase_buf = ""
        assistant_text = ""
        last_emit = time.time()

        def commit_needed(buf: str) -> bool:
            s = buf.strip()
            if len(s) >= SETTINGS.PHRASE_MAX_CHARS:
                return True
            if len(s) >= SETTINGS.PHRASE_MIN_CHARS and ends_sentence(buf):
                return True
            if len(s) >= SETTINGS.PHRASE_MIN_CHARS and (time.time() - last_emit) * 1000 >= SETTINGS.PHRASE_COMMIT_PAUSE_MS:
                return True
            return False

        try:
            tok_q: asyncio.Queue = asyncio.Queue(maxsize=4000)

            async def producer():
                loop = asyncio.get_running_loop()
                def run_iter():
                    for t in self.llm.stream_messages(messages):
                        tok_q.put_nowait(t)
                    tok_q.put_nowait(None)
                await loop.run_in_executor(None, run_iter)

            self._llm_prod_task = asyncio.create_task(producer())
            prod_task = self._llm_prod_task

            while True:
                if my_gen != self.generation_id:
                    prod_task.cancel()
                    return

                tok = await tok_q.get()
                if tok is None:
                    break

                last_emit = time.time()
                assistant_text += tok
                phrase_buf += tok
                await self.send({"type": "assistant_text_partial", "generation_id": my_gen, "text": assistant_text})

                if commit_needed(phrase_buf):
                    await self._commit_phrase(my_gen, phrase_buf)
                    phrase_buf = ""

        except Exception as e:
            await self.send({"type": "error", "where": "llm_stream", "message": str(e), "generation_id": my_gen})
            try:
                reply = self.llm.complete_messages(messages)
            except Exception as e2:
                await self.send({"type": "error", "where": "llm", "message": str(e2), "generation_id": my_gen})
                return
            await self.send({"type": "assistant_text", "generation_id": my_gen, "text": reply})
            await self._speak_phrase(my_gen, reply)
            # save memory
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply})
            self._trim_history()
            await self.send({"type": "event", "event": "BACK_TO_LISTENING", "generation_id": my_gen})
            return

        if my_gen != self.generation_id:
            return
        if phrase_buf.strip():
            await self._commit_phrase(my_gen, phrase_buf)

        final = assistant_text.strip()
        await self.send({"type": "assistant_text", "generation_id": my_gen, "text": final})
        # save memory
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": final})
        self._trim_history()

        await self.send({"type": "event", "event": "BACK_TO_LISTENING", "generation_id": my_gen})

    async def _commit_phrase(self, my_gen: int, phrase: str):
        phrase = (phrase or "").strip()
        if not phrase:
            return
        await self.send({"type": "assistant_phrase", "generation_id": my_gen, "text": phrase})
        if self.moshi and SETTINGS.MOSHI_SUPPORTS_TEXT_INJECT:
            await self.moshi.text_inject(phrase, my_gen)
            return
        await self._speak_phrase(my_gen, phrase)

    async def _speak_phrase(self, my_gen: int, phrase: str, is_filler: bool = False):
        if my_gen != self.generation_id:
            return

        rate = 1.0
        if SETTINGS.EMOTION_MODE and not is_filler:
            rate = detect_emotion_playback_rate(phrase)

        try:
            y = self.tts.synth(phrase)
            sr = self.tts.sr
        except Exception as e:
            await self.send({"type": "error", "where": "tts", "message": str(e), "generation_id": my_gen})
            return

        chunk = int(sr * 0.22)
        i = 0
        while i < y.size:
            if my_gen != self.generation_id:
                return
            part = y[i:i+chunk]
            i += chunk
            await self.send({
                "type": "audio_out",
                "generation_id": my_gen,
                "sample_rate": sr,
                "playback_rate": rate,
                "pcm16_raw": float32_to_pcm16_bytes(part.astype(np.float32))
            })
            await asyncio.sleep(0.0)

    async def _moshi_recv_loop(self):
        while not self._closed and self.moshi:
            msg = await self.moshi.recv()
            if msg.get("type") == "audio_out" and msg.get("pcm16_b64"):
                gen = msg.get("generation_id", self.generation_id)
                if gen != self.generation_id:
                    continue
                pcm = base64.b64decode(msg["pcm16_b64"])
                await self.send({
                    "type": "audio_out",
                    "generation_id": gen,
                    "sample_rate": msg.get("sample_rate", 24000),
                    "pcm16_raw": pcm
                })