import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Deque, List, Dict, Any
from collections import deque

import numpy as np
import requests
import webrtcvad

from .config import SETTINGS
from .conversation_memory import ConversationMemory
from .echo_commands import parse_and_route, strip_wake_word
from .wake_word_storage import load_wake_word, save_wake_word
from .adapters.stt_nemotron import NemotronUtteranceSTT
from .adapters.llm_openai_stream import OpenAICompatLLMStream
from .adapters.tts_piper import PiperTTS
from .adapters.moshi_ws import MoshiWsAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persona-specific intro phrases (keyed by PersonaType value from frontend)
# ---------------------------------------------------------------------------
_PERSONA_INTRO_PHRASES: dict = {
    "Teacher / Professor": (
        "Hello! I'm your professor and guide. Ask me anything you'd like to learn or understand — "
        "from your uploaded documents, transcripts, or any topic you're curious about. I'm here to teach!"
    ),
    "Financial Advisor": (
        "Hello! I'm your DoD financial management advisor. I can help with FMR regulations, compliance questions, "
        "and your uploaded documents and transcripts. What would you like to explore?"
    ),
    "Funny & Calming Assistant": (
        "Hey there! Great to have you here. Whatever's on your mind, I'm here to help — with a smile and zero stress. "
        "What can I do for you today?"
    ),
    "Lawyer": (
        "Good day. I'm your legal advisor. I can analyze contracts, regulations, and documents for legal obligations, "
        "risks, and compliance matters. How may I assist you?"
    ),
    "AI Expert & Manager": (
        "Hello! I'm your AI expert and technical manager. Ask me about AI architectures, software design, "
        "engineering decisions, or insights from your documents and meetings. What are we building today?"
    ),
}

_DEFAULT_INTRO_PHRASE = "Hi! I'm EchoMind, your AI assistant. How can I help you today?"


def _get_persona_intro(persona: str) -> str:
    """Return the persona-specific intro phrase, or the default if persona is unknown."""
    if persona and persona in _PERSONA_INTRO_PHRASES:
        return _PERSONA_INTRO_PHRASES[persona]
    # Case-insensitive partial match
    if persona:
        lower = persona.lower()
        for key, phrase in _PERSONA_INTRO_PHRASES.items():
            if key.lower() in lower or lower in key.lower():
                return phrase
    return (getattr(SETTINGS, "INTRO_PHRASE", "") or "").strip() or _DEFAULT_INTRO_PHRASE


def _log_llm_response(user_text: str, reply: str):
    """Log user input and LLM response for debugging."""
    logger.info(
        "[LLM_RESPONSE] user_text=%r reply=%r",
        (user_text or "")[:200],
        (reply or "")[:400],
    )


def pcm16_bytes_to_float32(pcm16: bytes) -> np.ndarray:
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    return x / 32768.0

def float32_to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def _fade_chunk_edges(a: np.ndarray, sr: int, fade_ms: float = 4.0) -> np.ndarray:
    """Apply short fade-in and fade-out to avoid clicks at chunk boundaries. Modifies in place, returns a."""
    if a.size == 0:
        return a
    n = int(sr * (fade_ms / 1000.0))
    n = min(n, a.size // 2)
    if n <= 0:
        return a
    # fade-in: linear 0 -> 1 over first n samples
    for i in range(n):
        a[i] *= (i + 1) / (n + 1)
    # fade-out: linear 1 -> 0 over last n samples
    for i in range(n):
        a[-(i + 1)] *= (i + 1) / (n + 1)
    return a

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
    """True when buffer ends on sentence-ending punctuation (optional closing quote)."""
    s = (buf or "").rstrip()
    if not s:
        return False
    return bool(re.search(r'[.!?]["\']?\s*$', s))


def ends_natural_clause(buf: str) -> bool:
    """True when buffer ends on a clause boundary (comma/semicolon/colon). Skips trailing-number patterns like 1,234."""
    s = (buf or "").rstrip()
    if not s:
        return False
    if re.search(r",\d\s*$", s):
        return False
    return bool(re.search(r'[,;:]["\']?\s*$', s))


def phrase_commit_needed(phrase_buf: str, phrases_enqueued: int, last_emit: float) -> bool:
    """
    Whether to enqueue ``phrase_buf`` for TTS. Used for both local LLM SSE and backend RAG NDJSON streams.
    """
    s = phrase_buf.strip()
    if not s:
        return False
    if len(s) >= SETTINGS.PHRASE_MAX_CHARS:
        return True
    clause_min = max(8, getattr(SETTINGS, "PHRASE_CLAUSE_MIN_CHARS", 18))
    if len(s) >= clause_min and ends_natural_clause(phrase_buf):
        return True
    min_sentence = (
        max(4, getattr(SETTINGS, "FIRST_SENTENCE_MIN_CHARS", 8))
        if phrases_enqueued == 0
        else SETTINGS.PHRASE_MIN_CHARS
    )
    if len(s) >= min_sentence and ends_sentence(phrase_buf):
        return True
    if (
        len(s) >= SETTINGS.PHRASE_MIN_CHARS
        and (time.time() - last_emit) * 1000 >= SETTINGS.PHRASE_COMMIT_PAUSE_MS
    ):
        return True
    return False

def approx_token_count(text: str) -> int:
    # crude but safe: ~4 chars per token in English
    return max(1, int(len(text) / 4))


def strip_markdown_for_speech(text: str) -> str:
    """Remove markdown (###, **, *, `, etc.) so TTS and LLM see plain English only."""
    if not (text or "").strip():
        return (text or "").strip()
    s = (text or "").strip()
    # Links: [link text](url) -> link text
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)
    # Bold/italic/underline markers (remove the delimiters, keep the text)
    s = re.sub(r"\*\*", "", s)
    s = re.sub(r"__", "", s)
    s = re.sub(r"\*", "", s)
    s = re.sub(r"_", " ", s)  # single _ often used as italic, replace with space to avoid glue
    # Inline code backticks
    s = re.sub(r"`", "", s)
    # Headers: leading # or ## or ### etc. at start of line
    s = re.sub(r"^#+\s*", "", s, flags=re.MULTILINE)
    # Collapse whitespace and newlines to single space
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# Minimum cleaned length to treat as real speech (avoid noise).
_MIN_ENGLISH_INPUT_LEN = 2
# Minimum word count to avoid single-word noise triggering full flow.
_MIN_ENGLISH_WORDS = 1

# Phrases that indicate the user is asking about knowledge-base content. Only then use RAG backend (same as backend/app/rag/advanced.py).
_RAG_INDICATOR_PHRASES = (
    "document", "documents", "resource", "resources",
    "live transcript", "live transcription", "transcript", "transcripts",
    "discussion", "discussions", "book", "books", "pdf", "pdfs",
    "file", "files", "uploaded", "saved transcript",
    # DoD / policy-style questions without saying "document"
    "dod fmr", "fmr", "certifying officer", "disbursing officer",
    "unliquidated", "appropriation", "disbursement",
    "regulation", "compliance", "audit",
)
# Section/paragraph references (word boundary so "intersection" does not match "section")
_RAG_REF_PATTERNS = (
    re.compile(r"\bsection\s+[0-9]", re.I),
    re.compile(r"\bparagraph\s+[0-9]", re.I),
    re.compile(r"\bvol(?:ume)?\.?\s*[0-9]", re.I),
    re.compile(r"\bchapter\s+[0-9]", re.I),
)


def preprocess_english_only(raw: str) -> Optional[str]:
    """
    Preprocess STT output for English-only flow: retain only English letters, apostrophe, and basic punctuation.
    Returns None if the result is empty, too short, or looks like noise (so we do not send to LLM).
    """
    if not (raw or "").strip():
        return None
    s = (raw or "").strip()
    # Keep only a-z, A-Z, apostrophe (for "don't"), space, and basic punctuation
    cleaned = re.sub(r"[^a-zA-Z\s'.,?!\-]", " ", s)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or len(cleaned) < _MIN_ENGLISH_INPUT_LEN:
        return None
    words = [w.strip("'.,?!") for w in cleaned.split() if w.strip("'.,?!")]
    if len(words) < _MIN_ENGLISH_WORDS:
        return None
    # Reject obvious noise: same character repeated (e.g. "aaaa", "ssss")
    if len(cleaned) <= 6 and len(set(cleaned.replace(" ", "").replace("'", "").replace(".", "").replace(",", ""))) <= 1:
        return None
    # Reject consonant-only short strings (common STT noise)
    letters = re.sub(r"[^a-zA-Z]", "", cleaned)
    if len(letters) <= 4 and letters and not re.search(r"[aeiouAEIOU]", letters):
        return None
    return cleaned


def _user_asks_about_knowledge(user_text: str) -> bool:
    """
    True when the user's message indicates document/transcript/resources (RAG context).
    Uses substring check: any phrase in _RAG_INDICATOR_PHRASES contained in the message → hit backend RAG flow.
    Backend will classify intent (transcript vs document) from context.
    """
    t = (user_text or "").strip().lower()
    if not t:
        logger.info("RAG check: input=%r intent=none (empty)", user_text)
        return False
    matched = next((phrase for phrase in _RAG_INDICATOR_PHRASES if phrase in t), None)
    if matched:
        logger.info("RAG check: input=%r intent=%s", user_text[:200], matched)
        return True
    for rx in _RAG_REF_PATTERNS:
        if rx.search(t):
            logger.info("RAG check: input=%r intent=ref_pattern=%s", user_text[:200], rx.pattern)
            return True
    logger.info("RAG check: input=%r intent=none", user_text[:200])
    return False


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
    """Convo-like session:
    - VAD endpointing -> Nemotron utterance-final ASR (shared nemotron_asr package)
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
        self._tts_phrase_task: Optional[asyncio.Task] = None
        self._cancel_lock = asyncio.Lock()

        self.in_speech = False
        self.silence_count = 0
        self.speech_count = 0
        self.utt = UtteranceBuffer(max_ms=15000)
        self._speech_lead_count = 0  # consecutive speech frames before we treat as user speech (barge-in robustness)
        # Frames classified as speech before in_speech flips True (must be replayed into utt or the first word is clipped)
        self._pending_lead_frames: List[Frame] = []
        self._assistant_is_speaking = False
        self._assistant_active_gen: Optional[int] = None
        self._is_playing_intro = False  # Disable barge-in during intro to avoid mic feedback cutting it off
        self._pending_intro = True      # Wait for first set_context (persona) before playing intro

        self.stt = NemotronUtteranceSTT()
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
            use_cuda=SETTINGS.PIPER_USE_CUDA,
        )

        self.moshi = MoshiWsAdapter(SETTINGS.MOSHI_URL) if SETTINGS.USE_MOSHI_CORE else None

        # ---- Conversation memory (LLM turn history) ----
        self.system_prompt: str = (
            "You are EchoMind, a helpful, knowledgeable voice assistant. "
            "Be concise, natural, and conversational—this is a voice interface, so keep responses clear and well-paced. "
            "Cite sections or sources when answering from documents. "
            "For procedural questions, give numbered steps. For comparisons, use clear bullet points. "
            "Never invent facts, section numbers, or page references.\n\n"
            "GUARDRAIL: Be genuinely helpful. Refuse harmful, offensive, or clearly illegal requests politely: "
            "'I can't help with that, but I'm happy to assist with something constructive. What else can I do for you?'"
        )
        self.history: List[Dict] = []  # [{"role":"user"/"assistant","content":...}, ...]
        self.max_history_turns: int = 12
        self.max_history_tokens: int = 1400  # keep prompt reasonable
        self.use_knowledge_base: bool = False
        self.persona: str = ""
        self.context_window: str = "all"
        self.voice_bot_name: str = ""
        self.voice_user_name: str = ""
        # Listen-only mode: accumulate user speech until trigger or wake word (single string, not split by utterance)
        self.listen_only: bool = False
        self.listen_buffer: str = ""
        self.trigger_phrases: List[str] = []  # Only wake word triggers; no phrases (avoids false exits on pauses)
        # EchoMind: rolling conversation memory (rolling window)
        self.conversation_memory = ConversationMemory(
            window_minutes=getattr(SETTINGS, "MEMORY_WINDOW_MINUTES", 30.0),
        )
        if getattr(SETTINGS, "ECHO_DEBUG", False):
            self.conversation_memory.set_debug_log(lambda msg: logger.info(msg))
        # Global profile (session-level; voice commands and set_context can update)
        default_wake = load_wake_word(getattr(SETTINGS, "DEFAULT_ASSISTANT_NAME", "EchoMind"))
        self.global_profile: Dict[str, str] = {
            "assistant_name": default_wake,
            "wake_word": default_wake,
            "user_name": getattr(SETTINGS, "DEFAULT_USER_NAME", "") or "",
            "timezone": getattr(SETTINGS, "DEFAULT_TIMEZONE", "America/New_York"),
            "location": getattr(SETTINGS, "DEFAULT_LOCATION", "") or "",
        }
        self.pending_wake_word_change: Optional[str] = None
        self._last_user_utterance: Optional[str] = None
        self._logged_first_audio_frame = False

    async def start(self, session_id: str):
        self.listen_buffer = ""
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
            "note": "EchoMind voice assistant ready. Intro will play after persona context is set."
        })
        await self.send({"type": "context_ack", "system_prompt": self.system_prompt})
        await self._emit_profile_update()
        # Intro TTS: deferred until first set_context so we know the persona.
        # _pending_intro is True; on_control will fire the persona-specific greeting.

    async def _play_intro(self, phrase: str):
        """Play intro TTS once after start. Barge-in disabled during intro to avoid mic feedback cutting it off."""
        phrase = strip_markdown_for_speech(phrase or "")
        if not phrase:
            self._is_playing_intro = False
            return
        my_gen = self.generation_id
        self._assistant_is_speaking = True
        try:
            await self.send({"type": "event", "event": "SPEAKING", "generation_id": my_gen})
            # Full text in UI immediately (intro did not send text before, so transcript looked empty or "cut off")
            await self.send({"type": "assistant_text", "generation_id": my_gen, "text": phrase})
            try:
                loop = asyncio.get_running_loop()
                y = await loop.run_in_executor(None, self.tts.synth, phrase)
                sr = self.tts.sr
            except Exception as e:
                await self.send({"type": "error", "where": "tts_intro", "message": str(e), "generation_id": my_gen})
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
                    "playback_rate": 1.0,
                    "pcm16_raw": float32_to_pcm16_bytes(part.astype(np.float32))
                })
                await asyncio.sleep(0.0)
        finally:
            self._is_playing_intro = False
            if self._assistant_is_speaking and self.generation_id == my_gen:
                self._assistant_is_speaking = False
            await self.send({"type": "event", "event": "BACK_TO_LISTENING", "generation_id": self.generation_id})

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

            for t in [self._reply_task, self._llm_prod_task, self._kickoff_task, self._finalize_task, self._tts_phrase_task]:
                if t and (not t.done()):
                    t.cancel()

            self._reply_task = None
            self._llm_prod_task = None
            self._kickoff_task = None
            self._finalize_task = None
            self._tts_phrase_task = None

            self._assistant_is_speaking = False
            self._assistant_active_gen = None

            if not keep_listening:
                self.in_speech = False
                self.silence_count = 0
                self.speech_count = 0
                self._speech_run = 0
                self._nonspeech_run = 0
                self._pending_lead_frames.clear()
                self.utt.reset()

            if send_cancel:
                logger.info(
                    "Voice: cancel assistant pipeline gen=%s keep_listening=%s",
                    self.generation_id,
                    keep_listening,
                )
                await self.send({"type": "cancel", "generation_id": self.generation_id})

            if self.moshi:
                asyncio.create_task(self.moshi.cancel(self.generation_id))

    def _assistant_active(self) -> bool:
        if self._assistant_is_speaking:
            return True
        if self._assistant_active_gen is not None:
            return True
        if (
            self._reply_task is not None
            or self._llm_prod_task is not None
            or self._kickoff_task is not None
            or self._finalize_task is not None
            or self._tts_phrase_task is not None
        ):
            return True
        return False

    async def on_audio_frame(self, ts: float, pcm16: bytes):
        if self._closed:
            return
        if len(pcm16) != self.frame_bytes:
            return
        if not self._logged_first_audio_frame:
            self._logged_first_audio_frame = True
            logger.info(
                "Voice: first audio_frame sr=%s frame_bytes=%d ts=%s gen=%s",
                self.sr,
                len(pcm16),
                ts,
                self.generation_id,
            )
        try:
            self.in_q.put_nowait(Frame(ts=ts, pcm16=pcm16))
        except asyncio.QueueFull:
            pass

    async def on_control(self, data: dict):
        """Handle control messages from the browser."""
        t = data.get("type")
        if t == "set_context":
            self.system_prompt = (data.get("system_prompt") or "").strip() or self.system_prompt
            self.use_knowledge_base = bool(data.get("use_knowledge_base", False))
            self.persona = (data.get("persona") or "").strip()
            self.context_window = (data.get("context_window") or "all").strip() or "all"
            self.voice_bot_name = (data.get("voice_bot_name") or "").strip()
            self.voice_user_name = (data.get("voice_user_name") or "").strip()
            # EchoMind profile (optional from client)
            if data.get("assistant_name") is not None:
                self.global_profile["assistant_name"] = str(data.get("assistant_name", "")).strip() or self.global_profile["assistant_name"]
                self.global_profile["wake_word"] = self.global_profile["assistant_name"]
            if data.get("wake_word") is not None:
                self.global_profile["wake_word"] = str(data.get("wake_word", "")).strip() or self.global_profile["wake_word"]
            if data.get("user_name") is not None:
                self.global_profile["user_name"] = str(data.get("user_name", "")).strip()
            if data.get("timezone") is not None:
                self.global_profile["timezone"] = str(data.get("timezone", "")).strip() or "America/New_York"
            if data.get("location") is not None:
                self.global_profile["location"] = str(data.get("location", "")).strip()
            prev_listen = self.listen_only
            self.listen_only = bool(data.get("listen_only", False))
            if prev_listen and not self.listen_only:
                logger.warning("[LISTEN_MODE_OFF] Reason: client set_context listen_only=False (UI or API)")
            triggers = data.get("trigger_phrases")
            if isinstance(triggers, list):
                self.trigger_phrases = [str(x).strip().lower() for x in triggers if str(x).strip()]
            # If user checks "clear memory"
            if bool(data.get("clear_memory")):
                self.history = []
                self.listen_buffer = ""
            # Optional: switch Piper TTS voice if client sent piper_voice (e.g. en_US-lessac-medium)
            piper_voice = (data.get("piper_voice") or "").strip()
            # Fire persona-specific intro on the first set_context (deferred from start())
            if self._pending_intro:
                self._pending_intro = False
                if not bool(data.get("skip_intro")):
                    intro_phrase = _get_persona_intro(self.persona)
                    phrase_speech = strip_markdown_for_speech(intro_phrase) if intro_phrase else ""
                    if phrase_speech:
                        self._is_playing_intro = True
                        asyncio.create_task(self._play_intro(intro_phrase))
            if piper_voice:
                model_path = f"/voices/{piper_voice}.onnx"
                if os.path.exists(model_path):
                    try:
                        self.tts = PiperTTS(
                            model_path,
                            speaker_id=SETTINGS.PIPER_SPEAKER,
                            noise_scale=SETTINGS.PIPER_NOISE_SCALE,
                            length_scale=SETTINGS.PIPER_LENGTH_SCALE,
                            use_cuda=SETTINGS.PIPER_USE_CUDA,
                        )
                    except Exception:
                        pass
            await self.send({"type": "context_ack", "system_prompt": self.system_prompt, "cleared": bool(data.get('clear_memory'))})
            await self._emit_profile_update()

        if t == "clear_memory":
            self.history = []
            await self.send({"type": "context_ack", "system_prompt": self.system_prompt, "cleared": True})

        if t == "speak_text_only":
            # One-shot TTS for Assistant Mode (approved suggestion). No STT/LLM; does not auto-reply to mic.
            text = strip_markdown_for_speech((data.get("text") or "").strip())
            if not text:
                return
            await self._cancel_assistant_pipeline(keep_listening=True, send_cancel=True)
            my_gen = self.generation_id
            await self.send({"type": "event", "event": "SPEAKING", "generation_id": my_gen})
            await self.send({"type": "assistant_text", "generation_id": my_gen, "text": text})
            await self._speak_phrase(my_gen, text)
            await self.send({"type": "event", "event": "BACK_TO_LISTENING", "generation_id": my_gen})

        if t == "interrupt_assistant":
            await self._cancel_assistant_pipeline(keep_listening=True, send_cancel=True)

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

    def _build_system_prompt_with_profile(self, compiled_context: Optional[str] = None) -> str:
        """Prepend profile to system prompt for EchoMind."""
        p = self.global_profile
        parts = [
            f"Assistant name: {p.get('assistant_name') or 'EchoMind'}.",
            f"User name: {p.get('user_name') or 'User'}.",
            f"Timezone: {p.get('timezone') or 'America/New_York'}.",
        ]
        if p.get("location"):
            parts.append(f"Location: {p['location']}.")
        profile_line = " ".join(parts)
        base = self.system_prompt.strip()
        if compiled_context:
            base = base + "\n\nRecent conversation context (for reference):\n" + compiled_context
        return profile_line + " " + base

    async def _emit_profile_update(self) -> None:
        """Send profile_update message to client."""
        await self.send({"type": "profile_update", **self.global_profile})

    def _build_messages(self, user_text: str, system_override: Optional[str] = None) -> List[Dict]:
        self._trim_history()
        sys_content = (
            system_override
            if system_override is not None
            else self._build_system_prompt_with_profile()
        )
        msgs = [{"role": "system", "content": sys_content}]
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
                self._speech_lead_count += 1

                # Skip barge-in during intro to avoid mic feedback cutting it off
                if not self._is_playing_intro:
                    lead_idle = max(1, getattr(SETTINGS, "BARGE_IN_SPEECH_LEAD_IDLE", 2))
                    lead_active = max(1, getattr(SETTINGS, "BARGE_IN_SPEECH_LEAD_ACTIVE", 6))
                    need_lead = lead_active if self._assistant_active() else lead_idle

                    if not self.in_speech:
                        self._pending_lead_frames.append(fr)
                        if self._speech_lead_count >= need_lead:
                            self.in_speech = True
                            self.utt.reset()
                            for _pf in self._pending_lead_frames:
                                self.utt.push(_pf)
                            self._pending_lead_frames.clear()
                            await self._cancel_assistant_pipeline(keep_listening=True, send_cancel=True)
                            await self.send({"type": "event", "event": "USER_SPEECH_START", "generation_id": self.generation_id})
                            if self.moshi:
                                asyncio.create_task(self.moshi.cancel(self.generation_id))
                    else:
                        self.utt.push(fr)

            else:
                self._speech_lead_count = 0
                if not self.in_speech:
                    self._pending_lead_frames.clear()
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
            user_text = await self.stt.transcribe(audio)
        except Exception as e:
            await self.send({"type": "error", "where": "stt", "message": str(e), "generation_id": my_gen})
            return

        if my_gen != self.generation_id:
            logger.info("Voice STT: dropped result after barge-in/cancel gen=%s current=%s", my_gen, self.generation_id)
            return

        user_text = (user_text or "").strip()
        if not user_text:
            return
        user_text = strip_markdown_for_speech(user_text)
        if not user_text:
            return
        # English-only preprocessing: retain only English words, reject noise (do not send garbage to LLM)
        user_text = preprocess_english_only(user_text)
        if not user_text:
            return

        try:
            return await self._finalize_and_reply_impl(my_gen, user_text)
        finally:
            self._last_user_utterance = user_text

    async def _reply_from_backend_rag_stream(
        self,
        my_gen: int,
        user_text: str,
        backend_url: str,
        payload: dict,
    ) -> None:
        """Stream NDJSON from POST /api/chat/ask-voice-stream; phrase-commit + TTS like local LLM stream."""
        phrase_buf = ""
        assistant_text = ""
        rag_citations: list = []
        last_emit = time.time()
        phrases_enqueued = 0

        tts_q: asyncio.Queue = asyncio.Queue()
        chunk_q: asyncio.Queue = asyncio.Queue()
        loop_c = asyncio.get_running_loop()
        base = backend_url.rstrip("/")
        stream_url = f"{base}/api/chat/ask-voice-stream"

        async def tts_consumer():
            while True:
                item = await tts_q.get()
                if item is None:
                    break
                if my_gen != self.generation_id:
                    continue
                await self._commit_phrase(my_gen, item)

        def run_backend_ndjson():
            def put(it):
                loop_c.call_soon_threadsafe(chunk_q.put_nowait, it)

            try:
                with requests.post(
                    stream_url,
                    json=payload,
                    stream=True,
                    timeout=180,
                    headers={"Content-Type": "application/json"},
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        t = obj.get("type")
                        if t == "chunk":
                            tx = obj.get("text") or ""
                            if tx:
                                put(tx)
                        elif t == "done":
                            put(
                                {
                                    "__done__": True,
                                    "answer": (obj.get("answer") or ""),
                                    "citations": obj.get("citations") if isinstance(obj.get("citations"), list) else [],
                                }
                            )
                        elif t == "error":
                            put({"__error__": str(obj.get("message") or "error")})
            except Exception as e:
                put({"__error__": str(e)})
            finally:
                put(None)

        _abort_tts = False
        self._tts_phrase_task = asyncio.create_task(tts_consumer())
        try:

            async def producer():
                await loop_c.run_in_executor(None, run_backend_ndjson)

            self._llm_prod_task = asyncio.create_task(producer())
            prod_task = self._llm_prod_task

            while True:
                if my_gen != self.generation_id:
                    prod_task.cancel()
                    return

                item = await chunk_q.get()
                if item is None:
                    break
                if isinstance(item, dict) and item.get("__error__"):
                    raise RuntimeError(item["__error__"])
                if isinstance(item, dict) and item.get("__done__"):
                    server_ans = (item.get("answer") or "").strip()
                    if server_ans and len(server_ans) > len(assistant_text.strip()):
                        assistant_text = server_ans
                    raw_cites = item.get("citations")
                    rag_citations = raw_cites if isinstance(raw_cites, list) else []
                    continue

                assistant_text += item
                phrase_buf += item
                await self.send(
                    {
                        "type": "assistant_text_partial",
                        "generation_id": my_gen,
                        "text": strip_markdown_for_speech(assistant_text),
                    }
                )
                if phrase_commit_needed(phrase_buf, phrases_enqueued, last_emit):
                    await tts_q.put(phrase_buf.strip())
                    phrases_enqueued += 1
                    phrase_buf = ""
                    last_emit = time.time()

            if my_gen != self.generation_id:
                return
            if phrase_buf.strip():
                await tts_q.put(phrase_buf.strip())
            final = strip_markdown_for_speech(assistant_text.strip())
            _log_llm_response(user_text, final)
            if final:
                await self.send({"type": "assistant_text", "generation_id": my_gen, "text": final})
            if rag_citations:
                await self.send(
                    {
                        "type": "assistant_citations",
                        "message_id": str(my_gen),
                        "generation_id": my_gen,
                        "citations": rag_citations,
                    }
                )
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": final})
            self._trim_history()
            try:
                self.conversation_memory.add_text(final, speaker="assistant")
            except Exception:
                pass

        except Exception as e:
            _abort_tts = True
            try:
                while True:
                    tts_q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                await tts_q.put(None)
            except Exception:
                pass
            wtx = self._tts_phrase_task
            self._tts_phrase_task = None
            if wtx and not wtx.done():
                wtx.cancel()
                try:
                    await wtx
                except asyncio.CancelledError:
                    pass
            # One-shot fallback: full answer then speak (offline / proxy issues)
            try:
                r = requests.post(
                    f"{base}/api/chat/ask-voice",
                    json=payload,
                    timeout=120,
                    headers={"Content-Type": "application/json"},
                )
                r.raise_for_status()
                body = r.json()
                answer = (body.get("answer") or "").strip()
                fallback_cites = body.get("citations") if isinstance(body.get("citations"), list) else []
                if my_gen != self.generation_id:
                    return
                reply_clean = strip_markdown_for_speech(answer)
                _log_llm_response(user_text, reply_clean or answer)
                if reply_clean:
                    await self.send({"type": "assistant_text", "generation_id": my_gen, "text": reply_clean})
                if fallback_cites:
                    await self.send(
                        {
                            "type": "assistant_citations",
                            "message_id": str(my_gen),
                            "generation_id": my_gen,
                            "citations": fallback_cites,
                        }
                    )
                await self._speak_phrase(my_gen, answer)
                self.history.append({"role": "user", "content": user_text})
                self.history.append({"role": "assistant", "content": reply_clean or answer})
                self._trim_history()
                try:
                    self.conversation_memory.add_text(reply_clean or answer, speaker="assistant")
                except Exception:
                    pass
            except Exception as e2:
                await self.send(
                    {
                        "type": "error",
                        "where": "backend_rag",
                        "message": f"{e} (fallback: {e2})",
                        "generation_id": my_gen,
                    }
                )
        finally:
            if _abort_tts:
                pass
            else:
                try:
                    await tts_q.put(None)
                except Exception:
                    pass
                wt = self._tts_phrase_task
                self._tts_phrase_task = None
                if wt:
                    try:
                        await wt
                    except asyncio.CancelledError:
                        pass

    async def _finalize_and_reply_impl(self, my_gen: int, user_text: str):
        """Inner implementation; _last_user_utterance is set by caller after return."""
        # Always store user utterance in EchoMind conversation memory
        try:
            self.conversation_memory.add_text(user_text, speaker="user")
        except Exception:
            pass

        ut_lower = user_text.lower()
        wake_word = (self.global_profile.get("wake_word") or "echomind").strip().lower()
        # In listen-only: switch when we see "EchoMind", "Echo Mind", or "Stop listening"
        if self.listen_only:
            wake_patterns = [ r"\b" + re.escape(wake_word) + r"\b", r"\becho\s+mind\b", r"\bstop\s+listening\b", r"\bstoplistening\b"]
            triggered = any(re.search(p, ut_lower) for p in wake_patterns)
            # "Stop listening" is handled by intent router (runs first)
        else:
            stripped_for_wake = strip_wake_word(user_text, self.global_profile.get("wake_word") or "")
            wake_word_triggered = wake_word and (stripped_for_wake != ut_lower)
            triggered = wake_word_triggered

        # Intent router (EchoMind commands)
        memory_summary = self.conversation_memory.get_entries_for_context(5, max_chars=500)
        last_utterance = (self.listen_buffer if self.listen_only else None) or self._last_user_utterance
        handled, response_text, extra = parse_and_route(
            user_text,
            self.global_profile,
            memory_summary,
            self.listen_only,
            self.trigger_phrases,
            pending_wake_word_change=self.pending_wake_word_change,
            last_utterance=last_utterance,
        )

        # Apply profile / listen / clear from intent
        if extra.get("pending_wake_word_change"):
            self.pending_wake_word_change = extra["pending_wake_word_change"]
        if extra.get("clear_pending_wake_word_change"):
            self.pending_wake_word_change = None
        if extra.get("confirm_wake_word_change") and self.pending_wake_word_change:
            new_wake = self.pending_wake_word_change
            self.global_profile["assistant_name"] = new_wake
            self.global_profile["wake_word"] = new_wake
            self.pending_wake_word_change = None
            save_wake_word(new_wake)
            logger.info("[WAKE_WORD_CHANGED] new=%r", new_wake)
            await self._emit_profile_update()
        if extra.get("set_assistant_name"):
            self.global_profile["assistant_name"] = extra["set_assistant_name"]
            self.global_profile["wake_word"] = extra["set_assistant_name"]
            save_wake_word(extra["set_assistant_name"])
            await self._emit_profile_update()
        if extra.get("set_user_name"):
            self.global_profile["user_name"] = extra["set_user_name"]
            await self._emit_profile_update()
        if extra.get("set_timezone"):
            self.global_profile["timezone"] = extra["set_timezone"]
            await self._emit_profile_update()
        if extra.get("set_location"):
            self.global_profile["location"] = extra["set_location"]
            await self._emit_profile_update()
        if "set_listen_only" in extra:
            prev_listen = self.listen_only
            self.listen_only = bool(extra["set_listen_only"])
            if prev_listen and not self.listen_only:
                logger.warning("[LISTEN_MODE_OFF] Reason: voice command (e.g. 'Stop listening'). user_text=%r", user_text[:120])
            elif not prev_listen and self.listen_only:
                logger.warning("[LISTEN_MODE_ON] Reason: voice command (e.g. 'Start listening'). user_text=%r", user_text[:120])
            if getattr(SETTINGS, "ECHO_DEBUG", False):
                logger.info("EchoMind listen_only=%s", self.listen_only)
            await self.send({"type": "memory_event", "event": "listening_mode_on" if self.listen_only else "listening_mode_off"})
        if extra.get("clear_memory"):
            self.history = []
            self.listen_buffer = ""

        # Handled command with direct response (e.g. "Your name is X", "Start listening", "Stop listening")
        # Must run BEFORE listen-only accumulate so "Start listening" / "Stop listening" get spoken
        if handled and response_text and not extra.get("fact_check") and not extra.get("memory_query_type"):
            self.turn_id += 1
            await self.send({"type": "asr_final", "turn_id": self.turn_id, "generation_id": my_gen, "text": user_text})
            await self.send({"type": "event", "event": "SPEAKING", "generation_id": my_gen})
            self._assistant_active_gen = my_gen
            await self.send({"type": "assistant_text", "generation_id": my_gen, "text": response_text})
            await self._speak_phrase(my_gen, response_text)
            try:
                self.conversation_memory.add_text(response_text, speaker="assistant")
            except Exception:
                pass
            await self.send({"type": "event", "event": "BACK_TO_LISTENING", "generation_id": my_gen})
            self._assistant_active_gen = None
            return

        # Listen-only and NOT trigger: only accumulate, do NOT send to LLM or speak (hold until wake word)
        if self.listen_only and not triggered:
            self.listen_buffer = (self.listen_buffer + " " + user_text).strip() if self.listen_buffer else user_text
            self.turn_id += 1
            await self.send({"type": "asr_final", "turn_id": self.turn_id, "generation_id": my_gen, "text": user_text})
            await self.send({"type": "listen_buffer", "text": self.listen_buffer})
            await self.send({"type": "event", "event": "BACK_TO_LISTENING", "generation_id": my_gen})
            self._assistant_active_gen = None
            return

        # Wake word seen: switch mode; send combined content to LLM; LLM responds to that message (no special trigger)
        reenter_listen_only = False
        if self.listen_only and triggered:
            self.listen_only = False
            reenter_listen_only = True
            combined = self.listen_buffer.strip()
            self.listen_buffer = ""
            # Combined content = prior buffer + current utterance; send as single user message to LLM
            user_text = (combined + " " + user_text).strip() if combined else user_text
            logger.warning(
                "[LISTEN_MODE_OFF] Wake word seen. buffer_len=%d combined_len=%d user_text=%r",
                len(combined),
                len(user_text),
                user_text,
            )
            await self.send({"type": "memory_event", "event": "listening_mode_off"})

        self.turn_id += 1
        await self.send({"type": "asr_final", "turn_id": self.turn_id, "generation_id": my_gen, "text": user_text})
        await self.send({"type": "event", "event": "SPEAKING", "generation_id": my_gen})
        self._assistant_active_gen = my_gen

        # Compiled context for LLM (recent memory + optional topic)
        compiled_context = self.conversation_memory.get_entries_for_context(15, max_chars=3500)

        # Fact-check flow: use recent memory, LLM with fact-check instruction, optional RAG
        if extra.get("fact_check"):
            fc_context = self.conversation_memory.get_entries_for_context(10, max_chars=3000)
            fc_prompt = (
                "You are a fact-checking assistant for financial and regulatory discussions. "
                "Based ONLY on the following conversation transcript, identify any factual claims "
                "(especially about DoD FMR, regulations, procedures, or compliance) and assess their accuracy. "
                "If you have no external sources, clearly state uncertainty and give reasoning. Be concise.\n\nTranscript:\n" + fc_context
            )
            messages_fc = self._build_messages(user_text, system_override=fc_prompt)
            backend_url = (getattr(SETTINGS, "BACKEND_CHAT_URL", None) or "").strip().rstrip("/")
            if self.use_knowledge_base and backend_url and _user_asks_about_knowledge(user_text):
                payload = {
                    "message": f"Fact-check the following. User request: {user_text}\n\nContext:\n{fc_context}",
                    "persona": self.persona or None,
                    "context_window": self.context_window or "all",
                    "use_knowledge_base": True,
                    "advanced_rag": True,
                    "voice_max_tokens": getattr(SETTINGS, "VOICE_RAG_MAX_TOKENS", 640),
                }
                await self._reply_from_backend_rag_stream(my_gen, user_text, backend_url, payload)
            else:
                try:
                    reply = self.llm.complete_messages(messages_fc)
                    reply_clean = strip_markdown_for_speech(reply)
                    _log_llm_response(user_text, reply_clean or reply)
                    if reply_clean:
                        await self.send({"type": "assistant_text", "generation_id": my_gen, "text": reply_clean})
                    await self._speak_phrase(my_gen, reply)
                    self.history.append({"role": "user", "content": user_text})
                    self.history.append({"role": "assistant", "content": reply_clean or reply})
                    self._trim_history()
                    try:
                        self.conversation_memory.add_text(reply_clean or reply, speaker="assistant")
                    except Exception:
                        pass
                except Exception as e:
                    await self.send({"type": "error", "where": "llm", "message": str(e), "generation_id": my_gen})
            if reenter_listen_only:
                self.listen_only = True
                await self.send({"type": "memory_event", "event": "listening_mode_on"})
            await self.send({"type": "event", "event": "BACK_TO_LISTENING", "generation_id": my_gen})
            self._assistant_active_gen = None
            return

        # Memory query: recap / summarize / when mentioned / timestamps
        if extra.get("memory_query_type"):
            minutes = extra.get("memory_query_minutes") or 5.0
            query_type = extra.get("memory_query_type")
            if query_type == "recap":
                recap = self.conversation_memory.summarize_last(minutes)
                if recap:
                    await self.send({"type": "memory_info", "generation_id": my_gen, "summary": recap, "minutes": minutes})
                    reply = f"In the last {int(minutes)} minutes, here's what was said:\n\n{recap}" if len(recap) < 1500 else recap[:1500] + "..."
                else:
                    reply = f"I don't have anything in the last {int(minutes)} minutes."
                _log_llm_response(user_text, reply)
                await self.send({"type": "assistant_text", "generation_id": my_gen, "text": reply})
                await self._speak_phrase(my_gen, reply[:500] if len(reply) > 500 else reply)
            elif query_type == "summarize":
                summary_text = self.conversation_memory.summarize_last(minutes)
                if summary_text:
                    messages_sum = self._build_messages(
                        f"Summarize this conversation from the last {int(minutes)} minutes in 2-4 sentences.",
                        system_override="You are a concise summarizer for financial and regulatory discussions. "
                        "Summarize key points, decisions, and any references to regulations or procedures. Output only the summary, no preamble.\n\nConversation:\n" + summary_text[:3000],
                    )
                    try:
                        reply = self.llm.complete_messages(messages_sum)
                        reply_clean = strip_markdown_for_speech(reply)
                        _log_llm_response(user_text, reply_clean or reply)
                        await self.send({"type": "assistant_text", "generation_id": my_gen, "text": reply_clean})
                        await self._speak_phrase(my_gen, reply_clean)
                    except Exception as e:
                        await self.send({"type": "error", "where": "llm", "message": str(e), "generation_id": my_gen})
                        await self._speak_phrase(my_gen, "I couldn't generate a summary.")
                else:
                    await self._speak_phrase(my_gen, f"No conversation in the last {int(minutes)} minutes to summarize.")
            elif query_type == "timestamps_tags":
                entries = self.conversation_memory.query_last(minutes)
                lines = []
                for e in entries:
                    ts_str = time.strftime("%H:%M", time.localtime(e.ts_start))
                    lines.append(f"[{ts_str}] {e.speaker or 'user'}: {e.text[:80]}{'...' if len(e.text) > 80 else ''}" + (f" tags={e.tags}" if e.tags else ""))
                reply = "\n".join(lines) if lines else "No entries in that window."
                _log_llm_response(user_text, reply)
                await self.send({"type": "memory_info", "generation_id": my_gen, "entries": [e.to_dict() for e in entries]})
                await self.send({"type": "assistant_text", "generation_id": my_gen, "text": reply})
                await self._speak_phrase(my_gen, reply[:400] if len(reply) > 400 else reply)
            else:
                # when_mentioned: keyword search
                topic = extra.get("memory_query_topic") or user_text
                entries = self.conversation_memory.query_topic(topic)
                if entries:
                    summary = self.conversation_memory.summarize_last(30)  # use last 30 min for context
                    messages_when = self._build_messages(
                        f"When did we talk about this? User asked: {user_text}",
                        system_override="Use only this transcript. List approximate times and who said what. "
                        "Focus on financial topics, regulations, or procedures when mentioned.\n\n" + summary[:2500],
                    )
                    try:
                        reply = self.llm.complete_messages(messages_when)
                        reply_clean = strip_markdown_for_speech(reply)
                        _log_llm_response(user_text, reply_clean or reply)
                        await self.send({"type": "assistant_text", "generation_id": my_gen, "text": reply_clean})
                        await self._speak_phrase(my_gen, reply_clean)
                    except Exception as e:
                        await self.send({"type": "error", "where": "llm", "message": str(e), "generation_id": my_gen})
                        await self._speak_phrase(my_gen, "I couldn't find that.")
                else:
                    await self._speak_phrase(my_gen, "I don't have any mentions of that in recent conversation.")
            if reenter_listen_only:
                self.listen_only = True
                await self.send({"type": "memory_event", "event": "listening_mode_on"})
            await self.send({"type": "event", "event": "BACK_TO_LISTENING", "generation_id": my_gen})
            self._assistant_active_gen = None
            return

        # RAG path: only when user message indicates document/transcript/resources/book/pdf/file (otherwise general conversation)
        backend_url = (getattr(SETTINGS, "BACKEND_CHAT_URL", None) or "").strip().rstrip("/")
        if self.use_knowledge_base and backend_url and _user_asks_about_knowledge(user_text):
            payload = {
                "message": user_text,
                "persona": self.persona or None,
                "context_window": self.context_window or "all",
                "use_knowledge_base": True,
                "advanced_rag": True,
                "voice_max_tokens": getattr(SETTINGS, "VOICE_RAG_MAX_TOKENS", 640),
            }
            await self._reply_from_backend_rag_stream(my_gen, user_text, backend_url, payload)
            if reenter_listen_only:
                self.listen_only = True
                await self.send({"type": "memory_event", "event": "listening_mode_on"})
            await self.send({"type": "event", "event": "BACK_TO_LISTENING", "generation_id": my_gen})
            self._assistant_active_gen = None
            return

        # LLM stream path with compiled context in system prompt
        messages = self._build_messages(user_text)
        if compiled_context:
            sys_with_ctx = self._build_system_prompt_with_profile(compiled_context=compiled_context)
            messages[0]["content"] = sys_with_ctx

        phrase_buf = ""
        assistant_text = ""
        last_emit = time.time()
        phrases_enqueued = 0

        # Serial TTS queue: main loop keeps draining LLM tokens while Piper plays phrases in order.
        tts_q: asyncio.Queue = asyncio.Queue()

        async def tts_consumer():
            while True:
                item = await tts_q.get()
                if item is None:
                    break
                if my_gen != self.generation_id:
                    continue
                await self._commit_phrase(my_gen, item)

        self._tts_phrase_task = asyncio.create_task(tts_consumer())
        _abort_tts = False
        try:
            tok_q: asyncio.Queue = asyncio.Queue()

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

                assistant_text += tok
                phrase_buf += tok
                await self.send(
                    {
                        "type": "assistant_text_partial",
                        "generation_id": my_gen,
                        "text": strip_markdown_for_speech(assistant_text),
                    }
                )

                if phrase_commit_needed(phrase_buf, phrases_enqueued, last_emit):
                    await tts_q.put(phrase_buf.strip())
                    phrases_enqueued += 1
                    phrase_buf = ""
                    last_emit = time.time()

            if my_gen != self.generation_id:
                return
            if phrase_buf.strip():
                await tts_q.put(phrase_buf.strip())
            final = strip_markdown_for_speech(assistant_text.strip())
            _log_llm_response(user_text, final)
            if final:
                await self.send({"type": "assistant_text", "generation_id": my_gen, "text": final})
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": final})
            self._trim_history()
            try:
                self.conversation_memory.add_text(final, speaker="assistant")
            except Exception:
                pass

        except Exception as e:
            _abort_tts = True
            try:
                while True:
                    tts_q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                await tts_q.put(None)
            except Exception:
                pass
            wtx = self._tts_phrase_task
            self._tts_phrase_task = None
            if wtx and not wtx.done():
                wtx.cancel()
                try:
                    await wtx
                except asyncio.CancelledError:
                    pass
            await self.send({"type": "error", "where": "llm_stream", "message": str(e), "generation_id": my_gen})
            try:
                reply = self.llm.complete_messages(messages)
            except Exception as e2:
                await self.send({"type": "error", "where": "llm", "message": str(e2), "generation_id": my_gen})
                self._assistant_active_gen = None
                return
            reply_clean = strip_markdown_for_speech(reply)
            _log_llm_response(user_text, reply_clean or reply)
            if reply_clean:
                await self.send({"type": "assistant_text", "generation_id": my_gen, "text": reply_clean})
            await self._speak_phrase(my_gen, reply)
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply_clean or reply})
            self._trim_history()
            try:
                self.conversation_memory.add_text(reply_clean or reply, speaker="assistant")
            except Exception:
                pass
        finally:
            if _abort_tts:
                pass
            else:
                try:
                    await tts_q.put(None)
                except Exception:
                    pass
                wt = self._tts_phrase_task
                self._tts_phrase_task = None
                if wt:
                    try:
                        await wt
                    except asyncio.CancelledError:
                        pass

        await self.send({"type": "event", "event": "BACK_TO_LISTENING", "generation_id": my_gen})
        self._assistant_active_gen = None

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
        phrase = strip_markdown_for_speech(phrase or "")
        if not phrase:
            return
        self._assistant_is_speaking = True
        try:
            rate = 1.0
            if SETTINGS.EMOTION_MODE and not is_filler:
                rate = detect_emotion_playback_rate(phrase)

            try:
                loop = asyncio.get_running_loop()
                y = await loop.run_in_executor(None, self.tts.synth, phrase)
                sr = self.tts.sr
            except Exception as e:
                await self.send({"type": "error", "where": "tts", "message": str(e), "generation_id": my_gen})
                return

            # Larger chunks = fewer boundaries = fewer clicks; 0.35s at 22kHz ~= 7700 samples
            chunk_samples = int(sr * 0.35)
            chunk_samples = max(chunk_samples, 256)
            i = 0
            while i < y.size:
                if my_gen != self.generation_id:
                    return
                part = y[i : i + chunk_samples].astype(np.float32)
                i += chunk_samples
                # Short fade at edges to avoid crackling at chunk boundaries
                _fade_chunk_edges(part, sr, fade_ms=4.0)
                await self.send({
                    "type": "audio_out",
                    "generation_id": my_gen,
                    "sample_rate": sr,
                    "playback_rate": rate,
                    "pcm16_raw": float32_to_pcm16_bytes(part),
                })
                await asyncio.sleep(0.0)
        finally:
            if self.generation_id == my_gen:
                self._assistant_is_speaking = False

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