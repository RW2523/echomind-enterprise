# Conversation AI & Continuous Listening (Wake Word) Flow

This document maps the **entire conversation AI flow** in EchoMind and how it implements (or can be extended for) **Continuous Listening Mode with Wake Word “EchoMind”**.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  FRONTEND                                                                        │
│  • React: KnowledgeChat (text), useVoiceConnection (voice WS), LiveTranscription │
│  • Voice UI: connect → set_context → stream mic → play TTS / handle events       │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────────────────┐
│ Backend       │     │ Voice service    │     │ Backend (transcribe)          │
│ /api/chat/*   │     │ /voice/ws        │     │ /transcribe/ws (Nemotron STT) │
│ • ask         │     │ OmniSessionA     │     │ (separate: live transcript   │
│ • ask-stream  │     │ • VAD → Whisper │     │  recording, not voice AI)    │
│ • ask-voice   │     │ • LLM stream     │     └─────────────────────────────┘
│ • RAG/LLM     │     │ • Piper TTS      │
└───────────────┘     │ • Listen-only   │
                      │ • Wake word     │
                      └─────────────────┘
```

- **Text chat**: Frontend → Backend `POST /api/chat/ask` or `ask-stream` → RAG/LLM → DB (messages, conversation_summary).
- **Voice**: Frontend → Voice WebSocket `/voice/ws` → **OmniSessionA** (VAD → Whisper STT → intent/wake word → LLM/TTS). Optionally voice service calls Backend `POST /api/chat/ask-voice` for RAG when the user asks about documents/transcripts.

---

## 2. Voice Conversation AI Flow (OmniSessionA)

The voice flow lives in **`voice/app/session.py`** and **`voice/app/echo_commands.py`**.

### 2.1 Connection & Startup

1. **Client** opens WebSocket to `voice/ws`.
2. **Server** creates `OmniSessionA(ws)`, calls `sess.start(session_id)`.
3. **start()**:
   - Initializes `listen_buffer = []`, `listen_only = False`.
   - Starts `_sender_loop`, `_consume_loop`, optional Moshi.
   - Sends `hello`, `context_ack`, `profile_update`.
   - Plays intro TTS (e.g. “Hi! I'm here. What would you like to talk about?”).
4. **Client** sends `set_context` (system_prompt, persona, use_knowledge_base, voice_bot_name, **listen_only**, **trigger_phrases**, etc.).  
   The frontend (`useVoiceConnection.ts`) does **not** currently send `listen_only`; it can be set by **voice** (“start listening”) or by adding it to `set_context`.

### 2.2 Audio → Speech (VAD + STT)

- **Client** streams PCM16 audio frames (`type: "audio_frame"`, `pcm16_b64`, `ts`) to the server.
- **Server** (`_consume_loop`):
  - Puts frames in `in_q`.
  - For each frame: VAD (webrtcvad + RMS) → if speech, push to **UtteranceBuffer** (max ~15s); on **silence endpoint** (configurable silence frames) → **USER_SPEECH_END**.
  - On endpoint: enqueue **`_finalize_and_reply(my_gen)`** (unless too short).

So **transcription is utterance-based**: each “chunk” is one VAD-ended utterance, then the voice service STT transcribes that chunk. There is **no** shared continuous streaming STT socket with the Transcribe tab (the backend has Nemotron streaming STT in `backend/app/transcribe/ws.py` for **Transcribe**, not for the voice assistant binary protocol).

### 2.3 After STT: Intent, Wake Word, Listen-Only

In **`_finalize_and_reply`** (after Whisper returns `user_text`):

1. **Preprocess**: `strip_markdown_for_speech`, `preprocess_english_only` (noise filter).
2. **Conversation memory**: Every user utterance is stored:  
   `self.conversation_memory.add_text(user_text, speaker="user")`  
   (rolling window, see `voice/app/conversation_memory.py`).
3. **Wake word & trigger**:
   - `wake_word = self.global_profile.get("wake_word")` (default `"EchoMind"`).
   - `stripped_for_wake = strip_wake_word(user_text, wake_word)` → if the utterance **starts** with the wake word, the rest is the query.
   - `wake_word_triggered = wake_word and (stripped_for_wake != ut_lower)`.
   - `triggered = wake_word_triggered or any(trig in ut_lower for trig in self.trigger_phrases)`  
     (trigger_phrases: “now you can speak”, “process that”, “fact check it”, etc.).
4. **Intent router**: `parse_and_route(user_text, profile, memory_summary, listen_only, trigger_phrases)` in **`voice/app/echo_commands.py`**:
   - **Start listening**: phrases like “listen to conversation”, “**start listening**”, “just listen” → `set_listen_only = True`, response *“I'm now listening to the conversation. Say your wake word or 'now you can speak' when you want me to respond.”*
   - **Stop listening**: “stop listening”, “pause”, etc. → `set_listen_only = False`.
   - Other intents: set name, timezone, location, clear memory, memory queries (recap/summarize/when mentioned), fact-check.

### 2.4 Listen-Only Mode (Continuous Listening)

- **Activation**: User says “Start listening” (or “listen to conversation”, etc.) → intent sets `listen_only = True`; server speaks the confirmation and sends `memory_event: listening_mode_on`.
- **While `listen_only` is True**:
  - Every VAD-ended utterance is still **transcribed** and **added to conversation_memory**.
  - If the utterance is **not** a trigger (no wake word, none of the trigger phrases):
    - Text is **appended to `listen_buffer`** (in-memory list of strings).
    - Server sends `asr_final` (so client can show transcript) and **BACK_TO_LISTENING** (no LLM, no TTS).
  - So in listen-only mode the assistant is **silent** and only **accumulates** transcribed segments in `listen_buffer` and in `conversation_memory`.
- **Wake word (or trigger) in an utterance**:
  - When `triggered` is True (wake word at start or trigger phrase):
    - Server sets `listen_only = False`.
    - **Compiled context**: `combined = " ".join(self.listen_buffer) + " " + user_text` (accumulated speech + current utterance).
    - If wake word was at start: the **query** sent to the LLM is the part after the wake word (`stripped_for_wake`), but the **context** for the LLM includes `compiled_context` (via conversation_memory / system context).
    - Then normal path: build messages, optional RAG (`ask-voice`), LLM stream or complete, TTS, history + conversation_memory update, **BACK_TO_LISTENING**.

So:

- **Buffered transcript**: Implemented as `listen_buffer` (list of utterance strings) plus **ConversationMemory** (time-windowed entries with timestamps). Accumulation is **per utterance** (after each VAD endpoint), not per streaming token.
- **Wake word detection**: Implemented **on the transcript** of each utterance (no separate acoustic wake-word engine). The server checks whether the **current** utterance starts with the wake word (or contains a trigger phrase).
- **Post-response**: After responding to a wake-word/trigger, the code sets `listen_only = False` and does **not** automatically set it back to True. So after one response, the session returns to **normal** (responds to every utterance) until the user says “Start listening” again.

---

## 3. Backend Chat (Text) Flow

- **Routes**: `backend/app/api/routes/chat.py`
  - **POST /api/chat/ask**: one-shot answer; loads history from DB, optional transcript-time query, then `answer_with_citations` or `_answer_general`; saves user + assistant message; background conversation summary update.
  - **POST /api/chat/ask-stream**: same but streams chunks; saves assistant message on done.
  - **POST /api/chat/ask-voice-stream**: used by the **voice** service for knowledge questions; streams NDJSON chunks (same RAG as chat) so TTS can start early. Falls back to **POST /api/chat/ask-voice** if streaming fails.
- **RAG/LLM**: `backend/app/rag/advanced.py`
  - General vs RAG path (e.g. `_is_general_conversation` → no retrieval; else retrieve → build context → LLM with “EchoMind” system prompt).
  - Conversation summary is stored in `chats.conversation_summary` and used for context.

The voice assistant uses this backend only when **use_knowledge_base** is True and the user message matches **RAG indicator phrases** (document, transcript, FMR, section/paragraph refs, etc.); then the voice service calls **ask-voice-stream** (chunked TTS) with fallback to **ask-voice**.

---

## 4. Mapping to Your Spec: Continuous Listening + Wake Word

| Spec item | Current implementation | Notes |
|----------|------------------------|--------|
| **Activation** “Start listening.” | ✅ “Start listening” (and “listen to conversation”, etc.) in `echo_commands.py` → `set_listen_only = True`. | Exact phrase “Start listening” is supported. |
| **Assistant response** “I am starting to listen. You can call me using the wake word ‘EchoMind’.” | ✅ Implemented: reply uses the profile wake word (default EchoMind): *“I am starting to listen. You can call me using the wake word 'EchoMind'.”* | |
| **Listening mode (silent)** | ✅ When `listen_only` is True, no LLM/TTS for non-trigger utterances; only accumulation. | |
| **Continuous capture + transcribe** | ✅ Audio captured and transcribed per **utterance** (VAD endpoint → Whisper). | Not **streaming** STT (no live character stream); accumulation is per utterance. |
| **Accumulate in memory** | ✅ `listen_buffer` (list of strings) + `conversation_memory.add_text(...)` for every utterance. | Synchronous/near real-time per utterance. |
| **Wake word “EchoMind”** | ✅ `strip_wake_word(utterance, "EchoMind")`; trigger if utterance **starts** with wake word. | Detection is on **transcript**, not on raw audio. |
| **Change wake word** | ✅ "Change wake word to X" → confirmation → "Yes" → applied. Stored in `/voices/wake_word.json`. | Supports "change wake word to that" (uses last utterance). |
| **Triggered response** | ✅ On trigger: combine `listen_buffer` + current utterance, extract query (wake-word-stripped if applicable), then LLM + TTS. | |
| **Post-response: continue listening** | ✅ After a triggered response, the session sets `listen_only = True` again and sends `memory_event: listening_mode_on`, so the assistant returns to silent listening until the next wake word or “Stop listening”. | |
| **Exit: “Stop listening”** | ✅ In `echo_commands.py`: “stop listening”, “pause listening”, “pause”, “don't listen” → `set_listen_only = False`. | |

---

## 5. Key Files Reference

| Area | File | Role |
|------|------|------|
| Voice session | `voice/app/session.py` | OmniSessionA: VAD, STT, listen_only, listen_buffer, wake word check, intent, LLM, TTS. |
| Voice intents | `voice/app/echo_commands.py` | parse_and_route, strip_wake_word, “start/stop listening”, profile, memory/fact-check. |
| Wake word storage | `voice/app/wake_word_storage.py` | load_wake_word, save_wake_word; persistent storage in `/voices/wake_word.json`. |
| Conversation memory | `voice/app/conversation_memory.py` | Rolling buffer, add_text, query_last, summarize_last, get_entries_for_context. |
| Voice config | `voice/app/config.py` | DEFAULT_ASSISTANT_NAME (wake word), MEMORY_WINDOW_MINUTES, etc. |
| Voice WS | `voice/app/server.py` | WebSocket accept → OmniSessionA, on_audio_frame / on_control. |
| Backend chat | `backend/app/api/routes/chat.py` | ask, ask-stream, ask-voice; DB messages, conversation summary. |
| Backend RAG | `backend/app/rag/advanced.py` | answer_with_citations, EchoMind system prompt, general vs RAG path. |
| Frontend voice | `frontend/hooks/useVoiceConnection.ts` | WS connect, set_context (no listen_only sent), playback, events. |

---

## 6. Optional Enhancements (from your spec)

1. **Activation phrase**  
   Optionally add an exact “Start listening.” response:  
   *“I am starting to listen. You can call me using the wake word ‘EchoMind’.”*  
   in `echo_commands.py` when `set_listen_only` is True (e.g. when the phrase is exactly “start listening” or a dedicated activation).

2. **Return to listen-only after response**  
   In `session.py`, after handling a wake-word/trigger response (after BACK_TO_LISTENING), set `self.listen_only = True` again so the assistant keeps listening until “Stop listening.” (Optionally make this configurable.)

3. **Streaming STT + live transcript buffer**  
   For “real-time buffer accumulation” in the strict sense (streaming words as they’re recognized), the voice pipeline would need integration with a streaming STT path (e.g. token streaming from the voice service’s Nemotron adapter) and a **streaming** buffer that is scanned for the wake word as text arrives. Current design is simpler: wake word is checked once per **utterance** after STT returns.

4. **Frontend**  
   ✅ Implemented: Control bar has “Start listening” / “Stop listening” button; `set_context` sends `listen_only`; `memory_event` (listening_mode_on/off) updates `state.listenOnly`; “Listening mode” hint is shown when active.

---

## 7. State Summary (matches your table)

| State | Behavior in code |
|-------|-------------------|
| **Idle** | Not connected, or connected but no speech. |
| **Start Listening** | User says “Start listening” → intent sets `listen_only = True` → server announces and sends `listening_mode_on`. |
| **Listening Mode** | `listen_only` True; each utterance → STT → add to conversation_memory + listen_buffer; no LLM/TTS; BACK_TO_LISTENING. |
| **Wake Word Detected** | Current utterance starts with wake word (or contains trigger phrase) → `triggered` True. |
| **Response Mode** | combined = listen_buffer + current text; query = wake-word-stripped or full; LLM (and optional RAG); TTS. |
| **Post-Response** | BACK_TO_LISTENING; currently **listen_only = False** (enhancement: set back to True to “continue listening”). |

This is the full flow of the conversation AI and how continuous listening with wake word “EchoMind” works today, plus where it already matches your spec and where small changes would align it fully.
