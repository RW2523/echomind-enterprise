# EchoMind — Voice AI + Conversation Intelligence

✅ Context/Role box (System prompt)  
✅ Session conversation memory (last ~12 turns, token budget)  
✅ **EchoMind layer**: rolling conversation memory, wake word, voice commands, fact-check, memory queries  
✅ Listen-only mode + trigger phrases / wake word to respond  
✅ Barge-in cancel + smooth fade

## Run
```bash
bash scripts/download_voice.sh
docker build -t echomind-voice .

docker run --rm -it --gpus all --network host \
  -v $PWD/voices:/voices \
  -e LLM_URL=http://127.0.0.1:11434/v1/chat/completions \
  -e LLM_MODEL=qwen2.5:7b-instruct-q4_K_M \
  -e PIPER_MODEL=/voices/en_US-lessac-medium.onnx \
  echomind-voice
```

Open: `http://<host>:8000`

## EchoMind Conversation Intelligence

### Features
- **Passive listening**: Say *"listen to conversation"* or *"start listening"* — the bot transcribes and accumulates context in a rolling window (configurable, default 30 min). Say the wake word or *"now you can speak"* / *"process that"* / *"fact check it"* to exit and respond.
- **Wake word**: Default *"EchoMind"*. Say *"EchoMind, what did I say in the last minute?"*. Change it by voice: *"Your name is Watson"* → wake word becomes *"Watson"*.
- **Profile (voice or UI)**: Assistant name, user name, timezone, location. Persisted for the session; set via voice or `set_context` (client).
- **Memory queries**: *"What did I say in the last 5 minutes?"*, *"Summarize last 10 minutes"*, *"When did we mention X?"*, *"Give timestamps and tags"*, *"Who said what"*.
- **Fact-check**: *"Fact check that"* / *"Fact check it"* — uses recent conversation context and optional backend RAG to fact-check claims.
- **Start/Stop**: *"start listening"*, *"stop listening"*, *"pause"*, *"resume"*, *"clear memory"*.

### Key phrases (quick reference)
| Intent | Example phrases |
|--------|------------------|
| Start listening | "listen to conversation", "start listening", "just listen" |
| Stop listening | "stop listening", "pause listening", "pause" |
| Resume | "resume", "resume listening" |
| Trigger (respond) | "EchoMind" (wake word), "now you can speak", "process that", "fact check it" |
| Set assistant name | "your name is X", "call yourself X", "change wake word to X" |
| Set user name | "my name is X", "call me X" |
| Set location | "I'm in New York", "set location to London" |
| Set timezone | "set timezone to Europe/London" |
| Memory recap | "what did I say in the last 5 minutes" |
| Summarize | "summarize last 10 minutes" |
| Timestamps | "give timestamps and tags", "who said what" |
| Fact-check | "fact check", "fact check that" |
| Clear memory | "clear memory", "forget everything" |

### LLM (TensorRT-LLM / OpenAI-compatible)
- **docker-compose** sets `LLM_URL` to `http://host.docker.internal:8355/v1/chat/completions` and `LLM_MODEL=nvidia/Llama-3.1-8B-Instruct-FP4` (same stack as backend chat).
- Main replies use **`stream: true`** via `OpenAICompatLLMStream.stream_messages` for low time-to-first-token; summaries/tool paths use non-streaming `complete_messages`.
- **TTS pipeline:** Tokens are read in the main coroutine while a **serial phrase queue** feeds Piper. Phrases commit on **max length**, **clause ends** (comma/semicolon/colon after `PHRASE_CLAUSE_MIN_CHARS`), **sentence ends** (`.?!` with `FIRST_SENTENCE_MIN_CHARS` for the first phrase), or **pause flush** (`PHRASE_MIN_CHARS` + `PHRASE_COMMIT_PAUSE_MS`). Defaults favor **earlier first audio** over fewer Piper calls. Piper **`synth` runs in a thread pool** so the LLM stream keeps draining while audio plays.
- **Logs** (container stdout): `VOICE_LLM stream start … stream=true` / `VOICE_LLM stream done … ttft_ms=… stream_total_ms=…` at INFO. Set `LLM_LOG_PAYLOAD=1` for full request JSON at WARNING.
- **Backend RAG** (`BACKEND_CHAT_URL`): document / transcript / FMR-style questions use **`POST /api/chat/ask-voice-stream`** (NDJSON, same streaming pipeline as web chat) so speech starts as soon as the backend LLM emits tokens—**not** after the full RAG answer is buffered. Falls back to `ask-voice` if the stream fails. Payload includes **`voice_max_tokens`** (default **`VOICE_RAG_MAX_TOKENS=640`**) so the backend uses **`stream: true`** with a bounded completion length.

### Config (env)
- `FIRST_SENTENCE_MIN_CHARS` — min chars before committing the **first** phrase on `.?!` (default 8); lower = earlier TTS start after first sentence.
- `PHRASE_CLAUSE_MIN_CHARS` — min chars before committing on `,` / `;` / `:` (default 18); enables streaming TTS before a full sentence.
- `PHRASE_MIN_CHARS`, `PHRASE_MAX_CHARS`, `PHRASE_COMMIT_PAUSE_MS` — pause-based flush and caps (defaults 18 / 96 / 120 ms; pause timer updates only on phrase commit).
- `MEMORY_WINDOW_MINUTES` — rolling window for conversation memory (default 30).
- `DEFAULT_ASSISTANT_NAME` — wake word / assistant name (default EchoMind).
- `DEFAULT_USER_NAME`, `DEFAULT_TIMEZONE`, `DEFAULT_LOCATION` — session defaults.
- `ECHO_DEBUG=1` — log when listen_only toggles, when memory entries are added, and when profile updates (server logs).
- `VOICE_RAG_MAX_TOKENS` — passed to `/api/chat/ask-voice-stream` as `voice_max_tokens` (default 640); lower = shorter spoken answers and less GPU time.

### New server→client messages
- `profile_update` — `{ assistant_name, wake_word, user_name, timezone, location }`.
- `memory_event` — `{ event: "listening_mode_on" | "listening_mode_off" }`.
- `memory_info` — optional, for recap/summary/timestamps queries (summary or entries).

## Notes
- Memory is per WebSocket session.
- LLM history: last ~12 turns, token budget.
- EchoMind conversation memory: rolling buffer by time; can be swapped for persistence later.
- Barge-in: sustained speech cancels assistant pipeline and continues capturing your utterance.

## v5.3 interrupt fix
- Barge-in cancels assistant output and sends `cancel`; client clears queue and smooth-stops playback.
