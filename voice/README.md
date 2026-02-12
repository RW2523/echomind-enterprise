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
  -e LLM_MODEL=qwen2.5:7b-instruct \
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

### Config (env)
- `MEMORY_WINDOW_MINUTES` — rolling window for conversation memory (default 30).
- `DEFAULT_ASSISTANT_NAME` — wake word / assistant name (default EchoMind).
- `DEFAULT_USER_NAME`, `DEFAULT_TIMEZONE`, `DEFAULT_LOCATION` — session defaults.
- `ECHO_DEBUG=1` — log when listen_only toggles, when memory entries are added, and when profile updates (server logs).

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
