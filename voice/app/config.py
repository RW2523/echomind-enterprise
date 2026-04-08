import os
from dataclasses import dataclass

@dataclass
class Settings:
    SR: int = int(os.getenv("SR", "16000"))
    FRAME_MS: int = int(os.getenv("FRAME_MS", "20"))

    VAD_AGGR: int = int(os.getenv("VAD_AGGR", "2"))
    ENDPOINT_SILENCE_MS: int = int(os.getenv("ENDPOINT_SILENCE_MS", "450"))
    MIN_SPEECH_MS: int = int(os.getenv("MIN_SPEECH_MS", "250"))
    END_TAIL_MS: int = int(os.getenv("END_TAIL_MS", "120"))
    # Barge-in: require this many consecutive speech frames before treating as user speech (reduces false triggers from noise)
    BARGE_IN_SPEECH_LEAD_IDLE: int = int(os.getenv("BARGE_IN_SPEECH_LEAD_IDLE", "2"))   # when assistant idle
    BARGE_IN_SPEECH_LEAD_ACTIVE: int = int(os.getenv("BARGE_IN_SPEECH_LEAD_ACTIVE", "6"))  # when assistant speaking (stricter)

    # Nemotron streaming ASR (same model/env as backend live transcript)
    NEMOTRON_MODEL_NAME: str = os.getenv(
        "ECHOMIND_ASR_MODEL_NAME",
        os.getenv("NEMOTRON_MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b"),
    )
    NEMOTRON_ATT_CONTEXT_RIGHT: int = int(os.getenv("ECHOMIND_ASR_ATT_CONTEXT_RIGHT", "6"))
    # Chunk size for utterance-final decode (ms); align with backend TRANSCRIPT_NEMOTRON_CHUNK_MS default
    NEMOTRON_CHUNK_MS: int = int(os.getenv("VOICE_NEMOTRON_CHUNK_MS", os.getenv("TRANSCRIPT_NEMOTRON_CHUNK_MS", "560")))

    # OpenAI-compatible chat/completions (full URL including /v1/chat/completions).
    # docker-compose sets TRT-LLM on host: LLM_URL=http://host.docker.internal:8355/v1/chat/completions
    #   and LLM_MODEL=nvidia/Llama-3.1-8B-Instruct-FP4. Defaults below suit local Ollama without Compose.
    LLM_URL: str = os.getenv("LLM_URL", "http://127.0.0.1:11434/v1/chat/completions")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")

    # LLM streaming / phrase commit knobs. Main dialogue uses stream=true via OpenAICompatLLMStream.stream_messages.
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "220"))
    # Log full LLM JSON body (set LLM_LOG_PAYLOAD=1). Timing lines always log at INFO (VOICE_LLM stream start/done).
    LLM_LOG_PAYLOAD: bool = os.getenv("LLM_LOG_PAYLOAD", "0").lower() in ("1", "true", "yes")

    PHRASE_MIN_CHARS: int = int(os.getenv("PHRASE_MIN_CHARS", "28"))
    PHRASE_MAX_CHARS: int = int(os.getenv("PHRASE_MAX_CHARS", "120"))
    PHRASE_COMMIT_PAUSE_MS: int = int(os.getenv("PHRASE_COMMIT_PAUSE_MS", "180"))
    # First spoken phrase: commit on sentence end with this lower minimum so TTS starts right after the first sentence.
    FIRST_SENTENCE_MIN_CHARS: int = int(os.getenv("FIRST_SENTENCE_MIN_CHARS", "8"))

    # Piper TTS (model path; voices dir for download is VOICES_DIR, default /voices)
    PIPER_MODEL: str = os.getenv("PIPER_MODEL", "/voices/en_US-lessac-medium.onnx")
    # ONNX Runtime: use CUDAExecutionProvider when True (image must have onnxruntime-gpu; see voice/Dockerfile).
    PIPER_USE_CUDA: bool = os.getenv("VOICE_PIPER_USE_CUDA", "0").lower() in ("1", "true", "yes")
    PIPER_SPEAKER: int = int(os.getenv("PIPER_SPEAKER", "0"))
    PIPER_NOISE_SCALE: float = float(os.getenv("PIPER_NOISE_SCALE", "0.667"))
    PIPER_LENGTH_SCALE: float = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))

    # Emotion mode (client playback-rate + tiny fillers)
    EMOTION_MODE: bool = os.getenv("EMOTION_MODE", "1") == "1"

    # Moshi optional
    MOSHI_URL: str = os.getenv("MOSHI_URL", "ws://127.0.0.1:8080/ws")
    USE_MOSHI_CORE: bool = os.getenv("USE_MOSHI_CORE", "0") == "1"
    MOSHI_SUPPORTS_TEXT_INJECT: bool = os.getenv("MOSHI_SUPPORTS_TEXT_INJECT", "0") == "1"

    # Greeting spoken by TTS when session starts (natural conversation opener)
    INTRO_PHRASE: str = os.getenv("INTRO_PHRASE", "Hi! I'm EchoMind, your AI assistant. How can I help you today?")

    # Backend RAG: when set, voice calls ask-voice-stream (NDJSON chunks → phrase TTS) and falls back to ask-voice.
    # Example: http://backend:8000 (no trailing slash).
    BACKEND_CHAT_URL: str = os.getenv("BACKEND_CHAT_URL", "")

    # EchoMind Conversation Intelligence
    MEMORY_WINDOW_MINUTES: float = float(os.getenv("MEMORY_WINDOW_MINUTES", "30"))
    DEFAULT_ASSISTANT_NAME: str = os.getenv("DEFAULT_ASSISTANT_NAME", "EchoMind")
    DEFAULT_USER_NAME: str = os.getenv("DEFAULT_USER_NAME", "")
    DEFAULT_TIMEZONE: str = os.getenv("DEFAULT_TIMEZONE", "America/New_York")
    DEFAULT_LOCATION: str = os.getenv("DEFAULT_LOCATION", "")
    ECHO_DEBUG: bool = os.getenv("ECHO_DEBUG", "0") == "1"

SETTINGS = Settings()
