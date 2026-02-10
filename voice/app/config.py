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

    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

    # OpenAI-compatible endpoint (Ollama: http://127.0.0.1:11434/v1/chat/completions)
    LLM_URL: str = os.getenv("LLM_URL", "http://127.0.0.1:11434/v1/chat/completions")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")

    # LLM streaming / phrase commit knobs (unmute-like)
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "220"))

    PHRASE_MIN_CHARS: int = int(os.getenv("PHRASE_MIN_CHARS", "28"))
    PHRASE_MAX_CHARS: int = int(os.getenv("PHRASE_MAX_CHARS", "120"))
    PHRASE_COMMIT_PAUSE_MS: int = int(os.getenv("PHRASE_COMMIT_PAUSE_MS", "180"))

    # Piper TTS (model path; voices dir for download is VOICES_DIR, default /voices)
    PIPER_MODEL: str = os.getenv("PIPER_MODEL", "/voices/en_US-lessac-medium.onnx")
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
    INTRO_PHRASE: str = os.getenv("INTRO_PHRASE", "Hi! I'm here. What would you like to talk about?")

    # Backend RAG: when set, voice can call this URL for knowledge-base answers (use_knowledge_base=true).
    # Example: http://backend:8000 (no trailing slash; /api/chat/ask-voice is appended).
    BACKEND_CHAT_URL: str = os.getenv("BACKEND_CHAT_URL", "")

SETTINGS = Settings()
