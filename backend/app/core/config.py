from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    APP_NAME: str = "EchoMind Backend"
    DATA_DIR: str = os.getenv("ECHOMIND_DATA_DIR", "/data")
    DB_PATH: str = "/data/echomind.sqlite"
    FAISS_PATH: str = "/data/faiss.index"
    META_PATH: str = "/data/faiss_meta.json"
    SPARSE_META_PATH: str = os.getenv("ECHOMIND_SPARSE_META_PATH", "/data/sparse_meta.json")
    LLM_BASE_URL: str = "http://ollama:11434/v1"
    LLM_MODEL: str = "qwen2.5:7b-instruct"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 512
    OLLAMA_EMBED_URL: str = "http://ollama:11434/api/embeddings"
    OLLAMA_EMBED_MODEL: str = os.getenv("ECHOMIND_EMBED_MODEL", "nomic-embed-text")
    # Max characters per chunk sent to embedding API (avoids "input length exceeds context length").
    # Conservative default (2000) works with 512-token models; set ECHOMIND_EMBED_MAX_CHARS=8000 for nomic-embed-text.
    EMBED_MAX_CHARS: int = int(os.getenv("ECHOMIND_EMBED_MAX_CHARS", "2000"))
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 140
    TOP_K: int = 8
    RAG_RELEVANCE_THRESHOLD: float = float(os.getenv("ECHOMIND_RAG_RELEVANCE_THRESHOLD", "0.45"))
    WHISPER_MODEL: str = "base"

    # Real-time transcription & knowledge capture
    ECHOMIND_AUTO_STORE_DEFAULT: bool = os.getenv("ECHOMIND_AUTO_STORE_DEFAULT", "0").lower() in ("1", "true", "yes")
    TRANSCRIPT_SILENCE_COMMIT_MS: int = int(os.getenv("TRANSCRIPT_SILENCE_COMMIT_MS", "800"))
    TRANSCRIPT_PARAGRAPH_SILENCE_MS: int = int(os.getenv("TRANSCRIPT_PARAGRAPH_SILENCE_MS", "2000"))
    TRANSCRIPT_MAX_PARAGRAPH_CHARS: int = int(os.getenv("TRANSCRIPT_MAX_PARAGRAPH_CHARS", "700"))
    TRANSCRIPT_RECENT_BUFFER_MAX_CHARS: int = int(os.getenv("TRANSCRIPT_RECENT_BUFFER_MAX_CHARS", "120"))
    TRANSCRIPT_OVERLAP_K: int = int(os.getenv("TRANSCRIPT_OVERLAP_K", "80"))
    TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC: float = float(os.getenv("TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC", "15"))
    SAMPLE_RATE: int = 16000

settings = Settings()
