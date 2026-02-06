from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    APP_NAME: str = "EchoMind Backend"
    DATA_DIR: str = os.getenv("ECHOMIND_DATA_DIR", "/data")
    DB_PATH: str = os.path.join(os.getenv("ECHOMIND_DATA_DIR", "/data"), "echomind.sqlite")
    FAISS_PATH: str = os.path.join(os.getenv("ECHOMIND_DATA_DIR", "/data"), "faiss.index")
    META_PATH: str = os.path.join(os.getenv("ECHOMIND_DATA_DIR", "/data"), "faiss_meta.json")
    LLM_BASE_URL: str = "http://ollama:11434/v1"
    LLM_MODEL: str = "qwen2.5:7b-instruct"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 512
    OLLAMA_EMBED_URL: str = "http://ollama:11434/api/embeddings"
    OLLAMA_EMBED_MODEL: str = os.getenv("ECHOMIND_EMBED_MODEL", "nomic-embed-text")
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 140
    TOP_K: int = 8
    WHISPER_MODEL: str = "base"

    # Real-time transcript stabilization & segmentation
    TRANSCRIPT_SILENCE_COMMIT_MS: int = int(os.getenv("TRANSCRIPT_SILENCE_COMMIT_MS", "900"))
    TRANSCRIPT_PARAGRAPH_SILENCE_MS: int = int(os.getenv("TRANSCRIPT_PARAGRAPH_SILENCE_MS", "2000"))
    TRANSCRIPT_MAX_BUFFER_CHARS: int = int(os.getenv("TRANSCRIPT_MAX_BUFFER_CHARS", "120"))
    TRANSCRIPT_MAX_PARAGRAPH_CHARS: int = int(os.getenv("TRANSCRIPT_MAX_PARAGRAPH_CHARS", "700"))
    TRANSCRIPT_OVERLAP_GUARD_CHARS: int = int(os.getenv("TRANSCRIPT_OVERLAP_GUARD_CHARS", "80"))
    TRANSCRIPT_PARTIAL_RATE_LIMIT_PER_SEC: float = float(os.getenv("TRANSCRIPT_PARTIAL_RATE_LIMIT_PER_SEC", "15"))
    ECHOMIND_AUTO_STORE_DEFAULT: bool = os.getenv("ECHOMIND_AUTO_STORE_DEFAULT", "0").lower() in ("1", "true", "yes")

settings = Settings()
