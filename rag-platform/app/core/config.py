"""RAG platform configuration. Tune constants here."""
from __future__ import annotations
import os
from pydantic_settings import BaseSettings

_DEFAULT_DATA_DIR = os.getenv("ECHOMIND_DATA_DIR", "/data")


def _default_device() -> str:
    """Use CUDA only if PyTorch was built with CUDA; else CPU so the app still starts."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Settings(BaseSettings):
    APP_NAME: str = "RAG Platform"
    DATA_DIR: str = _DEFAULT_DATA_DIR
    DB_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "rag_platform.sqlite")

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION_DOCUMENTS: str = "documents"
    QDRANT_COLLECTION_TRANSCRIPTS: str = "transcripts"

    # HuggingFace models (GPU, loaded at startup)
    EMBED_MODEL_ID: str = os.getenv("EMBED_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")
    GENERATOR_MODEL_ID: str = os.getenv("GENERATOR_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct")
    DEVICE: str = os.getenv("DEVICE", None) or _default_device()
    USE_BF16: bool = os.getenv("USE_BF16", "1").lower() in ("1", "true", "yes")
    LOW_CPU_MEM: bool = True
    USE_SAFETENSORS: bool = True

    # Chunking (documents)
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    TRANSCRIPT_CHUNK_SEC: int = 60

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "15"))
    T1_TRANSCRIPT: float = float(os.getenv("T1_TRANSCRIPT", "0.30"))
    T2_DOCUMENT: float = float(os.getenv("T2_DOCUMENT", "0.30"))
    T_LOW: float = float(os.getenv("T_LOW", "0.25"))
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))

    # Catalog (Postgres optional)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")  # If set, use Postgres; else SQLite


settings = Settings()
