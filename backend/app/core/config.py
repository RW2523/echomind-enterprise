from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    APP_NAME: str = "EchoMind Backend"
    DATA_DIR: str = "/data"
    DB_PATH: str = "/data/echomind.sqlite"
    FAISS_PATH: str = "/data/faiss.index"
    META_PATH: str = "/data/faiss_meta.json"
    LLM_BASE_URL: str = "http://ollama:11434/v1"
    LLM_MODEL: str = "qwen2.5:7b-instruct"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 512
    OLLAMA_EMBED_URL: str = "http://ollama:11434/api/embeddings"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    CHUNK_SIZE: int = 900
    CHUNK_OVERLAP: int = 140
    TOP_K: int = 8
    WHISPER_MODEL: str = "base"
settings = Settings()
