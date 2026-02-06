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
    # When False (default), do not expose citations/filenames to client (audit: internal grounding only).
    RAG_EXPOSE_SOURCES: bool = os.getenv("ECHOMIND_RAG_EXPOSE_SOURCES", "0").lower() in ("1", "true", "yes")

    # --- RAG quality improvements (all optional, no breaking changes) ---
    # Intent-aware query rewriting: classify (factual/procedural/exploratory/temporal) and rewrite for precision.
    RAG_INTENT_REWRITE: bool = os.getenv("ECHOMIND_RAG_INTENT_REWRITE", "1").lower() in ("1", "true", "yes")
    # Weighted RRF: dense_weight + sparse_weight (default 0.6 + 0.4) instead of equal. Improves recall/precision balance.
    RAG_DENSE_RRF_WEIGHT: float = float(os.getenv("ECHOMIND_RAG_DENSE_RRF_WEIGHT", "0.6"))
    RAG_SPARSE_RRF_WEIGHT: float = float(os.getenv("ECHOMIND_RAG_SPARSE_RRF_WEIGHT", "0.4"))
    # Time-decay scoring: multiply score by exp(-age_days/halflife). 0 = off (use hard filter only).
    RAG_TIME_DECAY_HALFLIFE_DAYS: float = float(os.getenv("ECHOMIND_RAG_TIME_DECAY_HALFLIFE_DAYS", "14"))
    # Boost chunks whose transcript tags overlap query terms (when doc is transcript). Small additive boost.
    RAG_TAG_BOOST_ENABLED: bool = os.getenv("ECHOMIND_RAG_TAG_BOOST", "1").lower() in ("1", "true", "yes")
    RAG_TAG_BOOST_FACTOR: float = float(os.getenv("ECHOMIND_RAG_TAG_BOOST_FACTOR", "0.08"))
    # Optional LLM rerank: score top RAG_RERANK_CANDIDATES and reorder. 0 = disabled (saves latency).
    RAG_RERANK_ENABLED: bool = os.getenv("ECHOMIND_RAG_RERANK", "0").lower() in ("1", "true", "yes")
    RAG_RERANK_CANDIDATES: int = int(os.getenv("ECHOMIND_RAG_RERANK_CANDIDATES", "12"))
    RAG_RERANK_TOP_N: int = int(os.getenv("ECHOMIND_RAG_RERANK_TOP_N", "8"))
    # Prefer authoritative documents (PDF/DOCX/PPTX) over transcripts when scores are close (tie-break).
    RAG_PREFER_AUTHORITATIVE: bool = os.getenv("ECHOMIND_RAG_PREFER_AUTHORITATIVE", "1").lower() in ("1", "true", "yes")
    # Max chars for parent chunk expansion; lower reduces context domination (default 1600).
    RAG_PARENT_CONTEXT_MAX_CHARS: int = int(os.getenv("ECHOMIND_RAG_PARENT_CONTEXT_MAX_CHARS", "1600"))
    # Deduplicate overlapping sentences in context (simple overlap threshold). 0 = off.
    RAG_DEDUPE_SENTENCES: bool = os.getenv("ECHOMIND_RAG_DEDUPE_SENTENCES", "1").lower() in ("1", "true", "yes")
    RAG_DEDUPE_OVERLAP_RATIO: float = float(os.getenv("ECHOMIND_RAG_DEDUPE_OVERLAP_RATIO", "0.6"))
    # Book/long-form retrieval: higher recall for TOC and concept queries (e.g. "Matthew Effect" in books).
    RAG_BOOK_K_PER_QUERY: int = int(os.getenv("ECHOMIND_RAG_BOOK_K_PER_QUERY", "20"))
    RAG_BOOK_SPARSE_WEIGHT: float = float(os.getenv("ECHOMIND_RAG_BOOK_SPARSE_WEIGHT", "0.5"))
    # TOC/chapters guardrail: when user asks for chapters/contents, require TOC signals in context or refuse.
    RAG_TOC_GUARDRAIL: bool = os.getenv("ECHOMIND_RAG_TOC_GUARDRAIL", "1").lower() in ("1", "true", "yes")
    # Bypass compression for chunks that contain key query terms (improves grounding for named concepts).
    RAG_VERBATIM_QUERY_TERMS: bool = os.getenv("ECHOMIND_RAG_VERBATIM_QUERY_TERMS", "1").lower() in ("1", "true", "yes")
    RAG_VERBATIM_MAX_CHARS: int = int(os.getenv("ECHOMIND_RAG_VERBATIM_MAX_CHARS", "1200"))
    # Generic RAG reliability: intent-based retrieval profiles, refusal when no evidence, structure fallback.
    RAG_USE_INTENT_PROFILES: bool = os.getenv("ECHOMIND_RAG_USE_INTENT_PROFILES", "1").lower() in ("1", "true", "yes")
    RAG_REFUSAL_ON_NO_EVIDENCE: bool = os.getenv("ECHOMIND_RAG_REFUSAL_ON_NO_EVIDENCE", "1").lower() in ("1", "true", "yes")
    RAG_STRUCTURE_FALLBACK: bool = os.getenv("ECHOMIND_RAG_STRUCTURE_FALLBACK", "1").lower() in ("1", "true", "yes")
    RAG_NORMALIZE_STRIP_HEADERS_FOOTERS: bool = os.getenv("ECHOMIND_RAG_NORMALIZE_STRIP_HEADERS_FOOTERS", "0").lower() in ("1", "true", "yes")
    RAG_SKIP_COMPRESS_TABLES_HEADINGS: bool = os.getenv("ECHOMIND_RAG_SKIP_COMPRESS_TABLES_HEADINGS", "1").lower() in ("1", "true", "yes")
    RAG_ADJACENT_CHUNK_EXPANSION: bool = os.getenv("ECHOMIND_RAG_ADJACENT_CHUNK_EXPANSION", "1").lower() in ("1", "true", "yes")
    RAG_EVIDENCE_POSTCHECK: bool = os.getenv("ECHOMIND_RAG_EVIDENCE_POSTCHECK", "1").lower() in ("1", "true", "yes")

    WHISPER_MODEL: str = "base"

    # Real-time transcription & knowledge capture
    ECHOMIND_AUTO_STORE_DEFAULT: bool = os.getenv("ECHOMIND_AUTO_STORE_DEFAULT", "0").lower() in ("1", "true", "yes")
    TRANSCRIPT_SILENCE_COMMIT_MS: int = int(os.getenv("TRANSCRIPT_SILENCE_COMMIT_MS", "800"))
    TRANSCRIPT_PARAGRAPH_SILENCE_MS: int = int(os.getenv("TRANSCRIPT_PARAGRAPH_SILENCE_MS", "2000"))
    TRANSCRIPT_MAX_PARAGRAPH_CHARS: int = int(os.getenv("TRANSCRIPT_MAX_PARAGRAPH_CHARS", "700"))
    TRANSCRIPT_RECENT_BUFFER_MAX_CHARS: int = int(os.getenv("TRANSCRIPT_RECENT_BUFFER_MAX_CHARS", "120"))
    TRANSCRIPT_OVERLAP_K: int = int(os.getenv("TRANSCRIPT_OVERLAP_K", "200"))
    TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC: float = float(os.getenv("TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC", "15"))
    SAMPLE_RATE: int = 16000

settings = Settings()
