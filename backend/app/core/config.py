from pydantic_settings import BaseSettings
import os

# Base dir for all persistence; env ECHOMIND_DATA_DIR overrides. Used so DB_PATH, FAISS_*, etc. stay under one root.
_DEFAULT_DATA_DIR = os.getenv("ECHOMIND_DATA_DIR", "/data")


class Settings(BaseSettings):
    APP_NAME: str = "EchoMind Backend"
    DATA_DIR: str = _DEFAULT_DATA_DIR
    DB_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "echomind.sqlite")
    FAISS_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "faiss.index")
    META_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "faiss_meta.json")
    SPARSE_META_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "sparse_meta.json")
    # Transcript-only index: used when intent=transcript so retrieval runs only over transcripts.
    FAISS_TRANSCRIPT_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "faiss_transcript.index")
    META_TRANSCRIPT_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "faiss_transcript_meta.json")
    SPARSE_TRANSCRIPT_META_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "sparse_transcript_meta.json")
    LLM_BASE_URL: str = "http://ollama:11434/v1"
    LLM_MODEL: str = "qwen2.5:7b-instruct"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 512
    OLLAMA_EMBED_URL: str = "http://ollama:11434/api/embeddings"
    OLLAMA_EMBED_MODEL: str = os.getenv("ECHOMIND_EMBED_MODEL", "nomic-embed-text")
    # Max characters per chunk sent to embedding API (avoids "input length exceeds context length").
    # Conservative default (2000) works with 512-token models; set ECHOMIND_EMBED_MAX_CHARS=8000 for nomic-embed-text.
    EMBED_MAX_CHARS: int = int(os.getenv("ECHOMIND_EMBED_MAX_CHARS", "2000"))
    CHUNK_SIZE: int = int(os.getenv("ECHOMIND_CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("ECHOMIND_CHUNK_OVERLAP", "120"))
    TOP_K: int = int(os.getenv("ECHOMIND_TOP_K", "20"))
    RAG_RELEVANCE_THRESHOLD: float = float(os.getenv("ECHOMIND_RAG_RELEVANCE_THRESHOLD", "0.45"))
    # When False (default), do not expose citations/filenames to client (audit: internal grounding only).
    RAG_EXPOSE_SOURCES: bool = os.getenv("ECHOMIND_RAG_EXPOSE_SOURCES", "0").lower() in ("1", "true", "yes")

    # --- RAG quality improvements (all optional, no breaking changes) ---
    # When False (default), skip LLM query rewrite and use only the user question + deterministic variants for search. Set ECHOMIND_RAG_QUERY_REWRITE=1 for intent-aware expansion (adds 1 LLM call).
    RAG_QUERY_REWRITE_ENABLED: bool = os.getenv("ECHOMIND_RAG_QUERY_REWRITE", "0").lower() in ("1", "true", "yes")
    # When query rewrite is enabled: use intent-based rewrite (document/transcript) vs generic. Set ECHOMIND_RAG_INTENT_REWRITE=0 to use generic rewrite only.
    RAG_INTENT_REWRITE: bool = os.getenv("ECHOMIND_RAG_INTENT_REWRITE", "1").lower() in ("1", "true", "yes")
    # Weighted RRF: dense_weight + sparse_weight (default 0.6 + 0.4) instead of equal. Improves recall/precision balance.
    RAG_DENSE_RRF_WEIGHT: float = float(os.getenv("ECHOMIND_RAG_DENSE_RRF_WEIGHT", "0.6"))
    RAG_SPARSE_RRF_WEIGHT: float = float(os.getenv("ECHOMIND_RAG_SPARSE_RRF_WEIGHT", "0.4"))
    # Time-decay scoring: multiply score by exp(-age_days/halflife). 0 = off (use hard filter only).
    RAG_TIME_DECAY_HALFLIFE_DAYS: float = float(os.getenv("ECHOMIND_RAG_TIME_DECAY_HALFLIFE_DAYS", "14"))
    # Boost chunks whose transcript tags overlap query terms (when doc is transcript). Small additive boost.
    RAG_TAG_BOOST_ENABLED: bool = os.getenv("ECHOMIND_RAG_TAG_BOOST", "1").lower() in ("1", "true", "yes")
    RAG_TAG_BOOST_FACTOR: float = float(os.getenv("ECHOMIND_RAG_TAG_BOOST_FACTOR", "0.08"))
    # Optional LLM rerank: score top RAG_RERANK_CANDIDATES and reorder. Kept false by default for faster responses.
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
    # When True (default), compress each chunk with LLM before sending to answer; when False, use chunk text as-is (truncated only). Set ECHOMIND_RAG_COMPRESS_CONTEXT=0 to disable.
    RAG_COMPRESS_CONTEXT: bool = os.getenv("ECHOMIND_RAG_COMPRESS_CONTEXT", "1").lower() in ("1", "true", "yes")
    # Bypass compression for chunks that contain key query terms (improves grounding for named concepts).
    RAG_VERBATIM_QUERY_TERMS: bool = os.getenv("ECHOMIND_RAG_VERBATIM_QUERY_TERMS", "1").lower() in ("1", "true", "yes")
    RAG_VERBATIM_MAX_CHARS: int = int(os.getenv("ECHOMIND_RAG_VERBATIM_MAX_CHARS", "1200"))
    # When True (default), if retrieval finds nothing or low relevance, answer conversationally instead of a fixed "no info" message. Set ECHOMIND_RAG_INSUFFICIENT_FALLBACK_TO_GENERAL=0 for strict behavior.
    RAG_INSUFFICIENT_FALLBACK_TO_GENERAL: bool = os.getenv("ECHOMIND_RAG_INSUFFICIENT_FALLBACK_TO_GENERAL", "1").lower() in ("1", "true", "yes")
    # When True (default), use RAG for any non-greeting question so the bot answers from the knowledge base. When False, use RAG only when the user mentions document/transcript/book/file etc.
    RAG_ALWAYS_TRY_FOR_CONTENT_QUESTIONS: bool = os.getenv("ECHOMIND_RAG_ALWAYS_TRY_FOR_CONTENT_QUESTIONS", "1").lower() in ("1", "true", "yes")
    # Temperature for RAG answer generation (default 0.15 for more accurate, grounded replies). Uses LLM_TEMPERATURE if not set.
    RAG_LLM_TEMPERATURE: float = float(os.getenv("ECHOMIND_RAG_LLM_TEMPERATURE", "0.15"))

    # Real-time transcription & knowledge capture (Kyutai STT only, 24kHz)
    ECHOMIND_AUTO_STORE_DEFAULT: bool = os.getenv("ECHOMIND_AUTO_STORE_DEFAULT", "1").lower() in ("1", "true", "yes")
    # When auto_store is on: store new transcript content to the KB every N seconds (0 = only on stop).
    AUTO_STORE_INTERVAL_SEC: int = int(os.getenv("ECHOMIND_AUTO_STORE_INTERVAL_SEC", "60"))
    TRANSCRIPT_SILENCE_COMMIT_MS: int = int(os.getenv("TRANSCRIPT_SILENCE_COMMIT_MS", "800"))
    TRANSCRIPT_PARAGRAPH_SILENCE_MS: int = int(os.getenv("TRANSCRIPT_PARAGRAPH_SILENCE_MS", "2000"))
    TRANSCRIPT_MAX_PARAGRAPH_CHARS: int = int(os.getenv("TRANSCRIPT_MAX_PARAGRAPH_CHARS", "700"))
    TRANSCRIPT_RECENT_BUFFER_MAX_CHARS: int = int(os.getenv("TRANSCRIPT_RECENT_BUFFER_MAX_CHARS", "120"))
    TRANSCRIPT_OVERLAP_K: int = int(os.getenv("TRANSCRIPT_OVERLAP_K", "200"))
    TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC: float = float(os.getenv("TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC", "15"))
    SAMPLE_RATE: int = 24000  # Kyutai STT (kyutai/stt-1b-en_fr)
    # Voice activity: skip feeding audio to STT when RMS below this (0 = disabled). Reduces noise/silence transcribed as text.
    TRANSCRIPT_VAD_RMS_THRESHOLD: float = float(os.getenv("TRANSCRIPT_VAD_RMS_THRESHOLD", "0.008"))
    # VAD sliding window: window size and step in samples (e.g. 1024/512). Only used when VAD threshold > 0; chunk passes if any window exceeds threshold.
    TRANSCRIPT_VAD_WINDOW_SAMPLES: int = int(os.getenv("TRANSCRIPT_VAD_WINDOW_SAMPLES", "1024"))
    TRANSCRIPT_VAD_STEP_SAMPLES: int = int(os.getenv("TRANSCRIPT_VAD_STEP_SAMPLES", "512"))
    # Backpressure: max PCM frames in queue per session; excess dropped (0 = unbounded, not recommended).
    TRANSCRIPT_PCM_QUEUE_MAX_SIZE: int = int(os.getenv("TRANSCRIPT_PCM_QUEUE_MAX_SIZE", "256"))
    # Max interval_buffer entries per session (prevents memory leak).
    TRANSCRIPT_INTERVAL_BUFFER_MAX: int = int(os.getenv("TRANSCRIPT_INTERVAL_BUFFER_MAX", "2048"))
    # GPU concurrency: max concurrent STT inference when device is CUDA (1 = serial).
    TRANSCRIPT_GPU_CONCURRENCY: int = int(os.getenv("TRANSCRIPT_GPU_CONCURRENCY", "2"))
    # STT warmup: number of frames to run on model load (CUDA kernels).
    TRANSCRIPT_STT_WARMUP_FRAMES: int = int(os.getenv("TRANSCRIPT_STT_WARMUP_FRAMES", "8"))
    # Timezone for interpreting transcript time in questions (e.g. "at 2pm" → 2pm in this zone, then convert to UTC). Use "UTC" or e.g. "America/New_York".
    TRANSCRIPT_QUERY_TZ: str = os.getenv("ECHOMIND_TRANSCRIPT_QUERY_TZ", "UTC")

    # --- RAG platform: Qdrant (optional) and vector backend ---
    # When set, use Qdrant for vectors; else use existing FAISS+BM25.
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    VECTOR_BACKEND: str = os.getenv("ECHOMIND_VECTOR_BACKEND", "faiss")  # "faiss" | "qdrant"
    # Transcript chunk size in seconds for ingestion pipeline.
    TRANSCRIPT_CHUNK_SECONDS: int = int(os.getenv("ECHOMIND_TRANSCRIPT_CHUNK_SECONDS", "60"))

settings = Settings()
