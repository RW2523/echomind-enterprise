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
    LLM_MODEL: str = "qwen2.5:7b-instruct-q4_K_M"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 512
    OLLAMA_EMBED_URL: str = "http://ollama:11434/api/embeddings"
    OLLAMA_EMBED_MODEL: str = os.getenv("ECHOMIND_EMBED_MODEL", "nomic-embed-text")
    # Max characters per chunk sent to embedding API (avoids "input length exceeds context length").
    # Conservative default (2000) works with 512-token models; set ECHOMIND_EMBED_MAX_CHARS=8000 for nomic-embed-text.
    EMBED_MAX_CHARS: int = int(os.getenv("ECHOMIND_EMBED_MAX_CHARS", "2000"))
    CHUNK_SIZE: int = int(os.getenv("ECHOMIND_CHUNK_SIZE", "650"))
    CHUNK_OVERLAP: int = int(os.getenv("ECHOMIND_CHUNK_OVERLAP", "120"))
    TOP_K: int = int(os.getenv("ECHOMIND_TOP_K", "15"))
    # Large-doc parent/child token limits (BOOK type). Override for dense regulatory docs.
    BOOK_PARENT_MIN_TOKENS: int = int(os.getenv("ECHOMIND_BOOK_PARENT_MIN_TOKENS", "2000"))
    BOOK_PARENT_MAX_TOKENS: int = int(os.getenv("ECHOMIND_BOOK_PARENT_MAX_TOKENS", "3500"))
    BOOK_CHILD_MIN_TOKENS: int = int(os.getenv("ECHOMIND_BOOK_CHILD_MIN_TOKENS", "500"))
    BOOK_CHILD_MAX_TOKENS: int = int(os.getenv("ECHOMIND_BOOK_CHILD_MAX_TOKENS", "700"))
    # Dynamic chunk sizing: auto-shrink children for dense technical content.
    BOOK_DYNAMIC_CHUNK_SIZING: bool = os.getenv("ECHOMIND_BOOK_DYNAMIC_CHUNK_SIZING", "1").lower() in ("1", "true", "yes")
    # Large section threshold: split sections exceeding this (tokens) into multiple chunks. Default 10000.
    LARGE_SECTION_TOKEN_THRESHOLD: int = int(os.getenv("ECHOMIND_LARGE_SECTION_TOKEN_THRESHOLD", "10000"))
    RAG_RELEVANCE_THRESHOLD: float = float(os.getenv("ECHOMIND_RAG_RELEVANCE_THRESHOLD", "0.45"))
    # Expose citations (section, page, filename) to client for source-aware answers.
    RAG_EXPOSE_SOURCES: bool = os.getenv("ECHOMIND_RAG_EXPOSE_SOURCES", "1").lower() in ("1", "true", "yes")

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
    # Max chars for parent chunk expansion; higher for large regulatory docs (2400). Lower reduces context domination.
    RAG_PARENT_CONTEXT_MAX_CHARS: int = int(os.getenv("ECHOMIND_RAG_PARENT_CONTEXT_MAX_CHARS", "2400"))
    # Deduplicate overlapping sentences in context (simple overlap threshold). 0 = off.
    RAG_DEDUPE_SENTENCES: bool = os.getenv("ECHOMIND_RAG_DEDUPE_SENTENCES", "1").lower() in ("1", "true", "yes")
    RAG_DEDUPE_OVERLAP_RATIO: float = float(os.getenv("ECHOMIND_RAG_DEDUPE_OVERLAP_RATIO", "0.6"))
    RAG_DEDUPE_BY_SECTION: bool = os.getenv("RAG_DEDUPE_BY_SECTION", "1").lower() in ("1", "true", "yes")
    RAG_DEDUPE_SECTION_MAX_CHUNKS: int = int(os.getenv("RAG_DEDUPE_SECTION_MAX_CHUNKS", "2"))
    # Book/long-form retrieval: higher recall for TOC and concept queries (e.g. "Matthew Effect" in books).
    RAG_BOOK_K_PER_QUERY: int = int(os.getenv("ECHOMIND_RAG_BOOK_K_PER_QUERY", "20"))
    RAG_BOOK_SPARSE_WEIGHT: float = float(os.getenv("ECHOMIND_RAG_BOOK_SPARSE_WEIGHT", "0.5"))
    # TOC/chapters guardrail: when user asks for chapters/contents, require TOC signals in context or refuse.
    RAG_TOC_GUARDRAIL: bool = os.getenv("ECHOMIND_RAG_TOC_GUARDRAIL", "1").lower() in ("1", "true", "yes")
    # When True, compress each chunk with LLM; when False, use chunk text as-is (truncated only). Disabled by default for regulatory docs to preserve precision.
    RAG_COMPRESS_CONTEXT: bool = os.getenv("ECHOMIND_RAG_COMPRESS_CONTEXT", "0").lower() in ("1", "true", "yes")
    # Bypass compression for chunks that contain key query terms (improves grounding for named concepts).
    RAG_VERBATIM_QUERY_TERMS: bool = os.getenv("ECHOMIND_RAG_VERBATIM_QUERY_TERMS", "1").lower() in ("1", "true", "yes")
    RAG_VERBATIM_MAX_CHARS: int = int(os.getenv("ECHOMIND_RAG_VERBATIM_MAX_CHARS", "1600"))
    # Max total context chars; 0 = no limit. When exceeded, trim lowest-scoring blocks.
    RAG_CONTEXT_MAX_CHARS: int = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "0"))

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
    # WebSocket receive timeout (seconds). No message for this long = stale connection. Default 24h = effectively indefinite with heartbeat.
    TRANSCRIPT_WS_RECEIVE_TIMEOUT_SEC: float = float(os.getenv("TRANSCRIPT_WS_RECEIVE_TIMEOUT_SEC", "86400"))

    # ── Hierarchical / Graph-aware RAG feature toggles ──────────────────────────
    # Step 1: Section-level FAISS index restricts child search to top-N sections first.
    RAG_USE_SECTION_RETRIEVAL: bool = os.getenv("RAG_USE_SECTION_RETRIEVAL", "1").lower() in ("1", "true", "yes")
    RAG_SECTION_SCORE_THRESHOLD: float = float(os.getenv("RAG_SECTION_SCORE_THRESHOLD", "0.30"))
    RAG_SECTION_TOP_K: int = int(os.getenv("RAG_SECTION_TOP_K", "5"))

    # Step 2: Cross-encoder re-ranker (sentence-transformers if available, else LLM fallback).
    RAG_USE_RERANKER: bool = os.getenv("RAG_USE_RERANKER", "1").lower() in ("1", "true", "yes")
    RAG_RERANK_TOP_K: int = int(os.getenv("RAG_RERANK_TOP_K", "25"))
    RAG_RERANK_FINAL_N: int = int(os.getenv("RAG_RERANK_FINAL_N", "15"))
    RAG_CE_MAX_CHARS: int = int(os.getenv("RAG_CE_MAX_CHARS", "1024"))
    # Cross-encoder model for reranking. Use RAG_CE_MODEL to swap in a domain-specific fine-tuned model (e.g. legal/regulatory).
    RAG_CE_MODEL: str = os.getenv("RAG_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RAG_USE_MMR: bool = os.getenv("RAG_USE_MMR", "0").lower() in ("1", "true", "yes")
    RAG_MMR_LAMBDA: float = float(os.getenv("RAG_MMR_LAMBDA", "0.7"))

    # Step 3: Query-type classifier for dynamic dense/sparse weighting.
    RAG_USE_QUERY_CLASSIFIER: bool = os.getenv("RAG_USE_QUERY_CLASSIFIER", "1").lower() in ("1", "true", "yes")
    # Citation + procedural multi-type: explicit weights (dense, sparse). Default 0.45/0.55 for balanced retrieval.
    RAG_CITATION_PROCEDURAL_DENSE_WEIGHT: float = float(os.getenv("RAG_CITATION_PROCEDURAL_DENSE_WEIGHT", "0.45"))
    RAG_CITATION_PROCEDURAL_SPARSE_WEIGHT: float = float(os.getenv("RAG_CITATION_PROCEDURAL_SPARSE_WEIGHT", "0.55"))

    # Step 4: Graph expansion — follow cross-references (max 3 additional sections).
    RAG_USE_GRAPH_EXPANSION: bool = os.getenv("RAG_USE_GRAPH_EXPANSION", "1").lower() in ("1", "true", "yes")
    RAG_GRAPH_MAX_ADDITIONS: int = int(os.getenv("RAG_GRAPH_MAX_ADDITIONS", "3"))
    RAG_RESCORE_GRAPH_ADDITIONS: bool = os.getenv("RAG_RESCORE_GRAPH_ADDITIONS", "1").lower() in ("1", "true", "yes")

    # Step 5: Strict citation enforcement — require (section_path, page) in every answer.
    RAG_STRICT_CITATIONS: bool = os.getenv("RAG_STRICT_CITATIONS", "0").lower() in ("1", "true", "yes")
    # Post-process answers: add inference transparency notice, suggest sections when info missing.
    RAG_CITATION_POSTPROCESS: bool = os.getenv("RAG_CITATION_POSTPROCESS", "1").lower() in ("1", "true", "yes")

    # Step 7: Glossary priority — search glossary index first for definition queries.
    RAG_USE_GLOSSARY_PRIORITY: bool = os.getenv("RAG_USE_GLOSSARY_PRIORITY", "1").lower() in ("1", "true", "yes")
    RAG_GLOSSARY_SCORE_THRESHOLD: float = float(os.getenv("RAG_GLOSSARY_SCORE_THRESHOLD", "0.55"))

    # Paths for new hierarchical indexes
    FAISS_SECTION_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "faiss_section.index")
    SECTION_META_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "section_meta.json")
    FAISS_GLOSSARY_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "faiss_glossary.index")
    GLOSSARY_META_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "glossary_meta.json")
    CROSS_REF_GRAPH_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "cross_ref_graph.json")

settings = Settings()
