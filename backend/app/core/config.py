from pydantic_settings import BaseSettings
import os

# Base dir for all persistence; env ECHOMIND_DATA_DIR overrides. Used so DB_PATH, FAISS_*, etc. stay under one root.
_DEFAULT_DATA_DIR = os.path.abspath(os.getenv("ECHOMIND_DATA_DIR", "./data"))
os.makedirs(_DEFAULT_DATA_DIR, exist_ok=True)


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
    LLM_MAX_TOKENS: int = int(os.getenv("ECHOMIND_LLM_MAX_TOKENS", "2048"))
    OLLAMA_EMBED_URL: str = "http://ollama:11434/api/embeddings"
    OLLAMA_EMBED_MODEL: str = os.getenv("ECHOMIND_EMBED_MODEL", "nomic-embed-text")
    # Max characters per chunk sent to embedding API. nomic-embed-text context varies by Ollama build;
    # 2000 chars (~500 tokens) is safe for all configurations. Set higher only if your model supports it.
    EMBED_MAX_CHARS: int = int(os.getenv("ECHOMIND_EMBED_MAX_CHARS", "2000"))
    CHUNK_SIZE: int = int(os.getenv("ECHOMIND_CHUNK_SIZE", "450"))
    CHUNK_OVERLAP: int = int(os.getenv("ECHOMIND_CHUNK_OVERLAP", "120"))
    TOP_K: int = int(os.getenv("ECHOMIND_TOP_K", "15"))
    # Large-doc parent/child token limits (BOOK type). Override for dense regulatory docs.
    BOOK_PARENT_MIN_TOKENS: int = int(os.getenv("ECHOMIND_BOOK_PARENT_MIN_TOKENS", "2000"))
    BOOK_PARENT_MAX_TOKENS: int = int(os.getenv("ECHOMIND_BOOK_PARENT_MAX_TOKENS", "3500"))
    BOOK_CHILD_MIN_TOKENS: int = int(os.getenv("ECHOMIND_BOOK_CHILD_MIN_TOKENS", "500"))
    BOOK_CHILD_MAX_TOKENS: int = int(os.getenv("ECHOMIND_BOOK_CHILD_MAX_TOKENS", "400"))
    # Dynamic chunk sizing: auto-shrink children for dense technical content.
    BOOK_DYNAMIC_CHUNK_SIZING: bool = os.getenv("ECHOMIND_BOOK_DYNAMIC_CHUNK_SIZING", "1").lower() in ("1", "true", "yes")
    # Large section threshold: split sections exceeding this (tokens) into multiple chunks. Default 10000.
    LARGE_SECTION_TOKEN_THRESHOLD: int = int(os.getenv("ECHOMIND_LARGE_SECTION_TOKEN_THRESHOLD", "10000"))
    RAG_RELEVANCE_THRESHOLD: float = float(os.getenv("ECHOMIND_RAG_RELEVANCE_THRESHOLD", "0.45"))
    # Expose citations (section, page, filename) to client for source-aware answers.
    RAG_EXPOSE_SOURCES: bool = os.getenv("ECHOMIND_RAG_EXPOSE_SOURCES", "1").lower() in ("1", "true", "yes")
    # Log citation pipeline debug (enriched count, valid metadata, citations built). Set RAG_CITATION_DEBUG=1 to troubleshoot missing sources.
    RAG_CITATION_DEBUG: bool = os.getenv("RAG_CITATION_DEBUG", "0").lower() in ("1", "true", "yes")

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
    # Max total context chars; 0 = no limit. When exceeded, trim lowest-scoring blocks. 24k for 7300-page DoD FMR.
    RAG_CONTEXT_MAX_CHARS: int = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "24000"))

    # Real-time transcription & knowledge capture (Nemotron NeMo ASR, 16 kHz)
    ECHOMIND_AUTO_STORE_DEFAULT: bool = os.getenv("ECHOMIND_AUTO_STORE_DEFAULT", "1").lower() in ("1", "true", "yes")
    # When auto_store is on: store new transcript content to the KB every N seconds (0 = only on stop).
    AUTO_STORE_INTERVAL_SEC: int = int(os.getenv("ECHOMIND_AUTO_STORE_INTERVAL_SEC", "60"))
    TRANSCRIPT_SILENCE_COMMIT_MS: int = int(os.getenv("TRANSCRIPT_SILENCE_COMMIT_MS", "800"))
    TRANSCRIPT_PARAGRAPH_SILENCE_MS: int = int(os.getenv("TRANSCRIPT_PARAGRAPH_SILENCE_MS", "2000"))
    TRANSCRIPT_MAX_PARAGRAPH_CHARS: int = int(os.getenv("TRANSCRIPT_MAX_PARAGRAPH_CHARS", "700"))
    TRANSCRIPT_RECENT_BUFFER_MAX_CHARS: int = int(os.getenv("TRANSCRIPT_RECENT_BUFFER_MAX_CHARS", "120"))
    TRANSCRIPT_OVERLAP_K: int = int(os.getenv("TRANSCRIPT_OVERLAP_K", "200"))
    TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC: float = float(os.getenv("TRANSCRIPT_EMIT_RATE_LIMIT_PER_SEC", "15"))
    SAMPLE_RATE: int = 16000  # Nemotron streaming ASR (16 kHz)
    # Nemotron: chunk length in ms (e.g. 560 matches livetranscript default for accuracy/latency tradeoff)
    TRANSCRIPT_NEMOTRON_CHUNK_MS: int = int(os.getenv("TRANSCRIPT_NEMOTRON_CHUNK_MS", "560"))
    # Must stay 1 with a single shared NeMo model (see ASRModelAdapter._forward_lock); >1 only if each worker had its own model.
    TRANSCRIPT_ASR_EXECUTOR_WORKERS: int = int(os.getenv("TRANSCRIPT_ASR_EXECUTOR_WORKERS", "1"))
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
    # Serial GPU STT avoids overlapping forwards on one shared Nemotron instance (illegal memory access).
    TRANSCRIPT_GPU_CONCURRENCY: int = int(os.getenv("TRANSCRIPT_GPU_CONCURRENCY", "1"))
    # STT warmup: number of frames to run on model load (CUDA kernels).
    TRANSCRIPT_STT_WARMUP_FRAMES: int = int(os.getenv("TRANSCRIPT_STT_WARMUP_FRAMES", "8"))
    # WebSocket receive timeout (seconds). No message for this long = stale connection. Default 24h = effectively indefinite with heartbeat.
    TRANSCRIPT_WS_RECEIVE_TIMEOUT_SEC: float = float(os.getenv("TRANSCRIPT_WS_RECEIVE_TIMEOUT_SEC", "86400"))

    # ── Hierarchical / Graph-aware RAG feature toggles ──────────────────────────
    # Step 1: Section-level FAISS index restricts child search to top-N sections first.
    RAG_USE_SECTION_RETRIEVAL: bool = os.getenv("RAG_USE_SECTION_RETRIEVAL", "1").lower() in ("1", "true", "yes")
    RAG_SECTION_SCORE_THRESHOLD: float = float(os.getenv("RAG_SECTION_SCORE_THRESHOLD", "0.35"))
    RAG_SECTION_RELAX_THRESHOLD: float = float(os.getenv("RAG_SECTION_RELAX_THRESHOLD", "0.25"))
    RAG_SECTION_TOP_K: int = int(os.getenv("RAG_SECTION_TOP_K", "10"))
    # Max sections to restrict chunk retrieval to. Higher for large docs (7300-page DoD FMR). Only global retrieval when 0 sections found.
    MAX_SECTIONS_PER_RETRIEVAL: int = int(os.getenv("RAG_MAX_SECTIONS_PER_RETRIEVAL", "5"))

    # Step 2: Cross-encoder re-ranker (sentence-transformers if available, else LLM fallback).
    RAG_USE_RERANKER: bool = os.getenv("RAG_USE_RERANKER", "1").lower() in ("1", "true", "yes")
    RAG_RERANK_TOP_K: int = int(os.getenv("RAG_RERANK_TOP_K", "25"))
    RAG_RERANK_FINAL_N: int = int(os.getenv("RAG_RERANK_FINAL_N", "15"))
    RAG_CE_MAX_CHARS: int = int(os.getenv("RAG_CE_MAX_CHARS", "1500"))
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
    # EvidenceGate: refuse when evidence is weak (confidence/coverage below threshold). Default OFF for single-doc DoD FMR.
    RAG_USE_EVIDENCE_GATE: bool = os.getenv("RAG_USE_EVIDENCE_GATE", "0").lower() in ("1", "true", "yes")
    # AnswerGating: verify context has explicit section id + key tokens before answering. Default OFF for single-doc DoD FMR.
    RAG_USE_ANSWER_GATING: bool = os.getenv("RAG_USE_ANSWER_GATING", "0").lower() in ("1", "true", "yes")
    # Post-process answers: add inference transparency notice, suggest sections when info missing.
    RAG_CITATION_POSTPROCESS: bool = os.getenv("RAG_CITATION_POSTPROCESS", "1").lower() in ("1", "true", "yes")
    # When True, run literal substring (grep-like) search over chunks when semantic/BM25 scores are weak. Catches acronyms (DDRS), exact terms BM25 may miss.
    RAG_USE_KEYWORD_FALLBACK: bool = os.getenv("RAG_USE_KEYWORD_FALLBACK", "1").lower() in ("1", "true", "yes")
    # Run keyword fallback when best document score is below this (default 0.45 = RAG_RELEVANCE_THRESHOLD).
    RAG_KEYWORD_FALLBACK_THRESHOLD: float = float(os.getenv("RAG_KEYWORD_FALLBACK_THRESHOLD", "0.45"))

    # Step 7: Glossary priority — search glossary index first for definition queries.
    RAG_USE_GLOSSARY_PRIORITY: bool = os.getenv("RAG_USE_GLOSSARY_PRIORITY", "1").lower() in ("1", "true", "yes")
    RAG_GLOSSARY_SCORE_THRESHOLD: float = float(os.getenv("RAG_GLOSSARY_SCORE_THRESHOLD", "0.55"))

    # Paths for new hierarchical indexes
    FAISS_SECTION_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "faiss_section.index")
    SECTION_META_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "section_meta.json")
    FAISS_GLOSSARY_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "faiss_glossary.index")
    GLOSSARY_META_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "glossary_meta.json")
    CROSS_REF_GRAPH_PATH: str = os.path.join(_DEFAULT_DATA_DIR, "cross_ref_graph.json")
    # BookRAG: TOC index for chapter/section routing when no explicit refs
    RAG_USE_TOC_ROUTING: bool = os.getenv("RAG_USE_TOC_ROUTING", "1").lower() in ("1", "true", "yes")
    RAG_TOC_TOP_K: int = int(os.getenv("RAG_TOC_TOP_K", "8"))
    RAG_TOC_THRESHOLD: float = float(os.getenv("RAG_TOC_THRESHOLD", "0.25"))
    # BookRAG: max chunks per section (4 for single-doc DoD FMR; 2 for multi-doc)
    RAG_MAX_CHUNKS_PER_SECTION: int = int(os.getenv("RAG_MAX_CHUNKS_PER_SECTION", "4"))

    # ── Section limiting (reduce cross-section contamination) ───────────────────
    # Max sections to include in context. 6 for single-doc DoD FMR (cross-volume answers); 2–3 for multi-doc.
    MAX_SECTIONS_PER_ANSWER: int = int(os.getenv("RAG_MAX_SECTIONS_PER_ANSWER", "6"))
    MAX_EVIDENCE_SENTENCES: int = int(os.getenv("RAG_MAX_EVIDENCE_SENTENCES", "15"))
    # Graph expansion: only when top section confidence below this (0 = always)
    RAG_GRAPH_EXPANSION_CONFIDENCE_THRESHOLD: float = float(os.getenv("RAG_GRAPH_EXPANSION_CONFIDENCE_THRESHOLD", "0.5"))
    MAX_GRAPH_EXPANSION_DEPTH: int = int(os.getenv("RAG_GRAPH_EXPANSION_DEPTH", "1"))
    # Graph additions must exceed direct hit score by this margin to override
    RAG_GRAPH_OVERRIDE_MARGIN: float = float(os.getenv("RAG_GRAPH_OVERRIDE_MARGIN", "0.15"))
    # Hybrid TOC: combine vector + BM25 with RRF
    RAG_TOC_HYBRID_SEARCH: bool = os.getenv("RAG_TOC_HYBRID_SEARCH", "1").lower() in ("1", "true", "yes")

    # ── Contextual retrieval (BOOK chunks) ───────────────────────────────────────
    # When True, generate context headers for BOOK children (document, volume, chapter, section, parent summary, chunk role).
    # Use contextualized_text for embeddings/BM25; raw text for evidence and citations.
    RAG_USE_CONTEXTUAL_RETRIEVAL: bool = os.getenv("RAG_USE_CONTEXTUAL_RETRIEVAL", "1").lower() in ("1", "true", "yes")
    # Max concurrent LLM calls for section/chunk summarization during ingestion.
    RAG_CONTEXTUAL_SUMMARY_CONCURRENCY: int = int(os.getenv("RAG_CONTEXTUAL_SUMMARY_CONCURRENCY", "3"))

settings = Settings()
