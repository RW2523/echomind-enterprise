# RAG Flow & GPU Audit

This document audits where each component runs (CPU vs GPU) and how to verify the full flow.

## Current Architecture

| Component | Service | Device | Notes |
|-----------|---------|--------|-------|
| **LLM (chat/completions)** | sglang-llm | **GPU** | SGLang with Llama-3.1-8B-Instruct-FP8 |
| **Embeddings** | ollama-embed | **CPU** | nomic-embed-text via Ollama (no GPU) |
| **FAISS (vector search)** | backend | **CPU** | faiss-cpu (USE_FAISS_GPU=0 frees GPU for SGLang) |
| **BM25 (sparse search)** | backend | **CPU** | rank-bm25, no GPU version |
| **Kyutai STT (Live Transcript)** | backend | **GPU** | Uses backend's GPU reservation |

## Data Flow

```
User question
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. EMBEDDING (CPU - ollama-embed)                                │
│    Query → nomic-embed-text → vector                             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. RETRIEVAL (CPU - backend)                                     │
│    • Dense: FAISS search (faiss-cpu)                             │
│    • Sparse: BM25 search (rank-bm25)                            │
│    • RRF merge → top-k hits                                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. CONTEXT BUILD (CPU - backend)                                │
│    Parent expansion, optional LLM compress, dedupe               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. LLM (GPU - sglang-llm)                                        │
│    System + user + context → SGLang → answer                      │
└─────────────────────────────────────────────────────────────────┘
```

## Why Embeddings & FAISS Are CPU

- **Single GPU (DGX Spark)**: SGLang LLM uses ~10–15 GB. Keeping embeddings and FAISS on CPU avoids OOM.
- **Embeddings**: Ollama nomic-embed-text runs fine on CPU; typical latency 30–100 ms per query.
- **FAISS CPU**: Vector search is fast enough for TOP_K=15; GPU helps mainly for very large indexes.

## Enabling More GPU (Optional)

### FAISS GPU

If you have enough VRAM (e.g. after reducing LLM mem-fraction):

1. In `docker-compose.yml`: `USE_FAISS_GPU: "1"`
2. Rebuild: `docker compose build --no-cache backend`
3. Backend will use faiss-gpu for vector search (faster on large indexes).

### Embeddings on GPU

SGLang supports `--is-embedding` for embedding models. To run embeddings on GPU you would need:

- A separate SGLang embedding container (e.g. e5-mistral, gte, mcdse)
- Or a different embedding service with GPU support

**Current choice**: Ollama CPU embeddings are sufficient for typical RAG latency.

## Verification Checklist

Run the verification script:

```bash
./scripts/verify-rag-flow.sh
```

Or manually:

1. **SGLang LLM**: `curl -s http://localhost:30000/health` → `{"status":"ok"}`
2. **Ollama Embed**: `curl -s http://localhost:11435/api/embeddings -d '{"model":"nomic-embed-text","prompt":"test"}'` → JSON with embedding
3. **Backend Chat**: `/ask` or `/ask-stream` returns a valid answer
4. **RAG Timing**: `docker compose logs backend 2>&1 | grep RAG_TIMING` shows per-step timing
