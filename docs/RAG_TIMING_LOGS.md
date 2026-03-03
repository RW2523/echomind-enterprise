# RAG Timing Logs

EchoMind logs timing for each step of the RAG pipeline. All logs use the prefix `RAG_TIMING` for easy filtering.

## Log Format

```
RAG_TIMING step=<step_name> duration_ms=<ms> [extra_key=value ...]
```

## Steps Logged

### Retrieval (`retrieve_semantic_first`)

| Step | Description |
|------|-------------|
| `retrieve_embed` | Query embedding (Ollama/SGLang embed API) |
| `retrieve_search_all` | Dense (FAISS) + sparse (BM25) search, parallel |
| `retrieve_rrf_filter` | RRF merge, context window filter, time decay, tag boost |

### Single-query retrieval (`retrieve_single_query`)

| Step | Description |
|------|-------------|
| `retrieve_single_query_search` | Index search (embed + FAISS) |
| `retrieve_single_query_filter` | Context window filter, time decay |

### Context building

| Step | Description |
|------|-------------|
| `build_rag_context` | Parent expansion, optional LLM compress, dedupe |

### LLM

| Step | Description |
|------|-------------|
| `answer_llm` | Chat completion (SGLang) |
| `answer_retrieve` | Full retrieve step (answer path) |
| `answer_build_context` | Full build_context step (answer path) |
| `answer_total` | Total RAG request time |

### Stream

| Step | Description |
|------|-------------|
| `answer_stream_retrieve` | Full retrieve step |
| `answer_stream_build_context` | Full build_context step |
| `answer_stream_llm` | Streaming LLM call |
| `answer_stream_total` | Total stream request time |

### API

| Step | Description |
|------|-------------|
| `ask_request_total` | Full `/ask` request (including DB writes) |
| `ask_stream_request_total` | Full `/ask-stream` request |

### Background

| Step | Description |
|------|-------------|
| `update_conversation_summary` | Conversation summary LLM call (runs in background) |

## Viewing Logs

Filter backend logs for RAG timing:

```bash
docker compose logs -f backend 2>&1 | grep RAG_TIMING
```

Example output:

```
RAG_TIMING step=retrieve_embed duration_ms=45
RAG_TIMING step=retrieve_search_all duration_ms=12 tasks=4
RAG_TIMING step=retrieve_rrf_filter duration_ms=3 transcript_hits=8 document_hits=5
RAG_TIMING step=answer_retrieve duration_ms=62 source=document hits=8
RAG_TIMING step=build_rag_context duration_ms=120 blocks=5 compress=False
RAG_TIMING step=answer_build_context duration_ms=120
RAG_TIMING step=answer_llm duration_ms=850
RAG_TIMING step=answer_total duration_ms=1050 path=rag source=document
RAG_TIMING step=ask_request_total duration_ms=1080
```

## Typical Breakdown

- **Embed**: 30–100 ms (Ollama CPU)
- **Search**: 5–20 ms (FAISS + BM25)
- **Build context**: 10–200 ms (depends on compress)
- **LLM**: 500–3000 ms (depends on model and response length)
