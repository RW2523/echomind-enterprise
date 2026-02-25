# RAG Platform (New Architecture)

Production RAG with **Qdrant**, **Qwen3 0.6B embeddings**, and **Qwen-14B generator** on GPU (DGX Spark ARM64). Models load once at startup via HuggingFace + safetensors.

## Features

- **Documents**: PDF, DOCX, PPTX, TXT, MD, CSV, XLSX → extract → chunk (~800 tokens, 120 overlap) → tag → embed → Qdrant `documents` + catalog table.
- **Transcripts**: Live stream → 60s chunks → tag → embed → Qdrant `transcripts`.
- **Query routing**: Deterministic intent (TRANSCRIPT_FIRST, DOCUMENT_FIRST, SUMMARIZE_DOC, GENERAL) with fallback: transcript first → if &lt;15 or best &lt; T_low → documents → if empty → general.
- **Retrieval**: Time filters (last N mins/hours), location/tags, score thresholds T1/T2/T_low, top_k=15.
- **Generation**: Answer with citations (transcript_id + time range + location; doc_title + page + section) and optional "Retrieved Evidence" block.

## Layout

```
rag-platform/
  app/
    main.py              # FastAPI + lifespan (load embedder + generator)
    core/                 # config, logging, timeutils
    models/               # embedder (Qwen3 0.6B), generator (Qwen-14B)
    qdrant/               # client, collections, search, upsert
    ingestion/            # pipeline_docs, pipeline_transcript, extractors, chunking, tagging
    router/               # intent, orchestrator, prompts
    catalog/              # db (SQLite/Postgres), dao
    api/                  # routes_docs, routes_transcript, routes_query
  docker/
    Dockerfile.arm64.cuda
    docker-compose.yml    # qdrant + app (GPU)
  requirements.txt
```

## API

- `POST /docs/upload` — upload file → returns `doc_id`
- `GET /docs/list` — list catalog
- `GET /docs/{doc_id}` — get document metadata
- `POST /transcripts/ingest` — body: `{ transcript_id, ts, location?, text, tags?, timezone? }`
- `POST /transcripts/ingest_batch` — body: `{ transcript_id, lines: [{ text, ts }], location?, timezone? }`
- `POST /query` — body: `{ user_query, mode?, doc_id? }` → `{ answer, evidence[], source_used, from_sources }`
- `GET /health` — health check

## Config (env)

- `QDRANT_URL` — Qdrant URL (default `http://localhost:6333`)
- `EMBED_MODEL_ID` — default `Qwen/Qwen3-Embedding-0.6B`
- `GENERATOR_MODEL_ID` — default `Qwen/Qwen2.5-14B-Instruct`
- `DEVICE` — `cuda` (GPU)
- `TOP_K`, `T1_TRANSCRIPT`, `T2_DOCUMENT`, `T_LOW`, `MAX_CONTEXT_CHUNKS` — retrieval tuning
- `CHUNK_SIZE`, `CHUNK_OVERLAP` — document chunking (default 800, 120)
- `DATABASE_URL` — optional Postgres; if unset, SQLite under `DATA_DIR`

## Run

```bash
# From rag-platform/
pip install -r requirements.txt
# Start Qdrant (e.g. docker run -p 6333:6333 qdrant/qdrant)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or with Docker Compose (V2 plugin; from `rag-platform/`):

```bash
cd rag-platform && docker compose -f docker/docker-compose.yml up --build
```

If you have the legacy standalone binary, use `docker-compose` instead of `docker compose`.

Models are loaded at first request (lifespan); ensure GPU is available and CUDA visible for `DEVICE=cuda`.
