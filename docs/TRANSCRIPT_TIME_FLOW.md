# Transcript time flow: UTC storage and query filtering

Transcripts are stored and filtered by time in **UTC with seconds**. Questions that mention a specific time or range are converted to UTC and only matching chunks are retrieved as context.

---

## 1. Storage: echodate in UTC with seconds

- **When storing** (POST `/api/transcribe/store` or WebSocket store): the transcript is assigned an **echodate** (ISO datetime).
  - If the client sends `echodate`, it is **normalized to UTC with seconds** via `normalize_echodate_to_utc_iso()` (e.g. `2025-02-20T14:30:00Z`). Inputs without timezone are treated as UTC.
  - If not provided, server uses `now_iso()` = current time in UTC with seconds (`%Y-%m-%dT%H:%M:%SZ`).
- **Where it’s stored**: in the `transcripts` table (`echodate`, `created_at`, `updated_at`) and in the RAG index as document/chunk metadata (`echodate`, `created_at` in `meta_json` / chunk `source`). All filtering uses this value.

---

## 2. Single reference time per request

- For each RAG request, the backend fixes **one** `reference_ts = datetime.now(timezone.utc)` at the start of `retrieve()`.
- All time-based logic (specific time range, “last 24hrs”, “yesterday”, date filters) uses this same instant so behaviour is consistent and reproducible for that request.

---

## 3. Query parsing: what the user can ask

When **intent is transcript**, the question is parsed in this order:

| Priority | Parsed form | Example | Filter applied |
|----------|-------------|---------|----------------|
| 1 | **Specific time or range** | “at 2pm”, “at 14:00 on Feb 20”, “between 2pm and 3pm”, “from 14:00 to 15:00 on Feb 20” | Chunks whose `echodate` is in `[start_utc, end_utc)` |
| 2 | **Specific date** (day only) | “today”, “yesterday”, “Feb 20”, “2025-02-20” | Chunks whose `echodate` date equals that day |
| 3 | **Last N transcripts** | “last 2 transcripts”, “pick last 2” | Chunks from the N most recent transcript documents |
| 4 | **Time window** | “last 24hrs”, “48hrs”, “last 5 mins” | Chunks with `echodate ≥ reference_ts − window` |
| 5 | **Latest/recent** (no N) | “recent transcript”, “latest transcript” | Chunks from the single most recent transcript |

- **Specific time** is interpreted in the **query timezone** (`TRANSCRIPT_QUERY_TZ`, default `UTC`), then converted to UTC. For “at 2pm” a 1‑hour window is used (e.g. 14:00–15:00 in that zone → UTC range).
- Only transcript chunks that pass the chosen filter are kept and sent as context to the model.

---

## 4. Retrieval flow (embedding + time filter)

1. **Embedding search**: Same as before — query (and optional query expansion) is run against the **transcript index** (dense + sparse, RRF, optional rerank). This returns a set of candidate chunks.
2. **Time/date filter**: Using the parsed intent above, candidate chunks are filtered so that only those whose document **echodate** (in UTC with seconds) satisfies the condition (time range, date, last N, window, or latest).
3. **Context**: Only these filtered chunks are used to build the context block for the LLM. So the model only sees content from the requested time/date/recency.

This keeps retrieval semantic (embedding search) while restricting results to the exact time or range the user asked for.

---

## 5. Configuration

- **`ECHOMIND_TRANSCRIPT_QUERY_TZ`** (default `UTC`): Timezone for interpreting “at 2pm” / “between 2pm and 3pm” in the question (e.g. `America/New_York`). Stored and compared values remain in UTC.

---

## 6. Manual add on test page

When adding a transcript manually, the **Date** picker sets the **day**; the backend uses that as **noon UTC** for that day (`YYYY-MM-DDT12:00:00.000Z`) so the transcript has a precise UTC timestamp with seconds for filtering (e.g. “at 14:00 on Feb 19” or “between 09:00 and 10:00 on Feb 20”).
