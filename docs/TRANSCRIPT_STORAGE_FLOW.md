# Transcript storage: end-to-end flow (RAG DB, 1‑min auto-store, grouped by session)

This doc explains how live transcription content gets into storage: **one flow** (WebSocket auto-store every 1 min + on stop), with **name, location, and time** stored so chat can answer queries like *"Give me the summary of last 5 mins in office"*.

---

## 1. Single storage flow (no Store button)

| Aspect | Behavior |
|--------|----------|
| **Trigger** | User clicks **Start** (with name/location from popup). Then: timer every `AUTO_STORE_INTERVAL_SEC` (default 60 s), and on **Stop** (EOS). |
| **transcripts table** | ✅ **One row per session** (grouped). First store in a session creates the row; every 1 min and on stop **appends** to `raw_text` and updates `updated_at`. |
| **documents + chunks** | ✅ Each 1-min (or on-stop) chunk is one RAG document (`transcript_kb_xxx`) with **meta**: `transcript_id`, `name`, `location`, `echodate` (session start time). |
| **Main RAG index** (FAISS + sparse) | ✅ Chunks added with metadata. |
| **Transcript-only index** | ✅ Same chunks; used when chat intent is "transcript". |
| **Name / location / time** | From **Start** popup (name, location); **echodate** = session start ISO. Stored in transcript row and in **document meta_json** for RAG filtering. |

**Store button removed.** All saving is automatic every 1 min and on stop; name and location are sent once at session start and applied to every stored chunk.

---

## 2. WebSocket flow: start, auto-store every 1 min, on stop

### 2.1 How the session starts

1. **Frontend** (LiveTranscription): user fills **Name** and **Location** in the popup (or **Default**), then clicks **Start**.  
   - Opens WebSocket to `GET /api/transcribe/ws`.  
   - Sends JSON: `{ type: "start", auto_store: true, sample_rate: 24000, name: "...", location: "..." }` (sample_rate from backend ready).

2. **Backend** (`transcribe/ws.py`):  
   - Loads Kyutai STT, sends `{ type: "ready", sample_rate: 24000 }`. On `type: "start"`: stores **name**, **location**, **started_at_iso** (now), resets **transcript_id**; creates/resets session (`SessionState`), starts **periodic auto-store task**.  
   - Sends `{ type: "ready", session_id, sample_rate }`.

3. **Frontend**: starts sending binary PCM16 (or JSON `{ type: "audio", pcm16_b64 }`) at 24 kHz (Kyutai).

### 2.2 Audio → text (in memory)

- **Backend** uses **Kyutai STT** (frame-by-frame); emits text pieces into **SessionState** (paragraphs/segments).  
- No DB write yet; this is all in-memory.  
- Client gets `partial` / `segment` / `final` messages with the live text.

### 2.3 Periodic auto-store (e.g. every 60 s)

- **Timer** (`_periodic_auto_store_fn`): every `auto_store_interval_sec` seconds (default **60**):
  1. Reads **full transcript** from session and takes only **new part** since last run: `to_store = full_text[last_auto_stored_length[0]:].strip()`.
  2. If `to_store` is non-empty:
     - **Transcripts table (grouped by session):**
       - If no **transcript_id** yet: **`create_transcript_for_session(name, location, started_at_iso, initial_text=to_store)`** → one new row, returns `tid`; set `transcript_id_ref[0] = tid`.
       - Else: **`append_transcript_chunk(transcript_id, to_store)`** → appends to that row’s `raw_text`, updates `updated_at`.
     - **RAG:** `conv_type, tags = get_metadata(to_store)`; then **`kb.kb_add_text(to_store, meta)`** with **meta** including **`transcript_id`**, **`name`**, **`location`**, **`echodate`** (session start ISO).
     - Pushes to `interval_buffer`; sends `{ type: "stored", session_id, transcript_id, items: [...] }`; updates `last_auto_stored_length[0]`.

- **`kb.kb_add_text`** → **`index.add_text(..., meta)`**: meta is stored in **documents.meta_json** (echodate, location, name, transcript_id). So every 1 min: **one transcript row** (created once per session, then appended) + **one RAG document** per chunk with time/location/name for filtering.

### 2.4 On user Stop (EOS)

- **Frontend**: user clicks **Stop** → sends `{ type: "stop" }`. Backend treats it as **EOS** (`type: "eos"`).

- **Backend** on **eos**:
  1. Stops the periodic auto-store task.
  2. Flushes any remaining audio and appends to session; **finalizes** session; sends `{ type: "final", ... }`.
  3. If `auto_store` and remainder text non-empty: same as periodic — create transcript row if none yet, else append; then **`kb.kb_add_text(to_store, meta)`** with same **name**, **location**, **echodate**, **transcript_id**. Sends `{ type: "stored", ... }`.
  4. WebSocket loop exits.

So: **one transcript row per session** (created on first store, then appended every 1 min and on stop) + **RAG documents** for each chunk with **name, location, echodate** in document meta for chat filtering.

---

## 3. Chat / RAG: time and location filters (e.g. “last 5 mins in office”)

- When the user asks e.g. **"Give me the summary of last 5 mins in office"**:
  1. **Intent** is classified as **transcript**.
  2. **`_parse_last_time_window(question)`** returns a timedelta for **"last 5 minutes"** (also supports "last N hours/days").
  3. **`_parse_location_from_question(question)`** returns **"office"** (from "in office", "at office", etc.).
  4. **Retrieve** runs over the **transcript-only index**; then hits are **filtered** using **document meta** (`documents.meta_json`): **Time:** keep only chunks whose **echodate** (session start) or document **created_at** is ≥ (now − 5 minutes). **Location:** keep only chunks whose meta **location** matches (e.g. contains "office").
  5. Filtered chunks are passed to the answer/summary flow; the model returns a summary. So **name**, **location**, and **echodate** are stored in transcript row and document meta and used to filter by time and location at query time.

### 3.2 Backend: `store_transcript_to_db`

- **Route** (`api/routes/transcribe.py`): `POST /store` → **`store_transcript_to_db(raw_text, refined_text, echotag, name, location, tags)`**.

- **`store_transcript_to_db`** (`transcribe/store_to_db.py`):

  1. **IDs and metadata**  
     - `tid = new_id("trn")` (e.g. `trn_xyz`).  
     - `echodate = now_iso()`.  
     - `name_val`, `location_val` from args (location default `"default"`).  
     - **Title**: `name_val` if provided, else `_title_for_transcript(tid, echodate)` (date_time + short id).  
     - **Tags**: if `tags` list provided, use it (up to 16); else LLM-extract from raw text.  
     - **echotag**: from arg or from tags/name.

  2. **transcripts table (SQLite)**  
     - **Single row** inserted:  
       `(id=tid, title, raw_text, polished_text, tags_json, echotag, echodate, created_at, name, location)`.  
     - This is the only place the **transcripts** table is written in the whole transcript flow.

  3. **RAG index**  
     - `index_text = raw_text + (refined_text if present)`.  
     - **`index.add_text("transcript_" + tid, index_text, meta)`** with `meta = { type: "transcript", tags, echotag, echodate, created_at }`.  
     - Same **add_document** path as above:
       - New **doc_id**.
       - **documents**: one row with `filename = "transcript_trn_xyz"`.
       - **chunks**: chunked and embedded.
       - **Main FAISS + sparse**: updated.
       - **Transcript FAISS + transcript sparse**: updated (because filename starts with `transcript_`).

So the **Store** flow does both:

- **transcripts** table: one row per stored transcript (with name, location, tags, full raw/polished text).  
- **RAG DB**: one “document” per transcript (`transcript_trn_xxx`) with chunks in **documents**, **chunks**, and both vector indexes.

---

## 4. Where things live (summary)

- **transcripts** table  
  - **One row per session** (created on first 1-min store, then appended every 1 min and on stop).  
  - Holds: id, title, raw_text (accumulated), polished_text, tags_json, echotag, echodate (session start), created_at, **updated_at**, name, location.  
  - Used for: Transcripts list in UI and as source of truth for “saved” sessions.

- **documents** table  
  - One row per **stored chunk** (each 1-min or on-stop store): `filename` like `transcript_kb_xxx`, **meta_json** includes **transcript_id**, **name**, **location**, **echodate** for RAG filtering.

- **chunks** table  
  - All chunks from both flows; each chunk has `doc_id` → documents.

- **Main RAG index** (FAISS + sparse)  
  - Contains **all** chunks (uploaded docs + both transcript flows).  
  - Used for general/global search.

- **Transcript-only index** (FAISS + sparse, transcript-only)  
  - Contains only chunks whose document `filename` starts with `transcript_`.  
  - Used when chat intent is “transcript” (e.g. “summarize my last hour of transcripts”).

So: **1‑min (and on-stop) auto-store** = **transcripts** table (one row per session, appended) **plus** RAG DB with **name**, **location**, **echodate** in document meta for queries like “last 5 mins in office”.

---

## 5. Config that affects auto-store

- **`ECHOMIND_AUTO_STORE_DEFAULT`** (default `true`): whether the WebSocket session has auto_store on when client sends `start` with `auto_store: true`.  
- **`AUTO_STORE_INTERVAL_SEC`** / **`ECHOMIND_AUTO_STORE_INTERVAL_SEC`** (default **60**): interval in seconds for the periodic auto-store. Set to **0** to disable periodic store (only on-stop store will run).

---

## 6. Quick diagram

```
[Frontend: Start with name, location] ──► WebSocket start (auto_store: true, name, location)
       │
       ▼
[Audio] ──► Kyutai STT ──► SessionState (in-memory transcript)
       │
       ├── Every AUTO_STORE_INTERVAL_SEC (e.g. 60 s):
       │     new text ──► create_transcript_for_session (if first) OR append_transcript_chunk
       │     then kb_add_text(to_store, { transcript_id, name, location, echodate })
       │       ──► documents + chunks + main + transcript index (meta has time/location)
       │     (transcripts table: one row per session, raw_text appended)
       │
       └── On Stop (EOS): same (append remainder to transcript row + kb_add_text with meta)

[Chat: "summary of last 5 mins in office"]
       │
       ▼
Intent=transcript ──► retrieve from transcript index ──► filter by echodate ≥ (now−5m) and location ~ "office"
       │
       ▼
Answer/summary from filtered chunks only.
```

This is the full end-to-end picture: one flow (no Store button), transcripts table grouped by session, and RAG filtering by **time** and **location** for chat.
