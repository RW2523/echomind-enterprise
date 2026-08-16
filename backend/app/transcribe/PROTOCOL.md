# Live transcript WebSocket protocol (`/api/transcribe/ws`) — Silent Assistant v2

Additive over the v1 protocol. Old clients that only understand `partial/segment/final/analysis(label)` keep working.

## Client → server

| type | fields | notes |
|---|---|---|
| `start` | `name?`, `location?`, `auto_store?`, `sample_rate?`, `namespace?`, `analysis_always_surface?` (legacy), **`scenario?`** = `auto\|customer_care\|legal\|banking\|general`, **`participants?`** = `[{role, name?}]`, **`subject_hint?`** = `{name?, ids?: string[]}`, **`analysis_mode?`** = `flags_only\|flags_and_records` | server replies with `session` ack |
| binary frame | PCM16 LE mono @ `sample_rate` | unchanged |
| `speaker` | `role: string\|null` | who is speaking from now on (role ids from `session.roles`); null = unknown |
| `scenario` | `scenario` | switch profile mid-session (state kept); server re-sends `session` |
| `subject` | `subject_id`, `action: confirm\|reject` | operator confirms/rejects a candidate subject (person) |
| `pause` / `resume` / `eos` / `stop` / `ping` | | unchanged |

## Server → client

| type | fields |
|---|---|
| `loading`, `ready{sample_rate}`, `error{message}`, `overloaded{...}` | unchanged |
| **`session`** | `session_id, scenario, scenario_label, namespace, analysis_mode, kb_docs:int, roles:{me,other}, tag_vocab:[{id,label,tone,proof,description}]` — ack after `start`/`scenario` |
| **`warning`** | `code: namespace_empty\|llm_slow\|overloaded\|stt_dropped`, `message` |
| `partial` | unchanged (`text`, `partial_text`, `segments[]`) |
| `segment` | v1 fields + **`role`**, **`sentences:[{sentence_id,text,char_start,char_end,role}]`** |
| `analysis_start` | `segment_id` + **`sentence_id`** |
| **`analysis`** (superset of v1 card) | `id, session_id, segment_id, sentence_id, sentence_text, char_start, char_end, role, kind: claim\|personal_detail\|commitment\|question\|filler, label (legacy v1 label derived from verdict/tags), verdict: supported\|contradicted\|unverified\|null, confidence, explanation, tags:[{tag,label,tone,confidence}], evidence:[{quote, chunk_id, doc_id, doc_title, page, section_path, kind: document\|transcript\|rule, rule_id?}], record_ids:[], searched_docs:[], llm_skipped:bool, latency_ms, phrase (TTS-ready sentence), source_chunks:[legacy {chunk_id,text,doc_title,doc_id}], segment_text (legacy), created_at` |
| `analysis_done` | `segment_id, sentence_id, status: checked\|skipped\|timeout\|no_tags, searched_docs?, result:null` |
| **`entity`** | `id, kind, value, normalized, sentence_id, segment_id, role, confidence, subject_id?` |
| **`subject`** | `id, kind: customer\|client\|account_holder\|counterparty\|person, display_name, matched_fields:[], entity_ids:[], confidence, status: candidate\|confirmed\|rejected, records_count` |
| **`record`** | `id, subject_id?, entity_id?, sentence_id, kind: customer_file\|contract\|ticket\|previous_call\|matter\|related_case\|account\|product\|policy\|kyc\|document, title, doc_id, doc_title, page, section_path, quotes:[{text,chunk_id}], score (0-1), match: exact\|fuzzy\|semantic, namespace, source_transcript_id?` |
| **`scenario_suggest`** | `scenario, confidence, reason` |
| `final`, `stored`, `stored_combined`, `refined` | unchanged |

## REST additions
- `GET /api/transcribe/scenarios` → `[{id,label,description,default_namespace,roles,analysis_mode_default,tag_vocab}]`
- `GET /api/transcribe/transcripts/{id}/assistant` → `{checks:[analysis payloads], entities, subjects, records, segments}`

## Legacy `label` mapping (for old renderers)
verdict supported → `Supported`; contradicted → `Contradicted`; tag violating → `Violating`; tag risk/disclosure-missing → `Risky Statement`; verdict unverified → `Unverified`; else (records/reference only) → `Relevant`.
