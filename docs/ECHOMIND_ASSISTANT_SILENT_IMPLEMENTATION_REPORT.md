# EchoMind Assistant and Silent Assistant Implementation Report

## Files changed

### Backend

- `backend/app/schemas/transcript_analyze.py` — **new**: Pydantic models for `POST /api/assistant/analyze-transcript` (request, response, `KbFindingLabel`, sources, items).
- `backend/app/assistant/kb_transcript_analyzer.py` — **new**: KB-only analysis (sentence spans → `retrieve_for_kb_probe` → overlap alignment from `silent_analyzer` → user labels `Supported` / `Contradicted` / `Related` / `Unverified` / `Needs Review` + confidence + evidence tier). Optional persistence to `assistant_suggestions` (Assistant) or `silent_findings` (Silent).
- `backend/app/api/routes/assistant.py` — **updated**: registered `POST /analyze-transcript`.
- `backend/app/assistant/suggestion_generator.py` — **updated**: legacy generate path minimum confidence restored to **0.7** (aligned with product spec).

### Frontend

- `frontend/types.ts` — **updated**: `KbFindingLabel`, `AssistantSource`, `AssistantAnalysisItem`, `AnalyzeTranscriptResponse`.
- `frontend/services/backend.ts` — **updated**: `analyzeAssistantTranscript()` calling `/api/assistant/analyze-transcript`.
- `frontend/components/AssistantMode.tsx` — **updated**: 60s loop uses unified analyze API + `transcript_offset` / `full_transcript` to avoid resending old text; hand-raise threshold **≥ 70%**; preview shows reason + confidence; **Speak now** / **Speak later**; copy updated.
- `frontend/components/SilentAssistantMode.tsx` — **updated**: 60s loop uses same analyze API with `mode: silent_assistant` and offset tracking; highlights require **confidence > 70%**; manual verify uses analyze endpoint; detail panel labels **Explanation** / **Assistant interpretation**; removed unused digest combine for periodic analyze (committed transcript only).

## New backend endpoints/services

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/assistant/analyze-transcript` | POST | Unified KB transcript analysis for `assistant` or `silent_assistant`; returns structured `items`; optionally persists rows. |

**Service:** `analyze_transcript()` in `kb_transcript_analyzer.py` — no external APIs; uses local `retrieve_for_kb_probe` (hybrid RAG path) + lexical alignment heuristics only.

## New frontend components

None (modes extend existing `AssistantMode` and `SilentAssistantMode`).

## Assistant Mode behavior

- Listens and transcribes via existing `useLiveTranscription`.
- Every **60 seconds**, sends **only new committed transcript** (`slice` since `lastAnalyzedOffset`) to `analyze-transcript` with `persist_results: true`.
- Backend persists suggestions when **confidence ≥ 70%** and label is actionable (Supported / Contradicted / Related / Needs Review), subject to cooldown and pending caps.
- UI: hand-raise + highlights for pending suggestions **≥ 70%**; detail shows **reason** (explanation), sources, evidence, **confidence %**; **Speak now** (approve + TTS + spoken), **Ignore**, **Dismiss**, **Speak later** (close panel, no TTS).

## Silent Assistant Mode behavior

- Same live transcription pipeline.
- Every **60s**, same offset-based analyze with `mode: silent_assistant`; persists findings when **confidence > 70%**.
- Highlights only for **confidence strictly above 70%**; never speaks; no Speak UI.
- Detail: status, explanation, assistant interpretation, sources, confidence, evidence.

## Knowledge-base evidence behavior

- Retrieval: `retrieve_for_kb_probe` → `retrieve_semantic_first` with fallback to `retrieve_single_query` (same stack as chat-oriented RAG).
- Labels and scores are **not** from a general-knowledge LLM; weak / missing retrieval maps to **Unverified** / **Needs Review** / lower confidence.
- Optional `ECHOMIND_ASSISTANT_SUGGESTION_LLM` still only **refines card copy** in the legacy `generate_suggestions` path (not used by the new analyze endpoint).

## Remaining limitations

- Span splitting is heuristic (sentence boundaries; short utterances may be batched).
- Char offsets depend on client `full_transcript` + `transcript_offset`; if transcript is edited externally, highlights may drift until the next segment-aligned refresh.
- Cooldowns (50s Assistant / 42s Silent per store) can skip back-to-back inserts even when new text arrived.
- Legacy `POST .../suggestions/generate` and `POST .../findings/analyze` remain for compatibility but primary UI uses **analyze-transcript**.

## Manual tests to run

1. **Transcribe Mode:** Start live transcript only — no KB hand-raises, no TTS from Assistant flows.
2. **Conversation Mode:** Normal voice duplex; KB only when user asks document-style questions (unchanged).
3. **Assistant Mode:** Start listening, speak 90+ chars of on-index content; after ~60s expect analyze; with strong retrieval see hand-raise **≥ 70%**; Review → **Speak now** → hear TTS; **Ignore** / **Dismiss** silent; **Speak later** closes panel without speech.
4. **Silent Assistant Mode:** Same speech; after ~60s highlights appear only **> 70%**; click highlight → detail; confirm **no** audio; **Verify selection** still runs analyze on selection.
5. **Offset / dedupe:** Leave session running multiple minutes; transcript grows; confirm no duplicate spam and suggestions still appear for new claims.
6. **Mode switch:** Assistant → Silent (or Transcribe) → confirm prior interval cleared (unmount) and no cross-mode TTS in Silent.
