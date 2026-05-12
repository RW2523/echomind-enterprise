# EchoMind Four-Mode Scope Update Report

Date: aligned with repo update implementing `docs/ECHOMIND_FOUR_MODE_IMPLEMENTATION_PLAN.md`.

## Removed scope

- **Volta** naming removed from UI and TypeScript comments (repo grep: none).
- **Rules Library** UI and API mount removed (`/api/rules-library` no longer registered on `FastAPI` app).
- **Session notes** UI and API mount removed (`/api/session-notes` no longer registered).
- **Assistant** suggestions: no rule-based cards, no session-notes context, no transcript-only or notes-only cards; **KB-only** rows with at least one retrieval citation.
- **Silent Assistant**: no rule matching or rule-hint findings; analysis requires `use_knowledge_base`; removed save/pin endpoints that wrote to notes.
- **Frontend** removed: `RulesLibrary.tsx`, `SessionNotesPanel.tsx`, rules/notes/pin/save API client helpers, save/pin/archive actions on Assistant and Silent UIs.

## Final four modes

1. **Transcribe** — `AppView.TRANSCRIPTION` / `LiveTranscription` (listen + live transcript; no voice AI).
2. **Conversation** — `AppView.VOICE_CONVERSATION` / `VoiceConversation` (duplex voice; KB when configured for voice).
3. **Assistant** — `AppView.ASSISTANT` / `AssistantMode` (listen + transcribe; hand-raises only from **local RAG** with citations; speak after approve).
4. **Silent Assistant** — `AppView.SILENT_ASSISTANT` / `SilentAssistantMode` (listen + transcribe; highlights; labels **Supported / Contradicted / Unverified / Needs Review** in UI).

**Knowledge Chat** remains a separate nav item (document Q&A), not counted as one of the four listening modes.

## Existing code reused

- Live transcription hook and WebSocket pipeline unchanged.
- Voice conversation WebSocket and `useVoiceConnection` unchanged.
- RAG `retrieve_single_query`, citation mapping, `ChunkCitationModal`, SQLite stores for suggestions/findings (schema columns for legacy rule fields retained but unused in new flows).

## Files changed

- `docs/ECHOMIND_FOUR_MODE_IMPLEMENTATION_PLAN.md` (new)
- `docs/ECHOMIND_FOUR_MODE_SCOPE_UPDATE_REPORT.md` (this file)
- `README.md` (link to plan; Transcribe wording)
- `backend/app/main.py` (drop rules + session-notes routers)
- `backend/app/api/routes/assistant.py` (remove save + notes)
- `backend/app/api/routes/silent_assistant.py` (remove save/pin + notes)
- `backend/app/assistant/suggestion_generator.py` (KB-only generation; drop rules/notes helpers)
- `backend/app/silent_assistant/silent_analyzer.py` (drop rules paths; KB required)
- `backend/tests/test_silent_findings.py` (kb_disabled test)
- `frontend/types.ts` (EchoMind comments; drop `AppView` rules/notes; remove rule/note TS interfaces)
- `frontend/App.tsx`, `Header.tsx`, `Sidebar.tsx`, `constants.tsx`
- `frontend/utils/modeChrome.ts` (Silent display labels/severity; source labels)
- `frontend/services/backend.ts` (prune rules/notes/save/pin APIs)
- `frontend/components/AssistantMode.tsx`
- `frontend/components/SilentAssistantMode.tsx`

## New files created

- `docs/ECHOMIND_FOUR_MODE_IMPLEMENTATION_PLAN.md`
- `docs/ECHOMIND_FOUR_MODE_SCOPE_UPDATE_REPORT.md`

## Files deleted

- `frontend/components/RulesLibrary.tsx`
- `frontend/components/SessionNotesPanel.tsx`
- `backend/tests/test_rules_library.py`
- `backend/tests/test_session_notes.py`

## Remaining gaps

- **Conversation Mode Phase 3**: richer citation/source panel for voice replies when RAG is used (chips exist in some paths; full polish TBD).
- **Phase 6 UI**: optional consolidation if Knowledge Chat should move under a single “Documents” area; settings copy for “Use knowledge base” vs Assistant/Silent always-on KB.
- **Legacy code on disk**: `backend/app/rules_library/`, `backend/app/session_notes/`, route modules `api/routes/rules_library.py` and `session_notes.py`, and SQLite migrations for old tables remain but are **unmounted** and not referenced by the four-mode UI.

## Manual tests to run

1. `docker compose up` (or local dev): open app → confirm sidebar has **no** Rules Library or Notes.
2. **Transcribe**: start/stop; partial + final text appear.
3. **Conversation**: connect voice; interrupt; confirm KB toggle still affects **voice** behavior.
4. **Assistant**: indexed documents present; start listening; after enough transcript + trigger, confirm suggestion appears only with **sources**; approve → hear TTS; dismiss/ignore work; no “Save to Notes”.
5. **Silent Assistant**: start listening; after analyze interval, findings show **Supported / Contradicted / Unverified / Needs Review**; detail has sources; no Save/Pin.
6. Backend: `python -m pytest tests/`; frontend: `npx tsc --noEmit`.
