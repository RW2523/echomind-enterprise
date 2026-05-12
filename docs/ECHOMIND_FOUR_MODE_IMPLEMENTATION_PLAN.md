# EchoMind four-mode product plan

This document is the authoritative implementation plan for EchoMind as a **local-first** stack (React/TypeScript frontend, FastAPI backend, voice service, local STT/TTS, local RAG, local SQLite storage, existing Docker/compose). **No cloud services** in scope for these modes.

**Naming:** Use **EchoMind** only. Do not introduce external product codenames in user-facing copy or primary docs.

**Removed from product scope (do not ship or extend):**

- Rules Library, policy management, rule sets, individual rules, policy packs, session rule activation
- Session notes, pinned findings, saved correction/suggestion archives, rule-based findings, rules+RAG badges
- Any UI or API surface that depended on the above

Legacy tables or modules may remain on disk for migration safety; they are **not** part of the product.

---

## Global rule

All four modes **listen and transcribe** while the session is active (shared transcription pipeline). Differences are **what happens after** text is available (suggestions, highlights, voice reply, or nothing beyond persistence).

---

## Mode definitions

### 1. Transcribe Mode

**Purpose:** Listen and convert speech into text.

**Behavior:** Live + partial/final transcript; local persistence if already supported; **no** TTS, suggestions, fact-check, or general reasoning.

**Code map:** `AppView.TRANSCRIPTION` + `LiveTranscription` (sidebar label: **Transcribe**).

### 2. Conversation Mode

**Purpose:** Real-time spoken conversation with EchoMind.

**Behavior:** Listen + transcribe; user speaks naturally; EchoMind replies by voice; **knowledge base only when the user asks a document/knowledge question**; show citations/source preview when RAG is used; barge-in / short replies by default; do not read citations aloud.

**Code map:** `AppView.VOICE_CONVERSATION` + `VoiceConversation` + voice WebSocket (sidebar: **Conversation**).

### 3. Assistant Mode

**Purpose:** Listen quietly; **raise a hand** only when the **local knowledge base** has useful, citable feedback.

**Behavior:** Continuous transcribe; **no auto speech**; suggestions **only** with KB evidence (or explicitly weak/unverified with evidence tier); user opens card → feedback, explanation, source, page/snippet, evidence status; approve → speak; ignore/dismiss; **no** rules, **no** notes, **no** general-knowledge claims without retrieval.

**Code map:** `AppView.ASSISTANT` + `AssistantMode` + `/api/assistant/*`.

### 4. Silent Assistant Mode

**Purpose:** Silently check transcript segments against the **knowledge base** and highlight stance.

**Behavior:** Continuous transcribe; **never** TTS; analyze stable/finalized segments; highlights use **Supported**, **Contradicted**, **Unverified**, **Needs Review** (display labels mapped from internal evidence/alignment); click → status, explanation, sources, evidence; **no** rules/policy packs; if evidence is insufficient → Unverified / Needs Review, not guesses.

**Code map:** `AppView.SILENT_ASSISTANT` + `SilentAssistantMode` + `/api/silent-assistant/*`.

**Companion (not a “listening mode”):** Knowledge Chat remains for document Q&A without replacing the four modes above.

---

## Phased delivery

| Phase | Goal |
|-------|------|
| **1** | Keep Transcribe + Conversation stable; sidebar/titles: **Transcribe**, **Conversation**. |
| **2** | Shared foundation: STT naming consistency, KB flags wired for Assistant/Silent, transcript segments stable, voice pipeline unchanged. |
| **3** | Conversation: source chips / panel when answers use RAG (no TTS of citations). |
| **4** | Assistant: hand-raise queue, KB-only generation, approve → speak, dismiss/ignore, source preview. |
| **5** | Silent: finalized-sentence analysis, four labels, detail panel, no speech. |
| **6** | UI polish: four modes prominent, consistent status + citation UI; no removed-scope entry points. |

---

## Hard rules (engineering)

- Do not implement or expose Rules Library, notes, pins, or saved archives as product features.
- Do not use cloud services for these flows.
- Do not invent evidence: Assistant and Silent use **local RAG only** as authority.
- Assistant speaks **only** after user approval.
- Silent Assistant **never** calls TTS.

---

## Report location

After each major scope pass, append or link a short **EchoMind Four-Mode Scope Update Report** (removed scope, files touched, gaps, manual tests) in the repo root or here; the latest chat may also paste the report for reviewers.
