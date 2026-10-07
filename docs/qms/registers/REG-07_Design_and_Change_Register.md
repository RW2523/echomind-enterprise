# REG-07 — Design and Change Register

| Field | Value |
|---|---|
| Document ID | REG-07 |
| Revision | 1.0 |
| Status | **LIVE REGISTER — currently empty** |
| Owner | Lead Engineer (Richard Watson Stephen Amudha) |
| ISO 9001:2015 clauses | 8.3.2, 8.3.4, 8.3.6, 8.5.6 |
| Governing procedures | SOP-05 (design), SOP-06 (change) |
| Reviewed | 2026-10-07 — Alexander Peter (EchoMind Project Lead); QA Sheryl Nazareth |

---

## 1. Purpose

One row per significant design item or change: what was planned, what review it received, and what
verification closed it. ISO 9001:2015 requires records of design planning (8.3.2), design controls
(8.3.4) and design changes (8.3.6).

## 2. What must be registered

Not every commit. Only changes meeting the significance criteria in SOP-06 §5 — in summary, anything
that changes intended behaviour, the retrieval or generation path, the data model, the deployment
topology, authentication or data handling, a model, or a customer-visible interface.

Routine changes remain controlled by Git alone; the commit is the record.

## 3. Register

| Ref | Date raised | Class | Title | Design inputs / acceptance criteria | Review (date, FRM-01 ref) | Verification evidence | Implemented (commit) | Status |
|---|---|---|---|---|---|---|---|---|
| DC-2026-001 | 2026-10-06 | Runtime configuration | Application login enabled on the reference instance (`AUTH_ENABLED=1`, `VOICE_AUTH_ENABLED=1` in `.env`) | Correction for NC-2026-009 | Not reviewed — single developer (REG-03 R-03) | TC-EM-AUTH-101 (external 401, voice 403/101) | `.env` (untracked) | Done |
| DC-2026-002 | 2026-10-06 | Code — defect fixes | Fixes for NC-2026-012…015 (token revocation, upload allow-list, small-talk routing, export key formats) | Acceptance criteria AC-101.7, DOC-102.5, RAG-103.3, EXP-108.2 | Not reviewed — single developer | Backend 50/50, voice 131/131, tsc 0; acceptance run 4 all pass | `e0ed4d0`, `3feb5db` | Done |
| DC-2026-003 | 2026-10-06 | Release | Release v1.4.1 (tag on `f1b8445`) | FRM-03 §3 checks | FRM-03 authorisation pending | `records/releases/2026-10-06_v1.4.1.md` | `f1b8445` | Released — authorisation pending |
| DC-2026-004 | 2026-10-06 | Runtime configuration | Voice speech recognition moved to the CPU (`VOICE_CUDA_VISIBLE_DEVICES=`, `VOICE_ASR_DEVICE=cpu`, `VOICE_ASR_REQUIRE_CUDA=0`) | GPU shared with another stack; voice could not create a CUDA context (22 restarts) | Not reviewed | Voice healthy first attempt; AC-101.9 pass | `.env` (untracked) | Done — revert when the GPU is not shared |
| DC-2026-005 | 2026-10-06 | Configuration | Ollama embeddings on the CPU (`OLLAMA_CUDA_VISIBLE_DEVICES` in compose; empty in `.env`) | Correction for NC-2026-016 | Not reviewed | Embedding call 200 (768 dims); acceptance run 4 all pass | `docker-compose.yml` + `.env` | Done — revert when the GPU is not shared |
| DC-2026-007 | 2026-10-07 | Process / tooling | Runtime-settings baseline, drift check (`scripts/check_runtime_config.sh`), SOP-06 §10.1; CI and external login check workflows; evaluation corpus pre-flight; real Ollama healthcheck | Corrective actions for NC-2026-008, -009, -010, -016 | Not reviewed — single developer | Drift check: match → exit 0, simulated login-off → exit 1; pre-flight aborted listing 15 missing documents; healthcheck exit 0 / broken model exit 1; CI and gate-check runs on GitHub | this commit | Done |
| DC-2026-006 | 2026-10-06 | Release identification | `BUILD_VERSION`/`BUILD_COMMIT`/`BUILD_DATE` held in `.env` for the running images | ISO 8.5.2 | Not reviewed | TC-EM-REL-106 | `.env` (untracked) | Done |

Use `forms/FRM-01_Design_Review_Record.md` for the review and `forms/FRM-02_Change_Request.md` for
the change, then summarise here.

## 4. Current state

Empty. Historically, design and change control has been exercised through Git alone — and the commit
bodies are unusually good, several recording rationale, root cause and verification. What they do not
record is a design review as a distinct act with participants and a decision, or acceptance criteria
agreed *before* the work. That is the gap this register closes (Gap Analysis G-16, G-18).
