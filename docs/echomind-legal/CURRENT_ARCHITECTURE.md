# EchoMind Legal — current architecture (as found, 2026-10-26)

Evidence-based description of what exists **before** the EchoMind Legal (blueprint) work, gathered from
the repository and the running appliance on 2026-10-26. Nothing here is taken from marketing documents.
Commands and their output are quoted where they matter.

## 1. Two code lines in one repository

| Line | Branch | What it is | Running? |
|---|---|---|---|
| EchoMind Enterprise 1.4.1 | `main` (f2e4b66) | General-purpose assistant: Knowledge Chat (hybrid RAG), live transcript + Silent Assistant, voice bot, Boardroom, Document Studio, vertical "packs" (health/law/…) that share **one** global index | **Yes** — production at echomind-ajace.com (`docker ps`: echomind-backend, -frontend, -voice, -trtllm, -ollama, -cloudflared) |
| EchoMind Law | `feature/echomind-law` (024d5e1, 245 commits ahead of the merge base) | A separate **Law edition** of the same code base (`ECHOMIND_EDITION=law`): firms, matters, memberships, ethical walls, encrypted per-matter storage, per-matter search partitions, verified quotes, comparison, drafting, exports with approval, consent-gated meetings, hash-chained audit, encrypted backup/restore | **Stopped** on 2026-10-19 at the owner's request ("Stop this application fully end to end"). Data volume `echomind-law_echomind_law_data`, keys and backups are kept (`~/echomind-law-backups/law-pilot-20261019-181121.emlb`) |

EchoMind Legal is built **on the Law line** (branch `feature/echomind-legal-chat`, cut from `origin/feature/echomind-law`),
because the Law line already implements most of the blueprint's P0 backend with tests. Rebuilding it on `main`
would violate the blueprint's "do not rebuild working components".

> The blueprint's "current baseline" paragraph describes `main` (v1.4.1). It is stale for the Law line: for example
> "Backups not operational" and "consent step missing" are true of `main`, but the Law line has encrypted backup/restore
> (`app/legal/backup`, DR drill) and per-participant consent gating (SEC-05).

## 2. Hardware (measured on the appliance, 2026-10-26)

| Item | Value | Command |
|---|---|---|
| CPU arch | `aarch64` | `uname -m` |
| OS | Ubuntu 24.04.4 LTS | `/etc/os-release` |
| GPU | NVIDIA GB10, driver 580.159.03, CUDA 13.0; memory "Not Supported" (unified) | `nvidia-smi` |
| Memory | 121 GiB total, **110 GiB used, 11 GiB available**, swap 15 GiB of 15 GiB used | `free -h` |
| Disk | 3.7 TB, 2.2 TB free | `df -h /` |

Memory is the binding constraint. The host also runs a separate 22-container "cio" stack (two vLLM engines, Temporal,
Grafana) and production EchoMind. The Law stack's `compose.sh` refuses to start below 12 GB available
(`LAW_MIN_FREE_GB`), each Law container is memory-capped (backend 8 GB) and marked first to be killed
(`oom_score_adj: 800`). The blueprint's 128 GB assumption holds for the device, but **only ~11 GB is free for EchoMind Legal today**.

## 3. Runtime services (Law edition)

```
Browser (React 19 + Vite, react-router, TanStack Query; fonts bundled via @fontsource)
  └─ nginx (frontend/nginx.law.conf, TLS)  ── /api/* ──►  FastAPI backend (one process, ECHOMIND_EDITION=law)
                                                         ├─ identity: sessions, CSRF, TOTP MFA, lockout, OIDC (off), demo sign-in (off)
                                                         ├─ policy.py: deny-by-default matter/firm actions, walls, break-glass
                                                         ├─ documents: AES-256-GCM blobs (master key outside the volume), OCR,
                                                         │   structure (pages, lines, clauses, defined terms), versions
                                                         ├─ retrieval: AuthorizedScope → per-matter partition (FAISS exact IP + BM25)
                                                         ├─ research: NDJSON stream, per-paragraph quote verification, abstention
                                                         ├─ compare / drafting / exports / meetings / records / reviews / …
                                                         ├─ audit: hash-chained append-only trail, daily anchors, verifier
                                                         └─ jobs: one daemon thread over legal_jobs (restart-safe)
                                                              │
                         host.docker.internal:8355  ◄─────────┤  TensorRT-LLM (production container, shared, read-only use)
                         host.docker.internal:11435 ◄─────────┘  Ollama nomic-embed-text (production container, shared)
voice service (Piper TTS; speech recognition on CPU in the Law stack)
```

- **Chat model:** `nvidia/Qwen3-30B-A3B-FP4` served by `trtllm-serve` (`curl :8355/v1/models` → `"owned_by":"tensorrt_llm"`), thinking disabled.
- **Embeddings:** `nomic-embed-text` via Ollama (`:11435/api/embeddings` answers).
- **Reranker:** sentence-transformers cross-encoder `ms-marco-MiniLM-L-6-v2` on CPU (`app/rag/reranker.py`).
- **Store:** SQLite (WAL) in `/data`, Law schema owned by versioned migrations `app/legal/migrations` (0001–0035, up/down, checksummed).
- **Isolation:** `docker-compose.law.yml` gives every Law service its own name, ports and volume, never starts a second LLM or
  embedding server, never starts the public tunnel (THREAT_MODEL T-17).

## 4. Retrieval path (what "case-specific RAG" means here today)

1. `scope.authorize(principal, matter_id)` asks `policy.authorize(..., "research.query")`; only it can mint an `AuthorizedScope`
   (constructor raises `TypeError` otherwise).
2. `search.search(scope, query)` re-checks the scope, searches **only that matter's partition** (separate FAISS + BM25 files per
   matter, rebuilt from the DB if the files name another partition), fuses dense + keyword + exact, reranks, applies the
   version policy (signed vs current) and per-document source permissions, then re-checks the scope.
3. `research/routes._answer` streams `sources`, then each paragraph **after** its quotes are verified against the cited passage
   (`evidence.verify` → verbatim / normalized / failed / not_quoted), then `done` with status
   `answered | partial | unverified | abstained`, exact character offsets, version conflicts and the turn id. The scope is
   re-checked every `LEGAL_STREAM_RECHECK_SEC` while streaming (`scope_revoked` ends the stream).
4. Passages that address an AI ("ignore previous instructions…") are flagged and withheld from the model's copy (T08).

`main`'s global index (`app/rag/index.py`) ranks globally and post-filters by namespace (`_ns_ok`) — the pattern the
blueprint forbids. The Law line does not use it for matters (S1.7, D-025 closes every legacy module in the Law edition).

## 5. Frontend (Law edition)

`frontend/law/` — 50 routes under one glass sidebar ("Paper & Glass", D-019): firm entries (Home, Matters, Reviews, Templates,
Playbooks, Checklists, admin pages) and ~15 per-matter tools (Overview, Documents, Research, Conversation, Compare, Drafting,
Obligations, Diligence, Meetings, Exports, Members, Activity…). Research is a page, not a chat; its design explicitly avoids
chat bubbles. Typed API client (`law/api/client.ts`, NDJSON streaming helper), session/matter contexts, Vitest + Testing Library
(37 files, 184 tests) and a 36-journey Playwright suite.

## 6. Tests (baseline on this branch before any change)

| Suite | Command | Result (2026-10-26) |
|---|---|---|
| Frontend type-check | `cd frontend && npx tsc --noEmit -p .` | clean |
| Frontend unit | `cd frontend && npx vitest run` | **184 passed** (37 files) |
| Backend (all) | pinned image, `--network none`: `python -m pytest tests -q` | **1402 passed, 3 skipped, 1 failed** (LibreOffice is not in the image; CI installs it) |
