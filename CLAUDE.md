# Claude Code — EchoMind Legal

EchoMind Legal is the chat-first ("Quiet Counsel") workspace for law firms, built **on the EchoMind Law edition**
(`feature/echomind-law`) on a single NVIDIA DGX Spark. The product blueprint is in [`docs/echomind-legal/`](docs/echomind-legal/README.md);
its original Claude instructions are in [`docs/echomind-legal/CLAUDE.md`](docs/echomind-legal/CLAUDE.md) and apply here in full.
The Law programme's own records (decisions, threat model, test runs) are in [`docs/echomind-law/`](docs/echomind-law/README.md).

## Read first
1. `docs/echomind-legal/IMPLEMENTATION_STATUS.md` — what is built and tested, with evidence; keep it current.
2. `docs/echomind-legal/GAP_ANALYSIS.md` and `CURRENT_ARCHITECTURE.md` — what exists, what is reused, what is new.
3. Blueprint `README.md`, `01`, `02`, `03`, `05`, `10` before changing behaviour; the rest for their areas.

## Facts about this repository (verified 2026-10-26)
- **Production shares this host** (echomind-ajace.com, `main` v1.4.1). Never run `docker compose` in a clone of this repo except
  through `scripts/law/compose.sh` (project `echomind-law`). Never touch production containers or volumes.
- The Law pilot was **stopped at the owner's request on 2026-10-19**; data, keys and backups are kept. Restarting it
  (`scripts/law/deploy_pilot.sh`) is the owner's decision.
- Memory is the constraint: ~11 GB available of 121 GB (other stacks run here). Add no resident model or service.
- Chat model: `nvidia/Qwen3-30B-A3B-FP4` on TensorRT-LLM (`:8355`); embeddings `nomic-embed-text` on Ollama (`:11435`); both shared.
- The repository is **public**: no real names, firm data or secrets; demo content is labelled `DEMO / SYNTHETIC`.

## How to test
```bash
# backend (pinned Law image, no network) — from the repo root
docker run --rm --network none --entrypoint "" -v "$PWD":/repo -w /repo/backend -e ECHOMIND_DATA_DIR=/tmp/emd \
  echomind-law-backend:latest python -m pytest tests -q
# frontend
cd frontend && npx tsc --noEmit -p . && npx tsc --noEmit -p tsconfig.counsel.json && npx vitest run
```
The pinned image has no LibreOffice, so `test_t09_the_released_word_file_reads_back_in_libreoffice…` fails locally; CI installs it.

## Where EchoMind Legal lives
- Backend: `backend/app/legal/counsel/` (conversations, orchestrator, trusted component schemas, citations, discrepancies,
  briefing), migration `0036_counsel`, tests `backend/tests/law/test_law_counsel.py`.
- Frontend: `frontend/law/counsel/` (shell, chat, component registry, cards, source panel), tests `frontend/law/test/counsel*.test.tsx`.

## Rules that never bend
Authorize before retrieval and again on every object access or action; scope everything by matter; the model never writes UI
(server-validated components from the registry only); citations are untrusted until re-verified; abstain when there is no
source; no fake functionality, no fake "verified" badges; human approval for anything that leaves the matter; recording off by
default with real capture state; offline runtime (no CDN, telemetry or cloud calls). Report test results honestly.
