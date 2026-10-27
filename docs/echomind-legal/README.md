# EchoMind Legal — Private AI Workspace for Law Firms
**Build blueprint | 2026-10-26 | Target: a single NVIDIA DGX Spark | Status: proposed product specification**

> Product promise: **Ask naturally. See only what matters. Work with verified evidence. Keep everything private.**

A build-ready documentation kit for evolving EchoMind Enterprise v1.4.1 into a lawyer-first, on-premises AI workspace. **This pack is a design and implementation plan, not evidence that features already exist, that the legal product is production-ready, or that legal compliance has been achieved.** The sample matters, evidence, dates, and screenshots described in these files are fictional.

## Navigation / reading order
1. [`CLAUDE.md`](CLAUDE.md) — paste into project root; instructions for Claude Code.
2. [`01_PRODUCT_PRD.md`](01_PRODUCT_PRD.md) — complete vision, personas, stories, scope, priorities.
3. [`02_UX_DESIGN_SYSTEM.md`](02_UX_DESIGN_SYSTEM.md) — visual rules, chat-first architecture, responsive layouts.
4. [`03_INTERACTIVE_COMPONENTS.md`](03_INTERACTIVE_COMPONENTS.md) — dynamic UI schema, safe interaction specification.
5. [`04_USER_JOURNEYS.md`](04_USER_JOURNEYS.md) — end-to-end lawyer experiences and screens.
6. [`05_SYSTEM_ARCHITECTURE.md`](05_SYSTEM_ARCHITECTURE.md) — deployable services, boundary and data flows.
7. [`06_DATA_MODEL_AND_PERMISSIONS.md`](06_DATA_MODEL_AND_PERMISSIONS.md) — SQL-oriented entities, tenant isolation, sources.
8. [`07_RAG_GRAPHRAG_AGENTS.md`](07_RAG_GRAPHRAG_AGENTS.md) — retrieval, legal verification, anomalies, Council, MCP/LangGraph.
9. [`08_DGX_SPARK_DEPLOYMENT.md`](08_DGX_SPARK_DEPLOYMENT.md) — hardware constraints, scheduling, offline operations.
10. [`09_APIS_AND_EVENTS.md`](09_APIS_AND_EVENTS.md) — REST/SSE/WS contracts and examples.
11. [`10_SECURITY_AND_LEGAL_SAFETY.md`](10_SECURITY_AND_LEGAL_SAFETY.md) — threat model, privilege, recordings, export, approval.
12. [`11_IMPLEMENTATION_ROADMAP.md`](11_IMPLEMENTATION_ROADMAP.md) — Claude build tickets and phased delivery.
13. [`12_TESTS_AND_ACCEPTANCE.md`](12_TESTS_AND_ACCEPTANCE.md) — evaluation suites and release gates.
14. [`13_DEMO_SEED_DATA.md`](13_DEMO_SEED_DATA.md) — fictional demo matter, test scripts, UI journeys.
15. [`14_REPO_AND_PROMPTS.md`](14_REPO_AND_PROMPTS.md) — repository structure, CI and actionable Claude prompts.

## North-star experience
The lawyer opens one elegant chat. They can ask “What changed in the Morgan case?”, and EchoMind returns **three prioritized findings** and an evidence-first card—not a dashboard. On request, the card expands into a source viewer, chronology, guided questions, draft editor or Council review **without losing chat or matter context**.

## MVP, deliberately narrow
- Multi-user login + organizations/matters + server-enforced ACL; source document ingestion.
- Legal matter chat with citations, exact-span viewer, strict abstention and permissions.
- Minimal home/chat shell, contextual matter picker, dynamic **answer / source / comparison / checklist** cards.
- Matter workspace with documents and timeline, human-verifiable discrepancy review.
- Editable document drafts (DOCX/PDF) gated by attorney review; immutable source provenance.
- Audit, encrypted backup and restore, recording OFF by default.

**Not MVP:** 24-hour ambient recording, autonomous legal decisions/filings, unrestricted external web access, unsupervised MCP write actions, full-scale multi-model Council, untested deadline computation. Later phases specify these.

## Hard constraints
- On-prem/private means **no cloud calls, no telemetry, no CDN fonts/assets, no remote dependencies** in runtime. Updates are explicit offline packages.
- Never allow the LLM to render arbitrary HTML/JS. Backend sends schema-validated UI components, rendered from a trusted registry.
- Never claim cited legal cases exist without source verification; legal outputs are drafts for licensed professional review.
- Never mix matter data via semantic index, cache, graph, embeddings, summaries, logs or agents.
- Background reasoning may be scheduled **locally** while appliance is running, with visible status and opt-in controls; no promised future deliverables.

## Current EchoMind baseline (provided by product team; verify in repository)
Existing: React/Vite frontend, FastAPI services, hybrid RAG with Qwen3-30B 4-bit on TensorRT-LLM, Nemotron speech, Parakeet final transcript, VibeVoice meeting diarization, Piper TTS, document generation, roles and audit. Historical proof points and limitations supplied by team are *not* validated here. Determine actual running model, endpoints, GPU use and capabilities before coding. Preserve what works.

## Important dependencies
- `docs/CAPABILITIES.md` may be stale: update based on real repository evidence, not marketing claims.
- Hardware assumption: **128 GB DGX Spark** (verify whether this appliance is 64 or 128 GB; both variants exist). ARM64/aarch64 native images required.
- References: NVIDIA https://docs.nvidia.com/dgx/dgx-spark/hardware.html ; MCP https://modelcontextprotocol.io/ ; LangGraph https://docs.langchain.com/oss/python/langgraph/ . Verify specific release compatibility in target environment.
