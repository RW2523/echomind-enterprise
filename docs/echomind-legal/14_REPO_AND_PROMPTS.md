# 14 — Repository plan and Claude Code prompts
## Suggested layout (adapt, do NOT bulldoze existing EchoMind repo)
```text
repo/
  CLAUDE.md
  docs/echomind-legal/*.md
  apps/web/src/features/legal/
    shell/ chat/ dynamic-ui/ matters/ source-viewer/ drafts/ activity/
  services/api/app/legal/
    auth/ policy/ matters/ documents/ retrieval/ citations/ findings/
    drafting/ workflows/ recordings/ audit/ schemas/
  workers/legal/
    ingestion/ indexing/ extraction/ verification/ background/
  infra/legal/
    docker-compose.yml env.example offline-images/ backup/
  tests/legal/
    fixtures/ unit/ integration/ e2e/ adversarial/ evals/
  scripts/legal/
    seed_demo.py verify_offline.sh benchmark_spark.sh restore_smoke.sh
```
Keep secrets out of Git. Local model data outside repo; immutable schema migrations. Avoid invented APIs that don't match existing services: inspect first.

## Claude prompt 0 — audit actual system
```text
Read CLAUDE.md and all docs in docs/echomind-legal. Audit the CURRENT repo before code changes. Identify current React/FastAPI apps, APIs, auth, retrieval, Qwen/TensorRT runtime, Docker, GPU assumptions, search indexes, document handling, test coverage and known blockers. Provide a reuse/refactor/new matrix and actual evidence from code paths. Create IMPLEMENTATION_STATUS.md. Do not change the product yet.
```

## Claude prompt 1 — matter-safe vertical slice
```text
Implement the first EchoMind Legal slice in the current repository: firm/matter/user RBAC, ethical wall policy, synthetic data seeding, list matters and view authorized matter metadata. Follow 06_DATA_MODEL_AND_PERMISSIONS, 09_APIS_AND_EVENTS and 12_TESTS_AND_ACCEPTANCE. Add migrations, REST API, typed React matter context, unit/integration/e2e tests including cross-matter denial. Report commands run and failures. No stubbed fake-success buttons.
```

## Claude prompt 2 — grounded chat and source viewer
```text
Extend existing RAG with matter ACL pre-filter, original document version+hash and exact citation matching. Implement source-answer chat component and source viewer that highlights original page/span. Must abstain for no source and prevent cross-matter leaks via vector cache, API and citation IDs. Add attack and evaluation fixtures. Preserve existing inference service, do not swap models without benchmark.
```

## Claude prompt 3 — polished chat-first UI
```text
Implement Quiet Counsel from 02_UX_DESIGN_SYSTEM and trusted dynamic components from 03_INTERACTIVE_COMPONENTS. Minimal home, selected matter context, streaming chat, one primary card, expandable evidence comparisons and actionable review checklist. Responsive, keyboard accessible, loading/error states. No dashboard noise, no fake functionality, no remote fonts/CDNs. Take UI screenshots via available local testing tools and refine.
```

## Claude prompt 4 — lawyer workflow
```text
Implement synthetic Morgan matter end-to-end: ingestion -> date discrepancy compare -> source viewer -> attorney explanation -> audited resolution -> source-linked client update draft in editable document viewer -> policy-gated DOCX export. Follow 04_USER_JOURNEYS, 13_DEMO_SEED_DATA and acceptance tests. Deliver one complete flow rather than independent modules.
```

## Claude prompt 5 — Spark readiness
```text
Audit the real DGX Spark (ARM64, unified memory, containers, model runtime). Introduce local bounded job queue with priority for live interactions, offline build/deployment, backup+restore smoke test, WAN deny test, and full demo benchmark. Log observed P50/P95 and peak RAM. Do not claim performance without measurements; keep advanced council disabled until resource tests pass.
```

## Review rubric after each coding session
`Working UX?`, `Permission enforced before search?`, `Citations verified?`, `Actions really persist?`, `On-prem offline?`, `Tests passing?`, `No high-risk automation?`, `Memory bounded?`, `Status accurately documented?`. If any answer no, fix before adding features.
