# 11 — Implementation roadmap: slices for Claude Code
Each task should be a reviewable change with migrations, UI, API, tests, demo fixture and update to implementation status. **Do not make decorative nonfunctional UI**. No fixed timelines given because repo and appliance benchmark are not yet inspected.

## Milestone 0: reconnaissance
- Audit EchoMind services, source tree, current APIs, model runtime, GPU memory, retrieval quality, Docker/compose, security and tests.
- Produce `IMPLEMENTATION_STATUS.md`, `CURRENT_ARCHITECTURE.md`, gap matrix: reuse/refactor/new.
- Verify current model from runtime, not stale `docs/CAPABILITIES.md`.
**Done** when local stack runs and baseline documented with actual commands/output.

## Milestone 1: secure matters
- Tenant/user/matter/membership schemas, admin grants, ethical wall, policy module.
- `/v1/me`, `/v1/matters`, consistent no-access behavior; role tests.
- Synthetic data seed loader.
**Done** when forbidden matter cannot be accessed via API, query, storage ID, autocomplete or logs.

## Milestone 2: legal ingestion + citation
- Original file hash/version and safe parser; page anchors and exact-span links.
- Scope hybrid retrieval by authorization before scoring; quote/citation verifier and abstention.
- Source viewer that opens original page.
**Done** when 20+ gold questions yield inspectable sources and unsupported quotes are visibly withheld.

## Milestone 3: chat-first polished UI
- Implement layout tokens, minimal shell, matter selector, accessible composer, streaming.
- Safe component registry with `source_answer`, `priority_list`, `source_chip`, `evidence_comparison`, `checklist`, `task_progress`.
- Real endpoints for inspect/review actions; responsive behavior and keyboard support.
**Done** when a user finishes two evidence tasks from one conversation without navigating to a new module.

## Milestone 4: case intelligence
- Matter file list, chronology extraction with source links, evidence comparison and human finding resolutions.
- Filter changes by matter and event time, with uncertain dates labeled.
**Done** when user uploads contradictory sample statements and can inspect, resolve and audit issue.

## Milestone 5: document drafting
- Template management, provenance-linked draft editor, version conflict checks, approval and DOCX/PDF export.
- Review checklist and audit; no automatic final delivery.
**Done** when fictional matter produces a source-backed client memo and DOCX editable outside app.

## Milestone 6: readiness
- Backup+restore, key recovery procedures, recording feature disabled until consent, incident and offline tests.
- Performance/queuing on actual DGX Spark, vulnerability and legal content review.
**Done** when restore passes and no critical auth/privacy vulnerabilities remain.

## Milestone 7: optional differentiation
- Opt-in live listening with consent and verified cards; GraphRAG source-linked 1-hop; Council bounded independent roles; MCP allowlisted read-only tools. Separate acceptance gates and benchmarks for each.

## Definition of done for each ticket
Actual running UI, meaningful tests, no invented data in production, backend permission checks, source refs, error/loading states, accessibility, responsive layout, offline deploy, documentation, regression safety, observed performance. All incomplete features clearly gated by flags.

## Suggested sprint ticket order
`LEGAL-001 repo audit`, `002 schema`, `003 RBAC`, `004 source ingestion`, `005 citation resolver`, `006 search filter`, `007 answer verifier`, `008 chat shell`, `009 UI component contracts`, `010 cards and panel`, `011 chronology`, `012 discrepancy resolution`, `013 draft editor`, `014 export approval`, `015 backup and restore`, `016 security/prompt injection suite`, `017 offline acceptance`, `018 advanced agents`.
