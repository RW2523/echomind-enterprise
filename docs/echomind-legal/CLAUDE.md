# Claude Code — EchoMind Legal instructions
Read `README.md`, `01_PRODUCT_PRD.md`, `02_UX_DESIGN_SYSTEM.md`, `03_INTERACTIVE_COMPONENTS.md`, `05_SYSTEM_ARCHITECTURE.md`, and `10_SECURITY_AND_LEGAL_SAFETY.md` **before implementation**. Remaining documents are authoritative for their areas.

## Goal
Extend existing EchoMind, not build an unrelated showcase: an **offline, chat-first, matter-aware legal AI** for a single DGX Spark with progressive disclosure and secure, evidence-grounded interactions.

## Never skip
1. Inspect repo, document existing services, verify GPU/model/deployment, identify reusable modules before changing code.
2. Work in phases described in `11_IMPLEMENTATION_ROADMAP.md`; implement **one end-to-end vertical slice at a time**, including backend authorization, UI, fixtures, tests.
3. Use TypeScript strict mode, typed API contracts, server-validated component schemas; never use `eval`, dynamic JSX from model output, arbitrary HTML or remote CDNs.
4. Every document chunk, graph relationship, cache record and background job is scoped by tenant and matter policy. **Authorize before retrieval**, and again on object access/action.
5. Legal assertions require exact, click-through document provenance; treat citations as untrusted until verified. Unanswerable requests should abstain.
6. Do not create fake case law or fake compliance claims. Demo content must always be labeled synthetic.
7. All destructive/export/external-send/filing actions require server-side policy checks and explicit human confirmation. Hearing deadlines are never set solely by LLM extraction.
8. Meeting/listening features are OFF by default, clearly opt-in, with consent and recording state UX. No hidden ambient recording.
9. Preserve offline runtime: no external inference, telemetry, cloud fonts, or dependency on network after setup.
10. Prefer **polished, minimal workflow UI** over adding module navigation: home displays only context, query composer and four subtle suggestions; initial answers show at most three prioritized items.
11. Write tests for every security-sensitive path, including cross-matter leaks, injected-document instructions, citation mismatch, audit events and backup/restore.
12. Verify ARM64 images, memory use, CPU/GPU contentions; background jobs use a bounded scheduler.

## Working method
Before each ticket, output: existing code touched, acceptance criteria, migration impact, risks. Implement small PR-size increments. Run tests and note failures honestly. Keep `IMPLEMENTATION_STATUS.md` current using statuses `Not started`, `In progress`, `Tested`, `Blocked` and test evidence. Never mark conceptual behavior as implemented without execution.

## MVP order
1. Audit current EchoMind repo and set up local dev/test fixtures.
2. Matter data model and policy-guarded API.
3. Matter-aware retrieval and verified source spans.
4. Chat shell and trusted dynamic UI cards.
5. Matter documents/timeline and discrepancy workflow.
6. Draft review/export, logging, backups/restore.
7. Optional speech + council only after acceptance gates.

## Response examples
- Short question: one sentence + one source pill.
- Contradiction: balanced two-source compare card, confidence/limitations, `Inspect`, `Mark reviewed`.
- Document draft: open right-side editor, citations sidebar, `Save draft`, `Request approval`; never auto-file.
- Ambiguous instruction: one contextual form with <=3 questions and default selections.

## UX sign-off
No empty decorative buttons, no nonfunctional mic controls, no oversized dashboard, no mystery-agent jargon, no fake "verified" badges. Ensure keyboard navigation, screen reader labels, high contrast, responsive narrow display, and persistent recording status.
