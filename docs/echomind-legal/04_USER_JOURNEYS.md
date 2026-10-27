# 04 — Screens and user journeys
## Navigation: only four main destinations
Chat (home), Matters (case list and detail), Drafts (in-progress work), Activity (jobs, tasks, audit-visible history). Settings under profile. Council, speech, research, graph and documents are **contextual experiences**, not separate top navigation modules.

### Journey A: first launch / onboarding
1. Device administrator creates local firm, admin login, password/MFA policy, encryption key handling, restore policy.
2. Authorized firm administrator creates users, roles and matter ethical walls.
3. Attorney creates matter, names jurisdiction and matter type, assigns team (no default global access).
4. Upload sources -> pending -> parsed -> indexed -> verified available; show errors clearly.
5. First assistant prompt: "Summarize this matter from available sources." Answer with citations and known gaps.
Acceptance: cannot see other firm's matter at any step; can open original from answer.

### Journey B: morning "What matters today?"
- Entry: home chat, matter context selector.
- Response: one-sentence briefing and up to three items: new document, potential discrepancy, upcoming **unverified** schedule event.
- Interact: choose one insight, expand evidence, mark for follow-up, assign task.
- Continue: "Draft questions from this." context carries matter + selected sources.
- Empty state: "Nothing new in the authorized records since last review" plus last sync time.

### Journey C: legal research and exact evidence
1. Ask a fact or argument question.
2. Backend enforces ACL, retrieves hybrid passages and verifies exact spans.
3. Answer separates `Supported from file`, `Inference`, `Missing source`.
4. Clicking citation opens original file with page/paragraph highlight, surrounding text and file hash info if appropriate.
5. `Compare with...` opens comparison card, only if both sides source-backed.
6. Lawyer can bookmark/source pin; source pin remains bounded by current matter authorization.

### Journey D: anomaly detection
1. Upload new witness statement, task queues extraction and candidate comparison.
2. Candidate findings ranked with reasons. Do not imply fraud.
3. Card has two sourced snippets, date/claim differences and `View original`, `Request clarification`, `Explain difference`, `Dismiss`.
4. Attorney resolves with annotation; retain original finding and reviewer decision in audit.
5. Next analysis respects verified amendment/supersession records.

### Journey E: drafting
1. Ask "Draft client update" while in a matter.
2. Minimal clarification: audience, time range, selected template (sensible defaults, <=3 items).
3. Create draft from source-backed facts and scoped search; label unsupported sections.
4. Expand editor on right; track draft version and diff; user edits or accepts proposed patch.
5. Review checklist: facts, citations, privilege, recipients, formatting. DOCX and PDF only after user-selected action; PDF should not be the main editable format.
6. Export is audited and may require policy approval; no automatic send.

### Journey F: listening session
1. Select matter and invite approved participants; explicit consent workflow and visible recording indicator.
2. User presses **Start recording**; OS capture begins only after authorization. `Pause` stops capture, not merely UI animation.
3. Live transcript streams via WebSocket; sensitive snippets are ephemeral until retention policy allows persistence.
4. Silent Assistant displays at most two cards; "Unverified" means no supporting record, not that speech is false.
5. Stop -> review transcript -> choose save/redact/discard; only saved approved data becomes searchable.
6. Device failure/reconnect: show status and last known retained segment; never silently resume after pause.

### Journey G: Council (later phase)
- Lawyer asks "Challenge my settlement argument" and selects issue/scope.
- Front card shows 3 sections: agreement, key objections, missing evidence.
- Expand to agent perspectives; each has exact evidence refs, claims, open questions; forbid invented percentages.
- Structured review concludes "options for attorney consideration", not AI legal verdict.
- Can turn any issue into a follow-up task or draft memo with lawyer approval.

### Journey H: graph
- Start from the question, not an empty network tab. Example "Who was involved in the May amendment?"
- Render <=12 relevant nodes, citations on edges, confidence and verified/claimed status.
- Click entity -> context card -> click source -> inspect original. Expand 1 hop at a time.

### Journey I: proposed time entries
- Evidence: explicit timer/approved work logs; show `suggested` and attributable activity only.
- Reviewer corrects duration, task category, matter and duplication. Never infer time spent solely from number of AI actions.
- Submit to billing only through approved connector with explicit user confirmation.

## Reusable layout wireframes (text)
**Empty chat:** top bar `EchoMind | [Matter selector] | Private | Not recording`; center `How can I help?`; composer; 4 chips; collapsed rail.
**Answer with issue:** chat bubble -> short takeaway -> single source comparison card -> actions (Inspect/Save/Ask follow-up); side panel opens only after inspect.
**Document editing:** 32% left chat / 68% right editable document, or user-resizable; mobile switches to document view with back-to-chat.
**Live:** prominent 'Recording' pill + stop button; transcript center; quiet single alert below; never cover active consent control.
