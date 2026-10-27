# 02 — UI / UX design system
## Concept: "Quiet Counsel"
Visual style: calm professional software, not legal clip-art or neon AI. Default: warm off-white `#F7F6F2`, ink `#17263D`, neutral `#667589`, white `#FFFFFF`, muted gold `#C29A55` used sparingly, alert amber `#B7791F`, error red `#B23B3B`, confirmed green `#277A5C`. Meet WCAG AA contrast: adjust actual rendered combinations after testing.

Typography: locally bundled licensed sans (e.g., system fallback), 16px body, comfortable 1.5 line height; document reading mode uses a sober serif if licensed and locally packaged. 8px base spacing; radius 10–16px; subtle 1px borders; minimal shadows. No decorative AI gradients. Motion 120–180ms, honors prefers-reduced-motion.

## Desktop shell
- Left collapsible rail ~240px: New chat, History, Matters, Drafts, Activity; not dozens of tools.
- Center ~720–920px reading column: messages, inline interactive cards, composer pinned bottom.
- Right contextual work surface 380–520px only when user opens source/edit/graph/council; resizable; preserves conversation.
- Header: EchoMind logo, matter/context selector (clearly scopes searches), privacy/network status, visible audio status, profile.
- Start screen: `How can I help?` plus 4 quiet suggestions; no metrics wall.
- Mobile/narrow: left rail drawer; contextual panel full-screen sheet; bottom composer; no tiny side-by-side evidence (stack cards).

## State and response grammar
- Empty: one question and contextual actions.
- Loading: skeleton only where needed; accessible progress text; no fictional progress percent.
- Basic answer: <= 120 words by default, inline clickable source chips.
- Multi-step: up to 3 highest priority cards, `Show more` disclosure.
- Evidence disagreement: compare card with both quotes/dates and 'possible difference' not 'lie'.
- Missing information: 1–3 short controls, usually in chat, not a modal.
- High-stakes action: explicit review panel with what will happen and which records will leave appliance.
- Offline update/import operations: show jobs and progress in Activity (no unexplained spinner).
- Error: distinguish permission denied, file parse failure, unavailable model, unsupported document, unverified citation, and canceled task.

## Surface inventory
**Chat message:** role, content, optional provenance, status; citations clickable.
**Source drawer:** source title, page, surrounding text, highlight, original-download policy, verified original hash.
**Case overview:** matter name, parties, 1-line status, up to 3 insights, documents and chronology switcher.
**Timeline:** sorted factual/alleged events with source and unknown-date states; event click opens evidence.
**Diff/evidence comparison:** quotations or exact passage with context, source chips and attorney resolution options.
**Checklist:** item ownership, completion status, `suggested` vs `confirmed`, saved only upon authorized action.
**Draft editor:** edit text, source mapping, comments, document versions; export gated.
**Council:** first show summary: shared points / disputed points / missing evidence; expand panel for per-agent arguments and citations.
**Live listening:** large mic state and timer, transcript, max 1–2 quiet prioritized findings; pause and stop always visible.
**Graph explorer:** no giant hairball; start 5–12 nodes, expand neighbor on tap, visible legend and evidence source.

## UX microcopy / safety
Use "Potential inconsistency" not "Witness lied"; "Draft—needs review" not "Court-ready"; "No source found" not invented citations; "Not recording" must map to a physical/OS audio capture OFF state. If fallback model is activated, transparently show status, not inner implementation complexity.

## Interaction accessibility
Keyboard: Tab, shift-Tab, Enter/Space on card controls, Esc closes drawer, focus returns. Screen readers: aria-live on streamed answer summary, do not announce every token; source labels include title and page. A color badge is never the sole channel of status. Avoid auto-scrolling if user is reading prior context.

## Design review checklist
Does every element have a job? Is the answer visible before details? Could one card replace five? Are actions real? Is source access policy-checked? Is recording status unambiguous? Does it remain useful at 390px width? Can a person finish the intended task without knowing what an AI agent is?
