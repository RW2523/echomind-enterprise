# EchoMind Law — Phase 3 lawyer review rubric (S3.9)

A practising lawyer scores what the document tools produced on the **held-out** sets. Those sets were
never used to tune the tools. Every item is fictional.

- **Packet:** `eval/law/lawyer_review/make_packet.py --phase 3` builds it in `eval/law/lawyer_review/phase3/`.
- **Scores:** recorded in `phase3/sheet.csv`.
- **Scoring:** `eval/law/lawyer_review/score.py --phase 3 phase3/sheet.csv` computes the results.

## Who scores, and how

- **Reviewer:** one lawyer who did not see the tools being tuned. A second lawyer scores a random 20% of
  the items, so that agreement between the two can be reported.
- **Groups:** the packet is in groups. Each group shows the documents a tool was run on, then everything
  the tool reported on them. The groups are shuffled.
- **Missed points:** each group also has its own row in the sheet (`G01`, `G02`, …) for anything the tool
  **missed**.
- **Time:** about 90 minutes.
- **The question each time:** "Would I rely on this, as written, after the checks the screen tells me to
  make?"

## Scales

### Drafts (DOC-01): each draft from a firm template

Each draft is shown with its fields, their quotes and the passages the model wrote.

| Score | Meaning |
|---|---|
| 3 — Ready after the marked checks | The fields are right, the written passages are accurate and supported, and nothing needs rewriting. |
| 2 — Usable with edits | Right in substance; wording, a repeated sentence or a citation needs a small fix. |
| 1 — Misleading | A field or sentence is wrong or overstated, or is cited to text that does not support it. |
| 0 — Harmful | Would mislead the client or the other side if sent. |

In `wrong_or_invented`, write **invented** (with the words) if the draft states anything the documents
do not.

### Playbook checks (DOC-02): each clause checked against the firm's position

Use the same 0–3 scale.
- A 3 means the verdict (standard, fallback, outside, not found, unclear) is right and the reason is
  accurate.
- A **0** means a clause outside the firm's position was passed as standard or fallback.

### Findings: citations, definitions and references, obligations, due diligence, governance (DOC-03 to DOC-07)

| Score | Meaning |
|---|---|
| 2 — Useful | Correct, and I would want a lawyer to see it. |
| 1 — Noise | Correct, but not worth a lawyer's time (for example, it repeats another finding). |
| 0 — Wrong | The finding is mistaken, or its wording overstates the documents. |

- **Obligations:** if the due date is wrong, write **date** in `wrong_or_invented`. A date left for a
  person is not wrong.
- **Due diligence:** the missing-documents list is scored like a finding. It is a 2 if the list is right
  for the checklist.

### What was missed: one row per group

In `missed`, write **yes** or **no**. If yes, list in `notes` what a careful lawyer would have flagged
and the tool did not.

## What the gate needs

These thresholds are proposals for the owner to confirm:

| Measure | Proposed bar |
|---|---|
| Drafts scored 2 or 3 | ≥ 80% |
| Drafts with an invented fact | 0 |
| Playbook lines scored 2 or 3 | ≥ 85% |
| Playbook lines scored 0 (a deviation passed) | 0 |
| Findings scored 0 (wrong) | ≤ 5% |
| Findings scored 2 (useful) | ≥ 70% |
| Obligation dates wrong | 0 |
| Groups with something missed | reported, with the list |
| Agreement between the two lawyers (same score ±1) | ≥ 80% |

A measure below its bar is not hidden. It is reported with the items. The owner then decides whether
the pilot goes ahead with that limit stated, or whether the slice is reworked.
