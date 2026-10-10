# EchoMind Law — Phase 4 lawyer review rubric

A practising lawyer scores what the two Phase 4 features that use the AI model produced on their **held-out** sets, which were never used to tune them:
- summaries (S4.2);
- the review panel (S4.5).

Intake (S4.1) and time entries (S4.3) follow fixed rules, so they are checked by tests and the browser journeys rather than scored here. Every item is fictional.

- **Packet:** `eval/law/lawyer_review/make_packet.py --phase 4` builds it in `eval/law/lawyer_review/phase4/`.
- **Scores:** recorded in `phase4/sheet.csv`.
- **Scoring:** `eval/law/lawyer_review/score.py --phase 4 phase4/sheet.csv` computes the results.

## Who scores, and how

- **Reviewer:** one lawyer who did not see the features being tuned. A second lawyer scores a random 20% of the items, so that agreement between the two can be reported.
- **Groups:** each group shows the documents, then what the feature produced on them. The groups are shuffled.
- **Missed points:** each group has its own row (`G01`, …) for anything the feature missed: a key point a summary should carry, or an angle the panel should have raised.
- **Time:** about 60 minutes.

## Scales

### Summary points (each point, with its citation and quote)

| Score | Meaning |
|---|---|
| 2 — Keep | Accurate, cited to the right place, and worth having in a summary. |
| 1 — Not needed | Accurate, but not worth a lawyer's attention. |
| 0 — Wrong | Inaccurate, overstated, or cited to words that do not support it. |

### Panel perspectives (each of the four answers, as a draft)

| Score | Meaning |
|---|---|
| 3 — Useful as is | Accurate from its angle, every point supported by the documents it cites. |
| 2 — Useful with edits | Right in substance; something to tighten or a citation to fix. |
| 1 — Misleading | A statement is wrong or overstated, or the angle is lost. |
| 0 — Harmful | Would lead a lawyer to a wrong conclusion if relied on. |

### The moderator's matrix (each row)

| Score | Meaning |
|---|---|
| 2 — Faithful | The positions match what the four answers say. |
| 1 — Loose | Roughly right, but a position is put too strongly or too weakly. |
| 0 — Misleading | A position or the disagreement mark does not match the answers. |

### The stop decision (one per panel)

Score **1** if the panel was right to stop, or right not to, given what the documents leave open; otherwise **0**.

## What the gate needs

These are proposed thresholds for the owner to confirm:

| Measure | Proposed bar |
|---|---|
| Summary points scored 0 | 0 |
| Summary points scored 2 | ≥ 80% |
| Panel perspectives scored 2 or 3 | ≥ 75% |
| Panel perspectives scored 0 | 0 |
| Matrix rows scored 0 | 0 |
| Stop decisions right | all |
| Agreement between the two lawyers (same score ±1) | ≥ 80% |

A measure below its bar is reported with its items, never hidden. The owner then decides whether the feature stays off, or goes to the pilot with that limit stated.
