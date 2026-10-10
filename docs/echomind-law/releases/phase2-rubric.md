# EchoMind Law — Phase 2 lawyer review rubric (S2.8)

A practising lawyer scores what the system produced on the **held-out** sets, which were never used
to tune it. Every item is fictional. Scores are recorded in the review sheet
(`eval/law/lawyer_review/sheet.csv`) and computed by `eval/law/lawyer_review/score.py`.

## Who scores, and how

- **Reviewer:** one lawyer who did not see the system being tuned. A second lawyer scores a random
  20% of the items, so agreement between them can be reported.
- **Order:** items are shuffled. The reviewer does not see which suite or model setting produced an item.
- **Time:** about 2 hours for the full packet.
- **The question is always:** "Would I rely on this, as written, in front of a client, after the
  checks the screen tells me to make?"

## Scales

### Research answers (RAG-01) — each answer with its evidence

| Score | Meaning |
|---|---|
| 3 — Reliable | Correct, complete for the question, and every statement is supported by the quote it cites. |
| 2 — Usable with edits | Correct in substance; wording, completeness or a citation needs a small fix. |
| 1 — Misleading | A statement is wrong, overstated, or cited to text that does not support it. |
| 0 — Harmful | Would lead a lawyer to a wrong legal conclusion if relied on. |

Also mark, yes or no:
- **Abstained correctly:** the documents do not answer, and the answer says so without adding unsupported content.
- **Version right:** the answer used the signed version, or warned where the draft differs.

### Comparisons and conflicts (RAG-02) — each explained change or conflict

Use the same 0–3 scale. A 3 means the change is real, labelled correctly, and both wordings are quoted exactly.

### Silent Assist alerts (LIVE-02) — each alert, in context of the call

| Score | Meaning |
|---|---|
| 2 — Useful | I would want to see this alert during the call. |
| 1 — Noise | Correct, but not worth interrupting a lawyer for. |
| 0 — Wrong | The alert is mistaken, or its wording overstates the evidence. |

Also list any moment where an alert **should** have been raised and was not.

### Minutes (MEET-01) — each meeting's draft

Score each item 0–3 as for research answers. Then score the draft as a whole:
- **Complete:** every decision, action (owner and deadline) and open point is there: yes or no, listing what is missing.
- **Invented:** anything not said: yes or no, quoting it.

## What the gate needs

These are proposed thresholds for the owner to confirm (P-008 applies to the Silent Assist thresholds):

| Measure | Proposed bar |
|---|---|
| Research answers scored 0 (harmful) | 0 |
| Research answers scored 2 or 3 | ≥ 85% |
| Correct abstentions | ≥ 2 of 3 held-out abstention items |
| Comparisons scored 2 or 3 | ≥ 85% |
| Alerts scored 0 | ≤ 1 per call |
| Minutes with anything invented | 0 |
| Minutes items scored 2 or 3 | ≥ 85% |
| Agreement between the two lawyers (same score ±1) | ≥ 80% |

A measure below its bar is not hidden: it is reported with the items, and the owner decides whether
the pilot goes ahead with that limit stated, or the slice is reworked.
