# EchoMind Law — pilot additions lawyer review rubric ("phase 5")

A practising lawyer scores two features added to the pilot on 2026-10-12:
- **regulatory updates (D-068):** which new public law the rule matched to which kind of matter;
- **machine translation (D-070):** paragraphs translated by this server's model.

The other additions are checked by tests, the browser journeys and the pilot checks rather than scored here: the SIEM forwarder, backups, the network lock and the citation matching.

- **Packet:** `python3 eval/law/lawyer_review/make_packet.py --phase 5` builds `eval/law/lawyer_review/phase5/`, from:
  - the real bundle of 12 October 2026 (`eval/fixtures/law/regulatory/`, real public law);
  - the newest `*-translation-review.json` (the fictional corpus, translated into French, German and Spanish).
- **Scores:** recorded in `phase5/sheet.csv`.
- **Scoring:** `python3 eval/law/lawyer_review/score.py --phase 5 phase5/sheet.csv`.

## Who scores, and how

- **Reviewer:** one lawyer, ideally able to read at least one of French, German or Spanish well. Paragraphs in a language the reviewer does not read are left blank, not guessed. A second lawyer scores a random 20%, so that agreement can be reported.
- **Regulatory groups:** one per practice area, for a matter in England & Wales and for one in New York. Each group shows:
  - the practice area's terms;
  - every update of the day (titles and summaries);
  - then the updates the rule matched.
  - On the group's own row (`G01`, …), write `yes` under `missed`, with the update's title in `notes`, if an update clearly concerns that kind of matter but was not matched.
- **Translation groups:** one per document and language. Each item shows a paragraph and its translation. The original document is in the group's details.
- **Time:** about 60 minutes.

## Scales

### Regulatory match (each matched update)

| Score | Meaning |
|---|---|
| 2 — Worth a look | A lawyer on such a matter would want to know about it. |
| 1 — Marginal | Related, but rarely worth the time. |
| 0 — Noise | Unrelated; the words matched by accident. |

A match is a prompt to look, not advice. The bar is that at least half are worth a look, and that misses are few. Terms that cause noise or misses are changed in `backend/app/legal/regulatory/topics.py`; `rematch` then applies the change.

### Translated paragraph

| Score | Meaning |
|---|---|
| 2 — Faithful | Says what the original says: parties, figures, dates, obligations and conditions. Small grammar or style slips are allowed. |
| 1 — Usable with care | The meaning survives, but a term is weak or a nuance is lost (for example "shall" read as "will"). |
| 0 — Wrong | A meaning is changed, dropped or added: an obligation, a party, a figure, a date, a condition or a negation. |

Every translation in the product is labelled "not reviewed; the original prevails". This review decides whether that label is enough for reading, per language. It is never enough for filing.

## Proposed bars

| Measure | Proposed bar |
|---|---|
| Regulatory matches worth a look (2) | ≥ 0.50 |
| Regulatory groups with a missed update | reported, with the update |
| Translated paragraphs faithful (2) | ≥ 0.85 per language |
| Translated paragraphs wrong in meaning (0) | 0 |
| Agreement within one point (second reviewer) | ≥ 0.80 |

A language that misses its bars is taken off the list of languages offered (`app/legal/translation/translate.py` and `ui/translation.tsx`), and the result is recorded in DECISIONS.
