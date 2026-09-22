# Golden-evaluation reports

Each `eval_<YYYYMMDD-HHMMSS>.json` is the output of one `python3 eval/run_eval.py` run and is a
**quality record** under ISO 9001:2015 9.1.1 — see `docs/qms/procedures/SOP-12_Monitoring_Measurement_and_Analysis.md`.

Retain them. They are the only evidence of what the retrieval quality actually was at a point in
time, and a release record (`FRM-03`) cites the report that verified it.

> Reports generated before 2026-09-22 were excluded from version control by an earlier
> `eval/.gitignore` and exist only on the machine that produced them. The last recorded scores
> were 50/52 (2026-07-30, best) and 49/52 (2026-08-06, most recent).
