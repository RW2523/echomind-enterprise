# Provenance — eval/fixtures/law

| File | Origin | Licence / use | Notes |
|---|---|---|---|
| `Master Services Agreement (DEMO).md` | Written for this project on 2026-10-08 | Internal test fixture; fictional | Recreated from the facts asserted by `eval/golden/law.jsonl`. The original DEMO documents existed only in the production data volume and were not copied from it. Also carries the clauses used by the design prototype and demo journey (§8.1, §14.2, §22.1). |
| `Contract Review Playbook (DEMO).md` | Written for this project on 2026-10-08 | Internal test fixture; fictional | Same as above. |
| `Advice memo on the MSA (DEMO).md` | Written for this project on 2026-10-10 | Internal test fixture; fictional | S3.4 citation checker (journey 25): cites the MSA (DEMO) with one altered quotation (clause 4.2), one accurate quotation (6.1), one clause that does not exist (18.3), and an invented case that cannot be real (F.5th, 14th Cir., 2031). The other cases named are invented too. |

All parties, amounts and terms are invented. No real client data. Do not add real documents to this folder.
Load into a **development** backend only: `python3 scripts/law/load_law_fixtures.py --base http://127.0.0.1:8110/api`.
