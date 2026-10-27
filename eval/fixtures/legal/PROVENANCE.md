# Provenance — eval/fixtures/legal

| Files | Origin | Licence / use | Notes |
|---|---|---|---|
| `MAT-1042/*.txt` (eight files) | Written for this project on 2026-10-26 from `docs/echomind-legal/13_DEMO_SEED_DATA.md` | Internal demo and test fixtures; fictional | *Morgan v. Apex Industries*: a contract (delivery by May 15), a signed amendment moving it to May 20, two witnesses who give May 12 and May 15, a shipment log showing dispatch on May 12 and arrival on May 15, an unclear email thread, draft notes with a prompt-injection canary (`INJECTION-CANARY`), and a memo citing an authority that does not exist. |
| `MAT-2033/Rivera_Confidential_Note.txt` | Written for this project on 2026-10-26 | Internal test fixture; fictional | A highly restricted matter alice cannot see; `RIVERA-CANARY-2033` must never appear in her answers. |

Every party, person, amount, date and authority is invented; every file says `DEMO / SYNTHETIC`. No real client data.
Load into a **development or demo** backend only: `python3 scripts/law/demo_load.py` (after `seed-demo`).
The blueprint's second firm ("White Oak Legal") is not seeded: the demo world has one firm; cross-firm isolation is
enforced by `org_id` in every policy query (`app/legal/policy.py`).
