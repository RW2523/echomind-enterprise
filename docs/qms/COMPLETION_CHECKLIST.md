# Completion Checklist — what is needed to finish the ISO 9001 system

| Field | Value |
|---|---|
| Document ID | (working checklist — not a controlled QMS document) |
| Revision | 1.0 |
| Basis | `docs/qms/` at revision 1.0, assessed 2026-09-21; **progress updated 2026-09-22** (D1, D2, D6 closed) |
| Companion documents | `ADOPTION_GUIDE.md` (the six-step path) · `ISO9001_Gap_Analysis.md` (the findings) |

---

## How to read this

Six categories, in dependency order. **A, B and C you can finish in a few days** — they are
information and decisions. **D is engineering work.** **E cannot be shortened at all** — it is
elapsed time accumulating real records, and it is the binding constraint on certification.

There are **1,581 blank fields** in the document set. That number is misleading and worth
decomposing, because most of them are not "missing":

| Where | Count | Do they need filling now? |
|---|---|---|
| Blank forms (`FRM-01`…`FRM-12`) | ~1,275 | **No.** These are templates. A field is filled when the form is used for a real event. |
| Approval blocks (36 documents × header) | ~144 | **Yes** — category A4 |
| Registers and core documents | ~162 | **Yes** — categories A, B, C |

---

> **The questions in categories A and B are laid out as a fillable form in [`INTAKE_FORM.md`](INTAKE_FORM.md).**
> Answer that one document and everything below in A and B is covered.

## A. Information only you have

Nothing in the repository can supply these. Roughly one hour of work.

| # | What | Where it goes | Notes |
|---|---|---|---|
| A1 | Registered company name, registered address, principal place of business, company registration number | `QM-01` §1 table | Also used in the scope statement |
| A2 | The location from which the service is provided | `QM-01` §4 scope statement (line 82) | ISO requires the scope to state where |
| A3 | Who holds each role: Managing Director, Lead Engineer, Quality representative, Data custodian | `QM-01` §6.1 | One person may hold all four — record the same name four times; the roles stay distinct in records |
| A4 | Approver name + date on every document, and a signature on the policy | 36 header blocks; `QP-01` §signature | **Must be your own act.** Do not pre-fill, do not back-date. This is the single action that turns the set into a system. |
| A5 | Personnel names for the competence matrix | `REG-05` §2, §3 | Nine competence rows already listed; add the person and the evidence for each |

## B. Decisions only you can make

Nobody can make these for you; they define what you are agreeing to be held to. Half a day.

| # | Decision | Where | Guidance |
|---|---|---|---|
| B1 | Accept, change or reject each of the **7 quality objectives and their targets** | `QO-01` | Only agree to targets you will actually measure. A missed target you track is fine; a target nobody measures is a finding. |
| B2 | Confirm or change the **14 proposed risk ratings**, and assign an owner and due date to each treatment | `REG-03` | The ratings are one reading of the evidence, deliberately marked *proposed*. This is the highest-value two hours in the whole adoption. |
| B3 | **Data retention periods per data class** — uploaded documents, transcripts, boardroom audio, chat messages, `activity_log` | `SOP-11` §7 | Currently there is no policy at all. Customers in regulated sectors will ask this in procurement. Decide it per class, in days. |
| B4 | **Management review cadence** — quarterly or annual | `SOP-15` §4; `QO-01` QO-7 | Quarterly is better for a system this young; annual is the ISO minimum |
| B5 | **Internal audit approach** — (a) contract an external auditor, (b) wait for a second competent person, or (c) interim self-assessment | `SOP-13` §4 | Only (a) and (b) satisfy 9.2.2 c. Option (c) is available now but is explicitly non-conformant and must stay logged as an open nonconformity. |
| B6 | **Change significance criteria** — what requires a formal change request rather than just a commit | `SOP-06` §5 | Draft criteria are proposed; confirm or narrow them |
| B7 | Whether `AUTH_ENABLED=1` becomes the **default for customer deployments** | `SOP-11` §5; risk R-04 | A product decision with a compliance consequence |
| B8 | Audit programme dates — which clauses are audited in which period | `SOP-13` §7 schedule table | 26 blank period cells |

## C. External facts to obtain

Research and verification, not authorship. One to two days.

| # | What | Where | Why it matters |
|---|---|---|---|
| C1 | **Licence terms for all 11 models** — commercial use, attribution, redistribution | `REG-02` §2 licence column | You sell commercially into regulated sectors. This is the most likely question from a customer's legal team, and the answer is currently unknown. |
| C2 | **Nemotron gated-model terms specifically** — does acceptance permit redistribution *baked into a customer-delivered image*? | `REG-02` action A-2 | The model is compiled into images you ship. If the terms do not permit it, that is a live legal exposure, not a paperwork gap. |
| C3 | **SDXL-Turbo commercial-use terms** | `REG-02` action A-3 | Stability AI licensing has commercial conditions worth checking explicitly |
| C4 | A licensed copy of **ISO 9001:2015** itself | `REG-02` §6 | The standard is copyrighted and cannot live in the repo. You and any auditor need access to the text. |
| C5 | Rationale for the **MIT Ubuntu mirror substitution**, or remove it | `SOP-09`; `REG-02` I-06 | An undeclared third-party dependency in the build path |

## D. Engineering work

These are the gaps in `ISO9001_Gap_Analysis.md`. Ordered by consequence, not by audit severity.

### Do before the next customer deployment

| # | Work | Gap / risk | Effort |
|---|---|---|---|
| ~~D1~~ | ~~Backup and restore for `echomind_data`, with a restore actually tested~~ — **DONE 2026-09-22.** `scripts/backup_data.sh` / `restore_data.sh`; restore verified (integrity_check ok, 32 tables). **Residual: it is manual and written to the same host — see D14.** | G-03 / R-02 / QO-4 | done |
| ~~D2~~ | ~~Log rotation on all six services~~ — **DONE 2026-09-22** (50 MB × 5). **Applies on the next `docker compose up -d`; running containers keep the old config until recreated.** | G-11 / R-07 | done |
| ~~D14~~ | ~~Schedule the backup and send it off-host~~ — **DONE 2026-09-22.** `install_backup_timer.sh` (nightly systemd timer) + `BACKUP_REMOTE` off-host replication, tested. **You must choose the destination and run the installer on each deployment.** | G-22 / R-02 | done |
| D3 | **Extend the auth middleware to WebSocket endpoints**; decide the CORS default | G-06 / R-04 | 2–3 days |

### Do before a certification attempt

| # | Work | Gap / risk | Effort |
|---|---|---|---|
| D4 | **Release identification** — git tag, image tags, and a build identifier the running system can report | G-01 / R-09 / QO-3 | 1 day |
| D5 | **Re-run the golden evaluation against HEAD** and record the true current score; correct any quoted figure | G-10 / QO-2 | 1 hour |
| ~~D6~~ | ~~Retain evaluation reports as records~~ — **DONE 2026-09-22**; 24 historical reports now tracked | G-12 | done |
| D7 | **CI** running the unit suites at minimum; wire in `scripts/verify_offline_readiness.sh`, which exists and is called by nothing | G-02 / R-08 | 1 day |
| D8 | **`LICENSE`, `NOTICE`, and an SBOM** for the three images | G-05 / R-10 | 2–3 days |
| ~~D9~~ | ~~Pin `nemo_toolkit`~~ — **DONE 2026-09-22**, pinned to `60ce9407` (the build verified in production). Risk R-01 residual 20 → 4. | R-01 | done |
| D10 | **Implement the retention policy decided in B3**, including `activity_log` pruning | G-04 / R-06 | 1–2 days |

### Worth doing, lower urgency

| # | Work | Gap |
|---|---|---|
| D11 | Make namespace filtering structural so a new retrieval path cannot bypass it | R-05 — this class of defect has already occurred twice |
| D12 | Pre-download the cross-encoder reranker in the Dockerfile; log a WARNING when the fallback engages | R-11 — a silent quality degradation today |
| D13 | Pin `ollama/ollama` and `cloudflare/cloudflared` to digests | R-01 family |

## E. Records that must accumulate — the real constraint

**This is what cannot be compressed, and it is why certification takes months rather than weeks.**
A certification body needs to see the system *operating*, and operating evidence is created by time
passing, not by writing documents. Creating these retrospectively would be falsification.

| # | Record | Minimum for certification | Where it goes |
|---|---|---|---|
| E1 | **Management review minutes** | At least one; two or three is materially stronger | `records/management-review/`, using `FRM-07` |
| E2 | **Internal audit report** covering every clause | One full cycle, by an impartial auditor | `records/audits/`, using `FRM-06` |
| E3 | **Nonconformities raised and closed with effectiveness verified** | Several, showing the full cycle — this is where most young systems fail | `REG-04` §B + `records/capa/`, using `FRM-05` |
| E4 | **Objectives measured at two or more points** so a trend exists | At least two measurement rounds | `QO-01` + `SOP-12` |
| E5 | **Release records** | One per release made after adoption | `REG-08` + `records/releases/`, using `FRM-03` |
| E6 | **Supplier evaluations** | The critical providers, using `FRM-04` | `REG-02` |
| E7 | **Requirements review + satisfaction review** for any engagement | Per customer, using `FRM-11` and `FRM-12` | `REG-06` |
| E8 | **Corrective action on the audit findings from E2** | Closed, with effectiveness verified | `REG-04` |

**Realistic elapsed time: three to six months** from adoption to a credible certification audit,
assuming A–D are done in the first few weeks.

## F. If you pursue certification

| # | Step | Note |
|---|---|---|
| F1 | Choose an **accredited** certification body | Accreditation matters — UKAS, ANAB or your national equivalent. An unaccredited certificate is worth little to a regulated buyer. |
| F2 | Stage 1 readiness review | A documentation review. Book it once A–D are done and E is underway. |
| F3 | Close stage 1 findings | Expect some |
| F4 | Stage 2 certification audit | The operating-evidence audit — this is where E is tested |
| F5 | Surveillance audits | Typically annual, over a three-year cycle |

Budget and timescales vary by body and scope; obtain quotes at F1 rather than assuming.

---

## Summary — the shortest honest answer

| Category | Work | Elapsed |
|---|---|---|
| A — your information | ~1 hour | Day 1 |
| B — your decisions | ~half a day | Day 1–2 |
| C — licence verification | 1–2 days | Week 1 |
| D1–D3 — operational gaps | 3–5 days | Week 1–2 |
| D4–D10 — certification gaps | ~1 week | Week 2–4 |
| E — accumulating records | **cannot be shortened** | **3–6 months** |
| F — certification | depends on the body | +1–2 months |

**Nothing in A–D is blocked on anything external.** The only genuine external dependency is B5/E2 —
an impartial auditor — and the only genuine time constraint is E.
