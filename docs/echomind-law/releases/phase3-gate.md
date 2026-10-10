# EchoMind Law — Phase 3 gate report (S3.9)

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Branch | `law/s3.9-gate` (from `feature/echomind-law` after S3.8, `41eafcf`) |
| Code under test | `b134208`. The final gate run (§4) ran on it, and later commits are documentation only. Two earlier full gate runs (on `9a3894f` and `df3b8a5`) are also recorded; the second is the one whose drafting results led to `s3.1e` and `s3.1f` |
| Answer model | `nvidia/Qwen3-30B-A3B-FP4`, TensorRT-LLM on the host (`localhost:8355`; shared with production, used read-only as approved) |
| Embeddings | `nomic-embed-text` (Ollama on the host, `localhost:11435`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, in-process from the local cache (offline) |
| Data | Fictional only |
| Production impact | None: nothing ran against production containers, volumes or ports |

**Exit criteria (IMPLEMENTATION_PLAN §7, S3.9):**

| Criterion | State |
|---|---|
| Planted-error suite | **Done.** `eval/law/run_gate.py --phase 3` runs every Phase 3 golden and held-out set, T10, the security suite and the browser journeys, and records the runs (§4) |
| Fresh blind held-out sets for every checker changed since its last held-out run | **Done.** Each was committed before it was run; every run is reported (§2), including the ones that led to fixes |
| Format round trips | **Done** for what can be automated (§5). The manual Microsoft Word check is **🔒** |
| Lawyer sign-off | **Ready, not held (🔒).** Rubric `releases/phase3-rubric.md`, a 66-item packet from the held-out outputs, and the scorer |
| Owner sign-off | **Required (🔒)**, at the end of this report |

## 1. What Phase 3 delivered

| Slice | Backlog | What a lawyer gets | Decision |
|---|---|---|---|
| S3.1 | DOC-01 | Drafts from approved firm templates: fields from the matter record, checked quotes or a lawyer; written passages citing their sources; any unsupported sentence or figure a gap; filed as Word with the approval | D-045 |
| S3.2 | DOC-02a | A clause library and firm playbooks; each clause classed standard, fallback, outside, not found or unclear, with figures and laws decided by rule | D-046 |
| S3.3 | DOC-02b | Redlines to the firm's positions as real Word tracked changes, verified before release; a second lawyer approves | D-047 |
| S3.4 | DOC-03 | References resolved to exact clauses or flagged missing; quotations compared word for word; every authority listed as UNVERIFIED, impossible ones flagged; nothing looked up | D-048 |
| S3.5 | DOC-04 | Definitions, numbering and cross-references checked; corrections only through a redline a second lawyer approves | D-049 |
| S3.6 | DOC-05 | Obligations and deadlines in the document's words; dates only where the document fixes them; proposed until a lawyer confirms | D-050 |
| S3.7 | DOC-06 | Approved firm checklists run over a data room: the missing-document list and a preliminary issue register | D-051 |
| S3.8 | DOC-07 | A company's documents read against each other (share capital, signing authority, constitution, IP chain), each finding quoting both | D-052 |
| S3.9 | OPS-02 | The Phase 3 gate: fresh blind sets, the gate runner, round trips, the review packet | D-053 |

## 2. Held-out results, every run as it went

Held-out sets are written and committed before they are run, and each is run once per version. When a run leads to a fix, the version changes, the set becomes tuning data, and a new blind set is written. The runs below are all of the gate's held-out runs, in order.

| Tool | Held-out run | Result | What followed |
|---|---|---|---|
| Citations (T07) | Set 3 on `citations-s3.4e` | 19/20, no false alarms. "CPR r. 3.4(2)(a)" was read without its "(a)" | `s3.4f` keeps lettered sub-paragraphs; set 3 became golden |
| | Set 4 on `s3.4f` | 20/21, no false alarms, nothing wrongly flagged. "Late Payment of Commercial Debts (Interest) Act 1998" was not read | `s3.4g` reads Act names with joining words and a bracketed part; set 4 became golden |
| | **Set 5 on `s3.4g`** | **24/24**: references 4/4, missing 2/2, quotations 4/4 (altered 1/1), authorities 16/16, malformed 4/4, no false alarms, no lookups | — |
| Definitions (DOC-04) | **Set 4 on `definitions-s3.5c`** | **9/11** caught, fixes 2/2, control clean. Missed: the undefined terms "Sub-processor" and "Investor Consent" (the known recall limit, D-049). Two further findings ("personal data", "shareholders" used where a term is defined) are true; their planted twins were counted, so they read as repeats | No change (a limit, §7) |
| Obligations (DOC-05) | **Set 6 on `obligations-s3.6g`, with the checked model pass** | **7/7**, timing in the document's words 4/4, dates 2/2, **none guessed**, no extras, control clean | — |
| Due diligence (DOC-06) | **Set 3 on `diligence-s3.8b`** | Missing lists 2/2, **issues 5/5**, no extras, control clean, every quote at its offsets | — |
| Governance (DOC-07) | **Set 2 on `s3.8b`** | **4/4**, no extras, control clean | — |
| Drafting (DOC-01) | Set 2 on `drafting-s3.1` | Fields 8/11, absent left open 1/1, **wrong 0, invented 0** | — |
| | Golden on `s3.1` (gate run) | 19/22 with **one wrong value**: the first party named again as the second | `s3.1b`: a party already named is left for a person, with the reason |
| | Set 2 on `s3.1b` | 7/11 with **one wrong value**: the agreement given as a party's name (right on `s3.1`; model variation on a field `s3.1b` does not touch) | `s3.1c`: an "agreement" field kind that must name a document; set 2 became golden |
| | Set 3 on `s3.1c` | Fields 5/9, wrong 0, invented 0. The research answerer's abstention floor emptied drafting's search on a one-passage email, and the writer wrote from the title | `s3.1d`: drafting's search keeps such passages (values are checked against quotes); with no source, a written passage is a gap; set 3 became golden |
| | Set 4 on `s3.1d` | Fields 9/10, absent left open 1/1, **wrong 0, invented 0**; 5 sentences marked for a lawyer. The miss is the model naming the first party again, left for a person by the `s3.1b` rule | — |
| | Golden on `s3.1d` (first full gate run) | 38/42 with **two wrong values** on the code that had given 41/42 and none minutes before: "the 2026 annual licence fee" accepted as an agreement, and "to be agreed" as the governing law | `s3.1e`: the document word must head an agreement's name; a value saying the point is open is left for a person |
| | Golden on `s3.1e`, two runs | 39/42 wrong 0; then 38/42 with **one wrong value**: a demand letter's recipient read as the firm's own client | `s3.1f`: the matter's client is compared first, whatever the field order, and legal forms (Ltd, AB) do not make another party |
| | Golden on `s3.1f`, two runs | 41/42 and 39/42, **wrong 0, invented 0**, absent 6/6 | — |
| | **Set 4 on `s3.1f`** (it informed neither `s3.1e` nor `s3.1f`) | Fields **9/10**, absent left open 1/1, **wrong 0, invented 0** | — |
| Playbooks (DOC-02a) | Held-out on `playbook-s3.2c` (S3.7, cited) | 9/10, deviations 4/4, **passed in error 0** | — |

**Reading these honestly:**
- **Every fix was made on what a held-out run showed, then measured on a new blind set.** None of the final results come from a set that informed the code it measured.
- **Drafting needed five fixes at the gate.** Run-to-run variation in the model is real: the same code gave 0 and then 2 wrong values on the tuning set. Each wrong value had a rule-shaped cause, and each now leaves the field for a person, with the reason:
  - the same party named twice;
  - a non-document as the agreement;
  - "to be agreed" taken as a value;
  - the firm's own client given as the other side.

  On the final code, wrong values were 0 in two golden runs and on the blind set. The sets are small, and the lawyer review measures quality.
- **Written passages:** a cited sentence can still narrow its quote. For example, "each party's total liability" was written as "the Provider's". Quote checks prove the words exist, not that the sentence follows from them. Uncited sentences are marked; the lawyer review scores the rest.
- **These are small, fictional sets.** They show the mechanisms work; the lawyer review measures quality.

## 3. Spec tests

| Spec test | What was run | Result |
|---|---|---|
| **T07** citations | Security-suite group, the golden set (124/124) and held-out set 5 (24/24); every authority unverified, none looked up | **PASS** |
| **T10** Word tracked changes | 487/487 revisions verified, LibreOffice round trips 12/12; the deletion-only correction round trip (§5) | **PASS (automated)**; the manual Word check is 🔒 |
| **T01, T02, T08, T09, T13, T15** | Security suite (§4): 24 groups, 901 tests. Phase 3 adds a group per slice. Each tool reads only documents the person can see, and every finding is hidden with a document the person loses | **PASS** |
| **T11** offline egress | Capture mode, as at Phase 2; the enforced run needs root (🔒) | **PARTIAL** |
| **T12** load | Unchanged from Phase 2: the measured run needs the window (🔒) | **PARTIAL** |

## 4. The gate run (`df3b8a5`)

`cd backend && python3 ../eval/law/run_gate.py --phase 3 --import-into ../.law-data`: **17 steps, 0 failed**, 25 minutes (`eval/law/reports/20261010-145253-gate-phase3.*`). The evaluation runs are recorded on the Evaluations page. A held-out set already run on the current version is cited, not run again.

| Suite | Set | Result |
|---|---|---|
| Drafting (DOC-01) | Golden (13) | Fields 41/42, absent left open 6/6, **wrong 0, invented 0**, 4 sentences marked |
| | Held-out 4 (cited) | 9/10, absent 1/1, wrong 0, invented 0 |
| Playbooks (DOC-02a) | Golden | 15/15, deviations 5/5, **passed in error 0** |
| | Held-out (cited, S3.7) | 9/10, deviations 4/4, passed in error 0 |
| Redlines (T10) | 14 documents | **487/487** revisions placed and verified; LibreOffice round trips **12/12** |
| Citations (T07) | Golden (16) | **124/124**: references 42/42, missing 17/17, quotations 19/19, altered 6/6, authorities 63/63, malformed 15/15, false alarms 0, lookups 0 |
| | Held-out 5 (cited) | 24/24 |
| Definitions (DOC-04) | Golden (10) | 52/55, fixes 8/8, false findings 0, controls clean (the misses are three undefined terms) |
| | Held-out 4 (cited) | 9/11, fixes 2/2 |
| Obligations (DOC-05) | Golden (16) | 67/68, timing 50/51, dates **18/18, none guessed**, controls 5/5 clean; 4 extra proposals (a notice clause's address line, two payment lines, a repair notice) for a lawyer to dismiss |
| | Held-out 6 (cited) | 7/7, dates 2/2, none guessed |
| Due diligence (DOC-06) | Golden (6 rooms) | Missing lists 6/6, issues 29/29, no extras |
| | Held-out 3 (cited) | 5/5 |
| Governance (DOC-07) | Golden (6 companies) | 15/15, no extras |
| | Held-out 2 (cited) | 4/4 |
| Security suite | 24 groups | **All pass, 901 tests** |
| Browser journeys | Law suite | **29 passed**, 0 failed, 0 flaky (12.6 min) |

**Other suites at this commit:**
- Backend (all): **1115 passed**;
- Vitest: **132 passed**;
- `tsc`: clean;
- Voice: not rerun (unchanged since run 24).

## 5. Format round trips

| Format | How it is checked | Result |
|---|---|---|
| Drafts: Word and the PDF preview | `test_the_filed_word_document_and_the_pdf_preview_round_trip_through_libreoffice` | Pass |
| Redlines: Word tracked changes | T10 evaluation (487/487 revisions verified, LibreOffice 12/12) and `test_t10_libreoffice_opens_the_redline_and_keeps_its_tracked_changes` | Pass |
| Corrections (S3.5) deleting a clause whole | `test_a_correction_deleting_a_clause_whole_survives_libreoffice` (new): LibreOffice keeps the tracked deletion and its author; accepted, the clause is gone; rejected, it is back once | Pass |
| Exports (T09): released Word file | `test_t09_the_released_word_file_reads_back_in_libreoffice_without_what_was_redacted` (new), with the existing T09 checks of the file itself | Pass |
| Exports (T09): PDF | The T09 tests read the PDF's text, metadata and objects | Pass |
| **Microsoft Word** opening a redline | Manual (🔒) | Waiting |

## 6. Found and fixed at the gate

- **Drafting gave the same company for both parties** (golden, `s3.1`). A party already named is now left for a person (`s3.1b`). The matter's own client is now compared first, whatever the field order, and legal forms are ignored (`s3.1f`).
- **Drafting accepted "the 2026 annual licence fee" as an agreement, and "to be agreed" as a governing law** (golden, `s3.1d`). The document word must now head the name, and a value saying the point is open is left for a person (`s3.1e`).
- **Drafting filled "which agreement" with a party's name** (held-out, `s3.1b`). The agreement field must now name a document.
- **Drafting's search returned nothing on a one-passage email** (held-out, `s3.1c`). The research answerer's abstention floor applied, so every field was left open and the writer wrote from the title. The floor is now off for drafting, whose values are checked against their quotes; with no source, a written passage is a gap.
- **Court rules lost lettered sub-paragraphs**, and **UK Act names with joining words were not read** (citations, held-out).
- **The obligations evaluation recorded "rules only"** although the checked model pass ran by default. It now records the model and its version. The earlier reports keep their wording, and this report says which runs used the model.

## 7. Known limits

- **Undefined terms** are found only by the patterns in D-049 (a defined first word, repeated use, a determiner). "Sub-processor" and "Investor Consent" were missed.
- **Drafting written passages** may narrow or widen a cited quote (§2).
- **Citations:** only the authority forms in D-048 are read. Treaty articles ("Article 101 TFEU") and state codes are not.
- **The checkers are advisory.** Nothing changes a document without a lawyer and a second approver.
- **Small evaluation sets.** The lawyer review is the quality measure.

## 8. Waiting for approval (🔒)

- The **lawyer review session** (rubric `releases/phase3-rubric.md`; packet `eval/law/lawyer_review/phase3/packet.html` and `sheet.csv`; then `score.py --phase 3`), and the owner's confirmation of the proposed bars.
- The **manual Microsoft Word check** of a redline and a corrections redline (S3.3, S3.5).
- From Phase 2, still waiting:
  - the Phase 2 lawyer review;
  - the load-test window;
  - the first `compose.sh up` on the shared host;
  - the root-run T11;
  - counsel's P-decisions (P-008 signs off templates and thresholds; P-009 licensed sources);
  - any push or PR.

## 9. Reproduce

```
cd backend && python3 ../eval/law/run_gate.py --phase 3 --import-into ../.law-data   # every Phase 3 suite, security and browser
python3 eval/law/lawyer_review/make_packet.py --phase 3
python3 eval/law/lawyer_review/score.py --phase 3 eval/law/lawyer_review/phase3/sheet.csv
```

## 10. Sign-off

Phase 3 is complete on `law/s3.9-gate` except for the items in §8.
- **Merged** into `feature/echomind-law` under the owner's standing approval (2026-10-10: "approved, merge S2.8 and start S3.1 / approve everything and finish everything").
- **Still the owner's to decide:** the items in §8.
- **Before the pilot:** the owner should schedule the lawyer review and the manual Word check.
