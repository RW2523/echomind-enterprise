# EchoMind Law — Phase 4 gate report

| Field | Value |
|---|---|
| Date | 2026-10-10 |
| Branch | `law/s4.6-gate` (from `feature/echomind-law` after S4.5, `14a0d9d`) |
| Code under test | `2b2fcd7` (two panel fixes made at the gate, §5); later commits are documentation, reports, the review packet and the gate runner's list of latencies only |
| Answer model | `nvidia/Qwen3-30B-A3B-FP4`, TensorRT-LLM on the host (`localhost:8355`; shared with production, used read-only as approved) |
| Embeddings and reranker | `nomic-embed-text` (Ollama, `localhost:11435`); `cross-encoder/ms-marco-MiniLM-L-6-v2` in-process |
| Data | Fictional only |
| Production impact | None |
| Business approval | Given by the owner on 2026-10-10 ("approve everything and finish everything"), which covers building and merging every Phase 4 slice |

**Phase 4 is optional by design:** every slice sits behind its own flag, off by default (IMPLEMENTATION_PLAN §7).

| Criterion | State |
|---|---|
| S4.1 Intake assistant (OPT-01) | **Done** (D-054). `LEGAL_INTAKE`, `LEGAL_INTAKE_LINKS` |
| S4.2 Summaries (OPT-02) | **Summaries done** (D-055). `LEGAL_SUMMARIES`. **Translation BLOCKED:** no approved local translation model and licence |
| S4.3 Time-entry drafts (OPT-03) | **Done** (D-056). `LEGAL_TIME_ENTRIES` |
| S4.4 Regulatory monitoring (OPT-04) | **BLOCKED:** needs a licensed source and network approval; off in air-gap mode. Not built |
| S4.5 Four-perspective panel (OPT-05) | **Done** (D-057). `LEGAL_PANEL` |
| Evaluation and benchmark | **Done.** The final gate run passed every step; two earlier runs and what they found are kept (§3) |
| Lawyer sign-off | **Ready, not held (🔒).** Rubric `releases/phase4-rubric.md`, a 53-item packet in 6 groups, and the scorer |
| Owner sign-off | The standing approval covers merging. The 🔒 items in §7 stay with the owner |

## 1. What Phase 4 delivered

| Slice | What a firm gets | Without a model? |
|---|---|---|
| S4.1 | Enquiries taken from a fixed script after a disclaimer, by the front desk or through a single-use link. Every name is screened against clients and a new matter party register. Staff and the enquirer see only "refer to the conflicts team"; the conflicts team sees the matches and decides. Calls are recorded only under consent. Template replies are approved by a second lawyer and never name another client. A cleared enquiry becomes a matter. Personal data can be purged | Yes |
| S4.2 | Key points of a document or transcript, each quoting it, cited by clause and line, page and line, or the time in a transcript, computed from the quote's position. Reviewers edit; a second person approves; filed; the original never changed | No (checked model) |
| S4.3 | Draft time entries from the person's own audited work, per matter and day, in 0.1-hour units, with a narrative naming kinds of work (never content). The person edits and approves; approved entries download as CSV. No billing integration | Yes |
| S4.5 | One question answered by four perspectives (deal, risk and compliance, litigation, a specialist), each citing the documents, then a moderator's disagreement matrix. It stops for missing facts. The lawyer gives them and it runs again, or continues without them, with a reason recorded for the partner. A partner approves. It is shown as one model's perspectives, not independent experts | No (checked model) |

## 2. Held-out results, every run as it went

| Feature | Run | Result | What followed |
|---|---|---|---|
| Summaries | Golden on `summary-s4.2` | Facts 17/19 (a dense contract lost two clauses to a six-point cap), citations 17/17, quotes 21/21, figures 0 | `s4.2b`: smaller sections, ten points |
| | Golden on `s4.2b`, two runs | 19/19 and 18/19; citations all right; figures 0 | — |
| | **Held-out 1 (blind, `s4.2b`)** | **Facts 15/15, citations 15/15** (clauses, pages and lines, timecodes), quotes 16/16, figures 0 | — |
| Panel | Golden on `panel-s4.5` | Stop 3/3, quotes 20/20; one matter cited by clause number, 17 uncited sentences | `s4.5b`: such citations matched to their source; echoed or repeated missing facts dropped |
| | Golden on `s4.5b` | Stop 3/3, quotes 32/32, uncited 7, figures 0, 37.8 s | — |
| | Held-out 1 (blind, `s4.5b`) | **Stop 2/3**, quotes 29/29, figures 0, 30.4 s. All four perspectives lacked facts, but the moderator did not stop the panel | `s4.5c`: three perspectives lacking facts also stops it; set moved to golden |
| | Golden on `s4.5c` (6 matters) | Stop 6/6, quotes 61/61, all perspectives answered, figures 0, 37.7 s | — |
| | **Held-out 2 (blind, `s4.5c`)** | **Stop 3/3, quotes 32/32**, all 12 perspectives answered, figures 0, **35.5 s per panel** | — |
| | Golden on `s4.5c` again (gate run 1) | Stop 6/6, quotes 55/55, figures 0, 36.1 s | Journey 33 then found citations written as `"quote" [12.2]` going unchecked: `s4.5d` |
| | Golden on `s4.5d`, two runs | Stop **5/6** and 6/6 (on one matter the panel listed the missing facts but did not stop: model variation on the same code), quotes 63/63 and 60/60, figures 0 | — |
| | **Held-out 2 (blind, `s4.5d`)** | **Stop 3/3, quotes 30/30**, all 12 perspectives answered, 36.5 s per panel. One figure flagged that was a clause number (the flag is advisory) | — |

Intake and time entries follow rules without a model. Their measures are tests, recorded in runs 31 and 33.

## 3. The gate run

**Final run `20261010-215020` on `2b2fcd7`: 6 steps, 0 failed.** The five evaluation runs and the gate were recorded for the Evaluations page.

| Step | Result | Time |
|---|---|---|
| Summaries, golden | Facts 19/19, citations 19/19, quotes 23/23, figures 0; 8.3 s a document | 36.7 s |
| Summaries, held-out | Cited: held-out 1 on `s4.2b` (facts 15/15, citations 15/15) | — |
| Panel, golden | Stop 6/6, quotes 63/63, all 24 perspectives answered, figures 0; 36.2 s a panel | 221.3 s |
| Panel, held-out | Cited: held-out 2 on `s4.5d` (stop 3/3, quotes 30/30) | — |
| Security suite | 28 groups, 1063 tests, all pass | 592.4 s |
| Browser journeys | **33/33** | 893.5 s |

**Two earlier gate runs, both kept:**
- **`20261010-203349` on `14a0d9d`:** failed only on journey 33. Two perspectives cited as `"quote" [12.2]`, which was not checked, so they showed as citing nothing. This led to `s4.5d`.
- **`20261010-211716` on `1723b12`:** failed only on journey 33. The panel stopped, and each re-run with the lawyer's facts found new missing facts, so nothing reached the partner. This led to continuing without the facts.

**What the final journey did not cover:** in the final run the panel did not stop, so the browser did not exercise the facts form or continuing without facts (the real model decides). Both are covered by backend tests and Vitest, and the journey takes either path when the panel stops.

The gate runner listed no latencies for these two features. That is fixed in `run_gate.py` for later runs; the times above come from the suite reports.

## 4. Spec tests and security

| Area | Result |
|---|---|
| **Deny by default (SEC-01)** | The only Law routes without a session are enquirer links. They are on only with both intake flags, otherwise 404, and a made-up token gets 404. A route-table test covers both flag states |
| **T08 prompt injection** | Summaries and panel prompts withhold AI-addressed text; passages are fenced. The shared detector gained "AI assistant:" labels and "ignore your instructions" (S4.2), for every feature |
| **T01 isolation** | Every Phase 4 view shows only matters and documents the person can see. Panels and summaries are hidden with a document the person loses; time entries are their owner's only; intake matches are for the conflicts team and managing partners |
| **Logs (T-11) and THREAT_MODEL T-24** | Link and enrolment tokens are masked in the backend's and nginx's access logs. Audit details hold counts and flags only (tested for intake, summaries, time and panel) |
| **T13 rollback** | Migrations 0031–0033 refuse a rollback that would delete intakes, party records, role grants, edited or approved time entries, or panel reviews |
| **Security suite** | **28 groups, 1063 tests, all pass**, including intake (30), summaries (10), time entries (9) and the panel (19) |

## 5. Found and fixed in Phase 4

- **Intake (journey 30):** a converted enquiry lost its matter link, and the new matter showed "Not available" because the matters list was stale.
- **The T08 detector (S4.2):** missed "AI assistant:" labels and "ignore your instructions".
- **Time entries (journey 32):** a 45-second sitting was suggested as 0.2 hour, and the page read "1 entries".
- **Panel:**
  - citations labelled by clause number went unchecked;
  - the moderator could leave a panel running when all four perspectives lacked facts;
  - a panel on a matter with several agreements stopped legitimately and could not be narrowed, so questions can now be limited to chosen documents;
  - nine missing facts exceeded the facts form's eight;
  - citations written as a quote and then its clause, `"quote" [12.2]`, or a bare `[6.1]`, went unchecked; they are now matched to their source (`s4.5d`);
  - a panel still lacking facts after the lawyer's answers could not reach approval, because each run found new missing facts. A lawyer can now continue without them, giving a reason that is recorded with their name and shown to the partner (gate runs 1 and 2 failed on journey 33).
- **Tests:** a flaky time-entry test (a planted id that random ids can contain) was fixed.

## 6. Known limits

- **Summaries:** points often restate their quote closely; facts spanning sections are summarised separately; Word documents are cited by clause and line (no pages).
- **Panel:**
  - the four perspectives come from one model and are often alike;
  - missing facts may repeat in different words (at most eight are asked);
  - whether a panel stops varies between runs on the same code (golden 5/6 once, 6/6 otherwise); the lawyer sees the listed facts either way;
  - the figure check can flag a clause number;
  - a short question that misses the documents' words can find no sources;
  - about 30 to 40 seconds a panel on this host.
- **Intake:** screening is by name only (no addresses, company numbers or corporate families beyond those recorded). nginx's error log can show a link path when the backend is unreachable.
- **Time entries:** only work done in EchoMind is seen. The gap and allowance are defaults for the firm to confirm.

## 7. Waiting for approval (🔒)

- The **lawyer review sessions**:
  - Phase 4 (`releases/phase4-rubric.md`; packet `eval/law/lawyer_review/phase4/`);
  - Phases 2 and 3, still unscored.
- **S4.4 regulatory monitoring:** a licensed source and an approved outbound channel.
- **Translation (OPT-02):** an approved local translation model and its licence.
- The **firm's policies:** consent wording (P-001), retention of enquiries (P-002), who signs off (P-008), and billing rules for time entries.
- From earlier phases:
  - the manual Word check;
  - the first `compose.sh up` (speech stack; transcribing intake recordings);
  - the root-run T11;
  - the load-test window;
  - any push or PR.

## 8. Reproduce

```
cd backend && python3 ../eval/law/run_gate.py --phase 4 --import-into ../.law-data
python3 eval/law/lawyer_review/make_packet.py --phase 4
python3 eval/law/lawyer_review/score.py --phase 4 eval/law/lawyer_review/phase4/sheet.csv
```

## 9. Sign-off

Every Phase 4 slice that can be built without further approval is built, tested and merged.
- **S4.4 and machine translation:** blocked by design.
- **Before the pilot:** the owner should decide which Phase 4 flags the pilot firm turns on, and schedule the lawyer reviews.
