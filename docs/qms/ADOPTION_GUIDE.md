# Adoption Guide — turning this document set into a management system

| Field | Value |
|---|---|
| Document ID | (not a controlled QMS document — this is guidance for adopting the set) |
| Revision | 1.0 |
| Applies to | `docs/qms/` at revision 1.0, authored 2026-09-21 |

---

## 1. What you have, and what you do not

**You have** a complete ISO 9001:2015 document set — a quality manual, a policy, objectives, fifteen
procedures, eleven forms and six registers — written specifically against what this repository
actually does, with the evidence cited by file and line.

**You do not have a quality management system.** Not yet. A QMS is a set of processes that are
*operating*: approved, followed, measured, audited and reviewed. What is in `docs/qms/` is the
description of one. The difference matters more than it sounds, and it is the difference an auditor
will test in the first ten minutes.

Three things are therefore true at once, and all three should be said plainly to anyone who asks:

1. The documentation is real and specific, not boilerplate.
2. Nothing in it has been approved, and no audit or management review has taken place.
3. The underlying engineering evidence — tests, a golden evaluation suite, recorded defect
   analysis — is genuinely strong, and is what the system is built on.

## 2. Do not do these things

| Do not | Why |
|---|---|
| Present this set as an *implemented* QMS | Every document says `DRAFT — not yet approved`. Saying otherwise is a misrepresentation an auditor will detect immediately, and it would undermine the honesty that the quality policy itself commits to. |
| Claim or imply ISO 9001 certification | Certification requires an accredited certification body to audit you. Nothing here confers it. |
| Fill in approval names, signatures or dates on someone's behalf | An approval is a person's act. A pre-filled signature block is a fabricated record. |
| Back-date anything | Same reason, worse consequence. |
| Write audit findings or management review minutes for meetings that did not happen | This is the single most common way a young QMS becomes fraudulent rather than immature. Immature is fine and fixable. |
| Quote the evaluation score from `README.md` | 48/52 matches no recorded run. The last recorded run is 49/52 on 2026-08-06 and predates HEAD by three feature commits. Re-measure before quoting anything (QO-2). |

## 3. Minimum viable adoption — the shortest honest path

Six steps. Until step 6, the honest description is "QMS documented, adoption in progress".

### Step 1 — Read and correct (½ day)

Read QM-01, QP-01 and QO-01. These were written from repository evidence, not from an interview with
you. Some of it will be wrong about intent even where it is right about fact. Correct it. In
particular, fill in the organisational details in QM-01 §1 and decide whether the QO-01 targets are
ones you will actually be held to.

### Step 2 — Approve (1 hour)

For each document: satisfy yourself it describes what you will actually do, then complete
*Approved by* and *Approval date* and change Status to `APPROVED`. Update REG-01 as you go. Commit
the approvals yourself — under SOP-01 §6 the commit is the approval event.

Approve in this order, because later documents depend on earlier ones: QP-01 → QM-01 → QO-01 →
SOP-01 → the remaining SOPs.

### Step 3 — Confirm the risk ratings (2 hours)

REG-03 contains 14 risks derived from real evidence, with **proposed** likelihood and impact scores.
They are one reading of the evidence. Work through them, change what you disagree with, assign
owners and due dates to the treatments, and set REG-03 to revision 1.1. This is the highest-value
two hours in the whole adoption.

### Step 4 — Close the two gaps that are not really about paperwork (1–2 days)

Two gaps are operational risks that happen to also be ISO findings. Do these before the first audit,
not after:

- **G-03 / QO-4 / risk R-02 — back up `echomind_data`.** It is the only volume holding customer
  data and there is no backup procedure. Write it, run it, and restore from it once to prove it
  works.
- **G-11 / risk R-07 — bound the container logs.** `docker-compose.yml` sets no logging options, so
  logs grow until the disk fills and takes the database with it. It is a few lines of YAML.

### Step 5 — Hold the first management review (2 hours)

Follow SOP-15's standing agenda. You will not have data for every input; record "no data — first
review" against those, which is an honest and acceptable entry. Record real decisions and real
actions with real owners and dates in FRM-07. This is the first genuine QMS record you will create.

### Step 6 — Arrange the first internal audit

This is the one step you cannot do alone. ISO 9001:2015 9.2.2 c requires that auditors do not audit
their own work, and there is currently one person in the organisation. SOP-13 §4 sets out the
options; the realistic one for a small organisation is a contract auditor for one or two days a
year. Until it happens, the gap stands as an open nonconformity of the QMS itself — which is exactly
how SOP-13 records it.

## 4. If you are heading for certification

Add, in roughly this order:

1. **Three to six months of operating records.** A certification body wants to see the system
   running, not just written: management review minutes, a completed internal audit cycle,
   nonconformities raised and closed with effectiveness verified, objectives measured over time. The
   records are the evidence; you cannot create them retrospectively.
2. **A full internal audit covering every clause**, by someone independent.
3. **Closure of the structural gaps** in `ISO9001_Gap_Analysis.md` — particularly release
   identification (G-01), automated verification (G-02) and the licence inventory (G-05).
4. **A stage 1 readiness review** with your chosen certification body before booking stage 2.

Budget the elapsed time honestly: the constraint is not the paperwork, it is accumulating genuine
operating evidence, and that takes months by definition.

## 5. Keeping it alive

A QMS decays quietly. Three habits prevent it:

- **Raise nonconformities in REG-04 as they happen**, not at audit time. You already write excellent
  root-cause analysis in commit messages — REG-04 §A proves it. The change required is small: a row
  in the register with an owner, and an effectiveness check before closing.
- **Run the management review on the calendar**, even when there is little to discuss. A short
  review that happened beats a thorough one that did not.
- **When a procedure stops matching reality, change the procedure.** A document that describes a
  process nobody follows is worse than no document: it fails the audit *and* it misleads whoever
  joins next. Changing it is a two-minute commit.

## 6. A note on why the gaps are written down

The gap analysis and this guide name every weakness found, including some an auditor might not have
reached on their own — no versioning, no CI, no backup of the customer-data volume, authentication
off by default, licences unverified.

That is deliberate, and it is the same principle the product itself is built on. EchoMind's central
quality claim is that it says "I could not find that" rather than inventing an answer. A quality
management system for that product cannot credibly assert conformity it does not have. An
organisation that has already found its own gaps, owns them and has dated actions against them is in
a materially stronger position — with an auditor and with a customer — than one presenting an
unblemished document set that does not survive contact with the evidence.
