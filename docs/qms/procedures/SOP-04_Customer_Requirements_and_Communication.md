# SOP-04 — Customer Requirements and Communication

| Field | Value |
|---|---|
| Document ID | SOP-04 |
| Revision | 1.0 |
| Status | **DRAFT — not yet approved** |
| Owner | Managing Director |
| Approved by | `________________________` |
| Approval date | `____ / ____ / ________` |
| ISO 9001:2015 clauses | 8.2.1, 8.2.2, 8.2.3, 8.2.4, 9.1.2 |

---

> **Read this first.** Unlike `SOP-01`, `SOP-02` and `SOP-05`, this procedure does **not** describe
> an established practice. EchoMind Enterprise is early-commercial. There is no contract register,
> no customer feedback log and no record of a completed requirements review anywhere in this
> repository. What follows is the control Ajace AI intends to operate from the point this procedure
> is approved. Nothing in it should be read by an auditor as a description of what has already
> happened. §9 states the position without softening it.

## 1. Purpose

To define how Ajace AI determines, reviews and agrees the requirements for a customer engagement
before committing to it; how changes to those requirements are handled; how Ajace AI communicates
with customers, including on enquiries and complaints; and how customer satisfaction is monitored.

## 2. Scope

All customer engagements for EchoMind Enterprise, comprising:

- **the platform** — a private, on-premises AI workspace (knowledge chat over the customer's own
  documents, live transcription with live fact-checking, boardroom meeting reports, a full-duplex
  voice assistant, and document generation) that runs fully offline on hardware the customer
  controls;
- **customisation** — configuring and extending the platform's building blocks for the customer's
  material, personas, workflows and branding. The five vertical packs (health, law, meetings,
  retail, bank) share one deployment through isolated knowledge-base namespaces — a pattern visible
  in `eval/golden/` where each pack has its own golden-question set, and enforced in code by the
  `_ns_ok` predicate at `backend/app/rag/index.py:36`;
- **advisory** — advice on hardware and cloud fit according to the customer's security posture.

**Out of scope: Ajace AI does not sell hardware.** It advises on what is suitable; the customer
procures it. Any commitment made about hardware is a commitment about advice, not about supply, and
must be recorded as such in the requirements review.

## 3. Responsibilities

| Role | Responsibility |
|---|---|
| Managing Director | Owns this procedure. Is the only role that may commit Ajace AI to a customer requirement, a delivery date or a performance claim. Conducts and records the requirements review. Owns the customer relationship and handles complaints. |
| Lead Engineer | Assesses technical feasibility of each stated requirement before commitment and states, in writing, anything that is not achievable, not yet built, or achievable only with a stated risk. Identifies what verification would be needed to demonstrate a requirement is met. |
| Anyone contributing code | Refers any requirement, promise or complaint received directly from a customer to the Managing Director rather than answering it. |

## 4. Determining requirements (8.2.2)

Requirements are determined before any commitment is made. Four categories must be captured for
every engagement. A requirements record with any category left blank is not complete and the
engagement must not be committed.

| Category | What must be captured | Notes specific to this product |
|---|---|---|
| **Customer-stated requirements** | What the customer says the system must do, for whom, with what material, to what standard | Includes delivery, post-delivery support expectations, and who trains their users |
| **Requirements not stated but necessary for the known intended use** | What the customer has not said but will need for the stated purpose to work | For this product the standing items are: which documents form the knowledge base and who may see each; how many concurrent users; which of the five pack behaviours apply; whether the deployment is air-gapped or network-attached; who administers it after handover |
| **Statutory and regulatory requirements applicable to the product** | Obligations that bind Ajace AI or the delivered system | Data protection; the customer's sector rules where they constrain the system (data residency, record retention, professional-advice boundaries). The lawyer-persona disclaimer behaviour is an example of a sector constraint already implemented and tested — `backend/tests/test_prompt_guards.py:29` |
| **Requirements Ajace AI itself imposes** | What Ajace AI will not deliver without | Grounding and abstention behaviour must not be weakened for a customer; tenant isolation must not be disabled; customer data must remain within the customer's perimeter (`QP-01` §2.1, §2.2) |

**Hardware and cloud fit.** Where the customer asks for advice on hardware or cloud placement, the
requirements record states the security posture the advice was given against (air-gapped,
network-attached but private, or hosted), the recommendation, and the explicit statement that Ajace
AI advises and does not supply. The reference platform is NVIDIA DGX Spark (GB10); the
configuration constraints that apply to it are recorded at `docker-compose.yml:41-47` and
`:108-113`.

**Claims about capability.** No performance figure, accuracy figure or benchmark may be stated to a
customer unless it is reproducible from a recorded run. The figures that exist today come from
`eval/reports/` (golden-question runs) and `eval/paper/results/`. Where an experiment has not been
run it is reported as not run — `eval/paper/results/SUMMARY.json` records E2 as `not_run` with the
reason, and that is the standard (`QP-01` §2.3).

## 5. Review of requirements before commitment (8.2.3)

The Managing Director conducts the review before Ajace AI commits to supply. The review answers six
questions, and the answers are the record.

| # | Question | Evidence expected |
|---|---|---|
| 1 | Are all four categories in §4 captured, including the unstated-but-necessary ones? | The requirements record itself |
| 2 | Can each requirement be met with the platform as it stands, with customisation, or not at all? | Lead Engineer's written feasibility assessment against the capability baseline in `docs/CAPABILITIES.md` |
| 3 | For anything requiring new development, has it been planned under `SOP-05`? | A design plan reference |
| 4 | How will each requirement be shown to be met at handover? | Named acceptance criteria — for a retrieval requirement, a set of golden questions added to `eval/golden/` for that customer's corpus |
| 5 | Do differing requirements from a proposal, a tender and a conversation conflict? If so, which one governs? | The resolved position, in writing, agreed with the customer |
| 6 | Can Ajace AI actually deliver — capacity, competence, hardware availability at the customer? | An honest statement of resource, including the single-person constraint recorded in `SOP-03` §10 |

**The record.** The review is recorded on `forms/FRM-11_Requirements_Review_Record.md` and filed in
`registers/REG-06_Customer_Feedback_and_Complaints_Log.md`. Where the customer's requirements are stated
only verbally, Ajace AI writes them down and obtains the customer's written confirmation before
accepting them; an unconfirmed verbal requirement is not a requirement.

**Nothing is committed before the review is recorded.** A proposal, a quotation and a signature are
each a commitment. The review precedes all three.

## 6. Changes to requirements (8.2.4)

When a requirement changes — whether the customer asks for the change, or Ajace AI discovers that a
requirement as written cannot be met — the following applies.

1. The change is captured in writing against the original requirements record; the original is not
   overwritten, so that the position at commitment remains visible.
2. The change is reviewed against the same six questions in §5, in proportion to its size.
3. The Managing Director confirms the revised position with the customer in writing, including any
   effect on scope, date or price, **before** work proceeds on the changed basis.
4. Where the change alters what the software must do, it becomes a design change and is additionally
   controlled under `SOP-05` §8.
5. Everyone affected is informed of the amended requirement. With one person in the organisation
   this step is trivial today; it is stated because it will not remain trivial.

A change accepted informally and not recorded is a nonconformity under `SOP-14`.

## 7. Customer communication (8.2.1)

| ISO 8.2.1 topic | Channel and rule |
|---|---|
| Information about the product and services | `docs/CAPABILITIES.md` is the internal source of truth for what the platform does; anything stated externally must be consistent with it. `docs/USER_MANUAL.md` (1257 lines, 16 chapters) is the user-facing operating documentation issued with a deployment. |
| Enquiries, contracts, order handling and amendments | Directed to the Managing Director. Every enquiry is logged in `registers/REG-06_Customer_Feedback_and_Complaints_Log.md` on receipt, whether or not it progresses, so that the pipeline and the response times are visible. Amendments follow §6. |
| Customer feedback, including complaints | Logged in `registers/REG-06_Customer_Feedback_and_Complaints_Log.md` — see §8. |
| Handling or controlling customer property | Customer documents, transcripts and audio are customer property under ISO 9001:2015 8.5.3 and are handled under `SOP-11`. They are held only in the deployment's data volumes and are never committed to this repository (`SOP-01` §9). The customer is told in writing where their data resides, that `docker-compose.yml:101-102` mounts volume `echomind_data` as the single store for it, and — while the gap in `SOP-02` §12 remains open — that no backup of that volume is provided by Ajace AI. |
| Specific requirements for contingency actions | Where the engagement warrants it, the requirements record states what happens on a model-serving failure, a hardware failure and an Ajace AI availability failure. The last of these must be stated honestly given the single-person constraint. |

**Public-facing deployment.** A demonstration instance runs at echomind-ajace.com behind Cloudflare
Access (`docs/PUBLIC_DEPLOYMENT.md` step 4, marked MANDATORY). Anyone granted access to it is told
in writing what it is — a demonstration containing seeded demonstration material — and that it must
not be used with real customer data.

## 8. Complaints

A complaint is any expression of dissatisfaction, however it arrives and however informally it is
phrased. It does not need the word "complaint" to be one.

| Step | Who | Timescale | Record |
|---|---|---|---|
| Acknowledge receipt to the customer | Managing Director | 2 working days | Entry in `registers/REG-06_Customer_Feedback_and_Complaints_Log.md` |
| Contain — stop any continuing harm | Lead Engineer | Immediately where data or grounding is affected | The containing change, cited by commit |
| Investigate and determine the cause | Lead Engineer | 10 working days, or a stated longer period agreed with the customer | Nonconformity record under `SOP-14` where the complaint is valid |
| Respond to the customer with the finding and the action | Managing Director | On completion of the investigation | The response, filed against the register entry |
| Close, and check the action worked | Managing Director | At the next management review | `SOP-15` minutes |

A complaint about an ungrounded, fabricated or misattributed answer is treated as the highest
severity class of defect under `QP-01` §2.1, and is investigated whether or not the customer
describes it as serious.

## 9. Monitoring customer satisfaction (9.1.2)

ISO 9001:2015 9.1.2 requires the organisation to monitor customers' perceptions of the degree to
which their needs and expectations have been fulfilled, and to determine the methods for doing so.
The methods determined for Ajace AI are:

| Method | When | Who | Record |
|---|---|---|---|
| Structured handover review at the end of each engagement, against the acceptance criteria agreed in §5 | At handover | Managing Director with the customer | `forms/FRM-12_Customer_Satisfaction_Review.md`, filed in `registers/REG-06` |
| Review call at 3 months and at 12 months after handover | 3 and 12 months | Managing Director | As above |
| Unsolicited feedback — anything the customer says or writes, positive or negative | On receipt | Whoever receives it | `registers/REG-06` |
| Complaints, as a negative satisfaction signal | On receipt | Managing Director | `registers/REG-06` and `SOP-14` |
| Usage signals from the deployed system, where the customer has agreed they may be shared | Continuous | Lead Engineer | Summarised into the management review |

The results are analysed at each management review (`SOP-15`) and feed the quality objectives in
`QO-01`. **Absence of complaints is not evidence of satisfaction and must not be recorded as such.**

## 10. Records

| Record | Location | Retention |
|---|---|---|
| Customer engagement register (enquiries, proposals, engagements) | `registers/REG-06_Customer_Feedback_and_Complaints_Log.md` | 3 years after the engagement ends (SOP-01 §8) |
| Requirements review record, per engagement | `forms/FRM-11_Requirements_Review_Record.md`, filed in REG-06 | As above |
| Record of changes to requirements | Appended to the requirements review record | As above |
| Customer feedback and complaints | `registers/REG-06_Customer_Feedback_and_Complaints_Log.md` | 3 years (SOP-01 §8) |
| Customer satisfaction reviews | `forms/FRM-12_Customer_Satisfaction_Review.md`, filed in REG-06 | 3 years |
| Statement to the customer of where their data resides | Requirements review record | Life of the engagement + 3 years |
| Customer data itself | The deployment's `echomind_data` volume, on the customer's hardware. **Never in this repository** (`SOP-01` §9) | Per the customer contract; `SOP-11` |

## 11. Current state and gaps

**This procedure defines an intended control. It does not describe an established practice.** That
statement is the most important line in the document and it is repeated here deliberately.

| Item | Actual state in the repository |
|---|---|
| Contract register | Does not exist. No contract, order or engagement record is present anywhere in the repository. |
| Requirements review records | None. No engagement has a recorded pre-commitment review. `forms/FRM-11_Requirements_Review_Record.md` is created by this QMS and is unused. |
| Customer feedback log | Does not exist. `registers/REG-06` is created empty. |
| Complaint records | None. No complaint has been recorded; this is not the same as none having been received. |
| Customer satisfaction monitoring | Not performed. No satisfaction review has taken place and no method was in operation before this procedure. |
| CRM or enquiry tracking | None. Enquiries, to the extent they have occurred, are not logged in any system inside the repository. |
| Named customers | None are recorded, and none are named in this QMS. Where an auditor expects a customer name, the field is `________`. |
| Acceptance criteria agreed with a customer | None recorded. The golden-question sets in `eval/golden/` were written by Ajace AI against demonstration material, not agreed with any customer. |

**What does exist** is the commercial and technical basis on which the control will operate: a
working, documented platform (`docs/CAPABILITIES.md`, `docs/USER_MANUAL.md`), a demonstration
deployment gated by Cloudflare Access (`docs/PUBLIC_DEPLOYMENT.md`), a measurable acceptance
mechanism that could be pointed at a customer corpus (`eval/run_eval.py`, 52 golden questions across
the five packs plus a conversational set), and a published evaluation of the approach in the
QASC 2026 paper with its harness at `eval/paper/`.

**Consequences an auditor should note.** Because no requirements review has ever been recorded,
there is at present no evidence that Ajace AI reviews requirements before commitment. Clause 8.2.3
is therefore **not met** today; it becomes capable of being met at the first engagement conducted
after this procedure is approved. The same applies to 9.1.2: the method is now determined, but no
data has been gathered.

All gaps above are carried into `ISO9001_Gap_Analysis.md`.
