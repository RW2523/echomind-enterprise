# EchoMind Law — demo flows

Everything below runs on the pilot stack with **fictional** demo data (D-061, D-064), except the reference library
R-001, which holds **real public law**, labelled as such.

## Before you start

- **Open** `https://localhost:3553` on the server, or `https://<server address>:3553` on the office network, and accept
  the self-signed certificate. Use the `localhost` address for Conversation: browsers may refuse the microphone on a
  network address with a self-signed certificate.
- **Sign in** with each demo person's password and current code, which this command shows on the server:
  `python3 scripts/law/demo_signin.py` (codes change every 30 seconds, and each code works once).
- **Jump anywhere** with **⌘K / Ctrl K**. Type a matter number, a client or a tool, then press Enter.
- **The first recording or spoken question after a restart** waits while speech recognition loads, for a minute or two.

## People

| Sign in as | Role | Works on |
|---|---|---|
| **harper** | Attorney, senior associate (disputes) | leads M-301, M-305; on M-130, M-303 |
| **alice** | Attorney | leads M-101, M-114, M-130 |
| **sofia** | Attorney, associate (employment, real estate) | leads M-302, M-304; on M-306 |
| **kenji** | Attorney, associate (corporate, regulatory) | leads M-303, M-306; on M-305 |
| **lena** | Attorney, trainee | M-301, M-302, M-304 |
| **omar** | Paralegal | M-301, M-303, M-304, M-306 |
| **carol** | Paralegal | M-101, M-130 |
| **theo** | Reviewer (knowledge lawyer) | M-302, M-305 |
| **nakamura** | Managing partner | approves on M-101, M-130, M-301, M-305 |
| **morgan** | Managing partner | templates and playbooks; R-001; approves on M-303, M-306 |
| **bob** | Attorney | leads M-202 (highly restricted), M-215 |
| **iris** | Front desk | enquiries |
| **conway** | Conflicts team | conflict decisions |
| **fiona** | Firm administrator | users, connectors, audit |
| **rhea** | Records administrator | holds, deletion |

Every lawyer can read **R-001**.

## Matters

| Matter | What it shows |
|---|---|
| **M-101** Harrowgate MSA renewal | Research, Conversation, meetings with Silent Assist (the cap is USD 250,000), the review panel (change of control), summaries |
| **M-114** Series B due diligence | Due diligence checklist and governance checks across a board pack |
| **M-130** Harrowgate base corpus | Compare the MSA's signed v1 with its draft v2; conflicts between the side letter and the MSA; obligations |
| **M-301** Northwind v Calder Freight | Claim notice deadlines (April's was late), force majeure, a letter before claim, time entries |
| **M-302** Brightwater Clinics | GBP 18,400 claimed against GBP 14,080 by the policy rate; the non-compete |
| **M-303** Kestrel Robotics Series A | Conflicts: option pool 12% (term sheet) against 10% (investor draft), closing 30 October against 15 November |
| **M-304** Harbourview Studios lease | Rent review (GBP 128,000 proposed against GBP 108,000–112,000 comparables), break notice deadline |
| **M-305** Lumen Software and Orbis Bank | Negotiation: 72-hour against 24-hour breach notice; unlimited liability asked for |
| **M-306** Meridian Foods inquiry | 21-day regulatory deadline, a recall memo, summaries |
| **M-202 / M-215** | Isolation: M-202 is highly restricted and invisible to anyone not on it |
| **R-001** Reference library | Real US and UK statutes, for research |

## Flows

### 1. The first five minutes (harper, 3 min)
1. Sign in as **harper**. **Home** greets you, with quick actions, counts, your matters as cards, and what is waiting for you.
2. Press **Ctrl K** and type `calder`. Choose **M-301 · Northwind v Calder Freight**.
3. The **Overview** shows the matter and **Work on this matter**, one tile per tool.

**You should see:** five matters on Home; the palette finding M-301; twelve or more tool tiles.

### 2. Ask the documents, then talk to them (harper, M-301, 5 min)
1. **Research**: ask *"Within what period must the Customer notify a claim, and was the April claim on time?"*
   - **You should see:** 30 days after the end of the month, so April's notice was late (28 July). Each statement has
     its exact quote (**Quoted exactly**) and opens in the document.
2. **Conversation**: press the microphone and ask *"What did Calder say about force majeure?"*. Press again.
   - **You should see:** the answer, read aloud, with its quotes.
3. Then ask *"And does the agreement cover congestion?"*.
   - **You should see:** *"You asked … understood from the conversation as: Does the Logistics Services Agreement's
     force majeure clause cover port congestion?"*, answered from clause 14. **New conversation** forgets it.

### 3. A live meeting with Silent Assist (alice, M-101, 6 min)
1. **Meetings**: **Plan a meeting** (client call), then add **Alice Hart** (host) and **Petra Voss** (client).
2. **Consent**: record each person's consent. Recording stays blocked until everyone has consented.
3. **Record**: say as Petra *"As I understand it, the liability cap under clause eight is five hundred thousand
   dollars."* Then say as Alice *"I will check clause eight and send you a note by Friday."*
4. **You should see:** the live transcript, and a Silent Assist alert **"Differs from signed agreement, cl. 8.1"**
   (USD 250,000) and an **action** alert. Decide each alert (a dismissal needs a reason).
5. **Stop**, then **Minutes**: draft them, then **nakamura** approves them from **Reviews**.

### 4. Compare and conflicts (kenji, M-303; alice, M-130, 4 min)
1. As **kenji**, open M-303, then **Compare**, then **Scan for conflicts** (latest drafts).
   - **You should see:** the option pool **12%** against **10%**, and closing **30 October 2026** against
     **15 November 2026**, each with both documents quoted.
2. As **alice**, open M-130, then **Compare**: the Harrowgate MSA **v1 → v2**.
   - **You should see:** the term shortened from 36 to 24 months, the cap cut to USD 150,000, and clauses moved
     or removed, each explained.

### 5. Draft, review, file (harper → nakamura, M-301, 5 min)
1. **Drafting**: start from the **Letter before claim** template.
   - **You should see:** fields filled from the matter with their sources; anything unsupported left as a gap.
2. Edit it, then **Send for review**. As **nakamura**, open **Reviews** and **Approve** it; it is filed as Word.

### 6. Summaries and the review panel (kenji M-306; harper M-305 → nakamura, 6 min)
1. As **kenji**, open M-306, then **Summaries**: summarise the **Internal compliance memo**.
   - **You should see:** each point cited by clause and line.
2. As **harper**, open M-305, then **Review panel**: ask *"Should Lumen accept unlimited liability for data protection
   breaches?"* (specialist: technology and data).
   - **You should see:** four perspectives, each citing the documents, and a disagreement matrix.
   - If the panel stops for missing facts, give them or **Continue without these facts** with a reason.
   - **nakamura** approves.

### 7. Check the numbers (sofia, M-302, 2 min)
**Research**: ask *"How much unpaid on-call allowance is Dr Rahman owed for 2025, and how was it calculated?"*

**You should see:** 64 nights at GBP 220 = **GBP 14,080**, against the **GBP 18,400** claimed.

### 8. Intake and conflicts (iris → conway, 4 min)
1. As **iris**, open **Enquiries**, then **New enquiry** (phone). Read the disclaimer. Enter a fictional enquirer with
   an email, matter type *Commercial*, and other party **Calder Freight Ltd**.
   - **You should see:** only *"refer to the conflicts team"*: no client or matter is named to the front desk.
2. As **conway**, open the enquiry.
   - **You should see:** the matches (M-301, adverse; M-215, related). Decide **cleared** or **conflict**.

### 9. The reference library (any lawyer, R-001, 2 min)
**Research** in R-001: ask *"Under the Unfair Contract Terms Act 1977, can a business exclude liability for death or
personal injury caused by negligence?"*

**You should see:** the answer quoted from the Act, marked as real public law (OGL).

### 10. Time and reviews (harper; nakamura, 3 min)
1. As **harper**, open **Time**, then **Suggest from my activity**. Edit a narrative, **Approve** it, then **Download
   approved** (CSV).
2. As **nakamura**, open **Reviews**: everything waiting for a decision, by matter.

### 11. Governance and safety (morgan, fiona, rhea, 5 min)
- As **morgan**: open **Ethical walls** (two-person approval) and **Emergency access** (break-glass, audited).
- As **fiona**: open the **Audit trail**, a hash-chained, verifiable ledger. It records counts and flags, never matter
  content.
- As **rhea**: open **Records**: legal holds and deletion with approval.
- As **anyone but bob**: M-202 does not exist; a link to it says *Not available*.

### 12. Regulatory updates (sofia, M-302; any lawyer, 3 min)
1. Open **Regulatory updates** in the sidebar.
   - **You should see:** new UK legislation and US rules from the last 30 days, fetched this morning from
     legislation.gov.uk and the Federal Register, each labelled as real public law with its licence. Updates that may
     concern your matters list them under **May concern**.
2. Tick **Only updates that may concern my matters**, then open a matter chip (for example **M-302**).
3. On the matter's **Regulatory** page, each update shows **Matched on:** the practice-area words found. Choose **Mark
   reviewed**, or **Not relevant…**, which needs a reason.
   - **You should see:** the decision with your name and the date. **Reopen** undoes it.
- Matches depend on what was published: on a quiet day there may be none for a matter.

### 13. Translation (alice, M-130; kenji, M-306, 3 min)
1. As **alice**, open M-130, then a document (for example the side letter), then **Translate**. Choose **French**, then
   **Translate**.
   - **You should see:** the label *"Machine translation by this server's model, not reviewed by anyone. The original
     prevails…"*, then each paragraph beside its translation, a few at a time. A figure missing from a translation is
     flagged (**Check: … is not in the translation**).
2. **Download (.txt)** saves both texts with the label. For a long document, **Continue** picks up where it stopped.
3. As **kenji**, open M-306, then **Summaries**, then the compliance memo's summary (made in flow 6), then **Translate into**: **German**.
   - **You should see:** each point's translation under it, with the same label.

