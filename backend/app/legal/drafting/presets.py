"""Standard presets (S3.1, DOC-01; decision D-045).

Starting points for the firm's own precedents, installed as drafts. A managing partner reviews and
approves each before anyone drafts from it (P-008). Every fact is a field: from the matter's
documents with a checked quote, from the matter record, or entered by a lawyer. The passages the
model writes ([[write: …]]) cite the documents and may not add facts.
"""
from __future__ import annotations

_DOC = "documents"

PRESETS = [
    {
        "key": "engagement_letter", "kind": "engagement_letter", "name": "Engagement letter",
        "body": """# Engagement letter

{{client}}
{{client_address}}

Dear {{client_contact}},

## Our engagement
Thank you for instructing us on matter {{matter_number}}, {{matter_title}}.

[[write: Describe the scope of the legal services the client has asked for, in two or three sentences, from the engagement documents.]]

## Fees
Our fees for this matter are {{fee_arrangement}}. We will invoice {{billing_frequency}}.

## Responsible lawyer
{{responsible_lawyer}} will be responsible for this matter.

## Terms
Our standard terms of business apply to this engagement. Please sign and return a copy of this letter to confirm your instructions.

Yours sincerely,

{{responsible_lawyer}}
{{today}}""",
        "fields": [
            {"key": "client", "label": "Client", "kind": "party", "source": "matter"},
            {"key": "client_address", "label": "Client address", "kind": "address", "source": _DOC, "question": "What is the client's postal address?"},
            {"key": "client_contact", "label": "Client contact", "kind": "party", "source": _DOC,
             "question": "Who is the client's contact person for this matter: the person who gave the instructions, or to whom letters are to be sent?"},
            {"key": "matter_number", "label": "Matter number", "kind": "text", "source": "matter"},
            {"key": "matter_title", "label": "Matter", "kind": "text", "source": "matter"},
            {"key": "fee_arrangement", "label": "Fee arrangement", "kind": "money", "source": _DOC, "question": "What fee or fee arrangement was agreed for this matter?"},
            {"key": "billing_frequency", "label": "Billing frequency", "kind": "text", "source": "person"},
            {"key": "responsible_lawyer", "label": "Responsible lawyer", "kind": "party", "source": "person"},
            {"key": "today", "label": "Date", "kind": "date", "source": "today"},
        ],
    },
    {
        "key": "nda", "kind": "nda", "name": "Mutual non-disclosure agreement",
        "body": """# Mutual non-disclosure agreement

This agreement is made on {{today}} between {{party_a}} and {{party_b}} (each a "Party").

## 1. Purpose
The Parties wish to exchange confidential information for the purpose of {{purpose}} (the "Purpose").

## 2. Confidential information
Each Party shall keep the other Party's confidential information confidential, use it only for the Purpose, and disclose it only to those of its employees and advisers who need to know it for the Purpose and are bound by obligations of confidentiality.

## 3. Term
The obligations in this agreement continue for {{confidentiality_period}} from the date of this agreement.

## 4. Governing law
This agreement is governed by the law of {{governing_law}}.

Signed for {{party_a}}: ______________________

Signed for {{party_b}}: ______________________""",
        "fields": [
            {"key": "party_a", "label": "First party", "kind": "party", "source": _DOC, "question": "Who is the first party to the proposed confidentiality agreement (usually the client)?"},
            {"key": "party_b", "label": "Second party", "kind": "party", "source": _DOC, "question": "Who is the other party to the proposed confidentiality agreement?"},
            {"key": "purpose", "label": "Purpose", "kind": "text", "source": _DOC, "question": "For what purpose will the parties exchange confidential information?"},
            {"key": "confidentiality_period", "label": "Confidentiality period", "kind": "period", "source": _DOC, "question": "For how long must the information be kept confidential?"},
            {"key": "governing_law", "label": "Governing law", "kind": "text", "source": _DOC, "question": "Which law governs the agreement?"},
            {"key": "today", "label": "Date", "kind": "date", "source": "today"},
        ],
    },
    {
        "key": "demand_letter", "kind": "demand_letter", "name": "Letter before claim",
        "body": """# Letter before claim

{{recipient}}
{{recipient_address}}

{{today}}

Dear Sirs,

## Our client: {{client}}
We act for {{client}} in relation to {{agreement}}.

## The facts
[[write: Set out the relevant facts and the breach complained of, from the documents, in neutral terms.]]

## What our client requires
Our client requires payment of {{amount_claimed}} within {{deadline}} of the date of this letter.

If our client's requirements are not met within that time, our client reserves its rights, including the right to issue proceedings without further notice.

Yours faithfully,

{{responsible_lawyer}}""",
        "fields": [
            {"key": "recipient", "label": "Recipient", "kind": "party", "source": _DOC, "question": "Who is the counterparty the letter is addressed to?"},
            {"key": "recipient_address", "label": "Recipient address", "kind": "address", "source": _DOC, "question": "What is the counterparty's address for notices?"},
            {"key": "client", "label": "Client", "kind": "party", "source": "matter"},
            {"key": "agreement", "label": "Agreement", "kind": "text", "source": _DOC, "question": "Which agreement between the parties is in dispute?"},
            {"key": "amount_claimed", "label": "Amount claimed", "kind": "money", "source": _DOC, "question": "What amount is owed to the client?"},
            {"key": "deadline", "label": "Deadline", "kind": "period", "source": "person"},
            {"key": "responsible_lawyer", "label": "Responsible lawyer", "kind": "party", "source": "person"},
            {"key": "today", "label": "Date", "kind": "date", "source": "today"},
        ],
    },
    {
        "key": "memo", "kind": "memo", "name": "Internal memorandum",
        "body": """# Memorandum

To: {{to}}
From: {{from}}
Date: {{today}}
Matter: {{matter_number}} — {{matter_title}}

## Question
{{question}}

## Short answer
[[write: Answer the question briefly from the matter's documents, saying what they state and where they are silent.]]

## Analysis
[[write: Explain the answer, quoting the relevant clauses of the matter's documents.]]

## Next steps
{{next_steps}}""",
        "fields": [
            {"key": "to", "label": "To", "kind": "party", "source": "person"},
            {"key": "from", "label": "From", "kind": "party", "source": "person"},
            {"key": "matter_number", "label": "Matter number", "kind": "text", "source": "matter"},
            {"key": "matter_title", "label": "Matter", "kind": "text", "source": "matter"},
            {"key": "question", "label": "Question", "kind": "text", "source": "person"},
            {"key": "next_steps", "label": "Next steps", "kind": "text", "source": "person"},
            {"key": "today", "label": "Date", "kind": "date", "source": "today"},
        ],
    },
    {
        "key": "contract_review", "kind": "contract_review", "name": "Contract review report",
        "body": """# Contract review report

Matter: {{matter_number}} — {{matter_title}}
Agreement: {{agreement}}
Parties: {{parties}}
Date: {{today}}

## Term and termination
[[write: Summarise the term of the agreement and how each party may terminate it, quoting the clauses.]]

## Fees and payment
[[write: Summarise the fees and payment terms, quoting the clauses.]]

## Liability
[[write: Summarise the limitation and exclusion of liability, quoting the clauses.]]

## Governing law and disputes
The agreement is governed by {{governing_law}}.

## Points for the client
{{review_points}}""",
        "fields": [
            {"key": "matter_number", "label": "Matter number", "kind": "text", "source": "matter"},
            {"key": "matter_title", "label": "Matter", "kind": "text", "source": "matter"},
            {"key": "agreement", "label": "Agreement", "kind": "text", "source": _DOC, "question": "What is the title of the agreement under review?"},
            {"key": "parties", "label": "Parties", "kind": "party", "source": _DOC, "question": "Who are the parties to the agreement?"},
            {"key": "governing_law", "label": "Governing law", "kind": "text", "source": _DOC, "question": "Which law governs the agreement?"},
            {"key": "review_points", "label": "Points for the client", "kind": "text", "source": "person"},
            {"key": "today", "label": "Date", "kind": "date", "source": "today"},
        ],
    },
    {
        "key": "board_minutes", "kind": "board_minutes", "name": "Board minutes",
        "body": """# Minutes of a meeting of the board of directors of {{company}}

Held on {{meeting_date}} at {{meeting_place}}.

Present: {{present}}
In attendance: {{in_attendance}}

## 1. Quorum
The chair noted that a quorum was present.

## 2. Business
[[write: Record the business considered and the decisions taken, from the meeting papers and notes, quoting them.]]

## 3. Close
There being no further business, the meeting was closed.

Chair: ______________________""",
        "fields": [
            {"key": "company", "label": "Company", "kind": "party", "source": _DOC, "question": "Which company's board held the meeting?"},
            {"key": "meeting_date", "label": "Meeting date", "kind": "date", "source": _DOC, "question": "On what date was the board meeting held?"},
            {"key": "meeting_place", "label": "Place", "kind": "address", "source": _DOC, "question": "Where was the board meeting held?"},
            {"key": "present", "label": "Directors present", "kind": "party", "source": _DOC, "question": "Which directors were present at the meeting?"},
            {"key": "in_attendance", "label": "In attendance", "kind": "party", "source": "person"},
        ],
    },
]
