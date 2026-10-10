# EchoMind Law drafting evaluation — heldout

- Commit: `ff23019`
- Model: `nvidia/Qwen3-30B-A3B-FP4`
- Run: 20261010-134751

| Measure | Result |
|---|---|
| scenarios | 3 |
| field accuracy | 9/10 |
| absent left open | 1/1 |
| wrong values | 0 |
| invented figures | 0 |
| written passages | 4 |
| unsupported sentences | 5 |
| seconds per draft | 16.3 |

## nda-heron-short-note (nda, 6.5 s, 2 gaps)

- party_a: ok — sourced: 'Heron Robotics Ltd' (expected 'Heron Robotics')
- party_b: **missed** — needs_input: None (expected 'Calder Mining') — suggested 'Heron Robotics Ltd', not used: the same party is already given as First party
- purpose: ok — sourced: 'to evaluate an autonomous drilling pilot' (expected 'autonomous drilling')
- confidentiality_period: ok — sourced: 'three (3) years' (expected 'three')
- governing_law: ok — sourced: 'English law' (expected 'English')

## engagement-calder-hourly (engagement_letter, 8.2 s, 4 gaps)

- client_address: ok — sourced: '3 Harbour Walk, Aberdeen AB11 5BN' (expected '3 Harbour Walk')
- client_contact: ok — sourced: 'Fiona Baird' (expected 'Fiona Baird')
- fee_arrangement: ok — sourced: 'GBP 450 for partners and GBP 300 for associates' (expected '450')
- written (2 sources, 1 unsupported): The client has requested legal services related to the preparation of pilot contracts with Heron Robotics Ltd, as specified in the engagement letter. The legal services include the drafting and review of these contracts, with the client confirming the fee arrangement for the work, which is GBP 450 for partners and GBP 300 for associates. [[NEEDS INPUT: check against the documents]] The matter number assigned to this engagement is EVAL-D2, and the client's contact is Fiona Baird, Legal Director, at 3 Harbour Walk, Aberdeen AB11 5BN.

## review-heron-services-no-law (contract_review, 34.2 s, 6 gaps)

- agreement: ok — sourced: 'SERVICES AGREEMENT (fictional)' (expected 'Services Agreement')
- parties: ok — sourced: 'Heron Robotics Ltd' (expected 'Heron Robotics')
- governing_law (not in the documents): left open
- written (3 sources, 0 unsupported): The agreement is a Services Agreement between Heron Robotics Ltd (the Provider) and Calder Mining plc (the Client). The term of the agreement is not specified in the provided text. The Provider shall supply the drilling automation services described in Schedule 1. Each party's total liability is limited to GBP 2,000,000. The manner in which each party may terminate the agreement is not specified in the provided text.
- written (2 sources, 2 unsupported): The Services Agreement specifies that the Provider shall supply the drilling automation services described in Schedule 1. The total liability of each party is limited to GBP 2,000,000. [[NEEDS INPUT: check against the documents]] The fees and payment terms are not specified in the provided text. [[NEEDS INPUT: check against the documents]] The matter number is EVAL-D3, and the agreement is dated 10 October 2026.
- written (1 sources, 2 unsupported): The Services Agreement limits the liability of the Provider to GBP 2,000,000 in total. [[NEEDS INPUT: check against the documents]] This limitation applies to all claims arising under this Agreement, including but not limited to claims for breach of contract, negligence, or other tortious acts. The Provider is not liable for any indirect, special, incidental, or consequential damages, including but not limited to loss of profits, data, or business opportunities. [[NEEDS INPUT: check against the documents]] The Client acknowledges that the Provider has relied on this limitation of liability in entering into this Agreement. The Client also acknowledges that the Provider would not have entered into this Agreement without this limitation of liability. The limitation of liability set forth in this Section 9.1 shall survive the termination or expiration of this Agreement. The Client agrees that the limitation of liability is reasonable and reflects the nature of the services provided. The Client further agrees that the limitation of liability is not a failure of consideration or a breach of any implied or express warranty. The Client acknowledges that the limitation of liability is an essential term of this Agreement and that the Client has had the opportunity to negotiate its terms. The Client further acknowledges that the limitation of liability is not a bar to the Client's right to seek remedies for the Provider's willful misconduct or gross negligence. The Client agrees that the limitation of liability shall not apply to any claims for personal injury or death. The Client also agrees that the limitation of liability shall not apply to any claims for breach of confidentiality or intellectual property rights. The Client acknowledges that the limitation of liability is not a substitute for insurance and that the Client should maintain its own insurance coverage. The Client further acknowledges that the limitation of liability does not affect the Client's right to seek injunctive relief or other equitable remedies. The Client agrees that the limitation of liability is not a waiver of any rights the Client may have under applicable law. The Client also agrees that the limitation of liability does not affect the Client's right to claim damages for the Provider's failure to perform the services as described in Schedule 1. The Client acknowledges that the limitation of liability is not a limitation of the Provider's obligation to perform the services as described in Schedule 1. The Client further acknowledges that the limitation of liability does not affect the Provider's obligation to provide the services in a professional and workmanlike manner. The Client agrees that the limitation of liability is not a defense to any claim for breach of contract [9.1:
