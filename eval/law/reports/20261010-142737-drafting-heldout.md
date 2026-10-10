# EchoMind Law drafting evaluation — heldout

- Commit: `e85b90a`
- Model: `nvidia/Qwen3-30B-A3B-FP4`
- Run: 20261010-142737

| Measure | Result |
|---|---|
| scenarios | 3 |
| field accuracy | 9/10 |
| absent left open | 1/1 |
| wrong values | 0 |
| invented figures | 0 |
| written passages | 4 |
| unsupported sentences | 6 |
| seconds per draft | 9.5 |

## nda-heron-short-note (nda, 6.6 s, 2 gaps)

- party_a: ok — sourced: 'Heron Robotics Ltd' (expected 'Heron Robotics')
- party_b: **missed** — needs_input: None (expected 'Calder Mining') — suggested 'Heron Robotics Ltd', not used: the same party is already given as First party
- purpose: ok — sourced: 'to evaluate an autonomous drilling pilot' (expected 'autonomous drilling')
- confidentiality_period: ok — sourced: 'three (3) years' (expected 'three')
- governing_law: ok — sourced: 'English law' (expected 'English')

## engagement-calder-hourly (engagement_letter, 7.5 s, 3 gaps)

- client_address: ok — sourced: '3 Harbour Walk, Aberdeen AB11 5BN' (expected '3 Harbour Walk')
- client_contact: ok — sourced: 'Fiona Baird' (expected 'Fiona Baird')
- fee_arrangement: ok — sourced: 'GBP 450 for partners and GBP 300 for associates' (expected '450')
- written (3 sources, 0 unsupported): The client has instructed the firm to prepare the pilot contracts with Heron Robotics Ltd, as stated in the letter of instruction. The client’s contact for this work is Fiona Baird, Legal Director, as confirmed in the letter of instruction. The firm’s fee arrangement for this matter is GBP 450 for partners and GBP 300 for associates, as specified in the engagement letter.

## review-heron-services-no-law (contract_review, 14.5 s, 9 gaps)

- agreement: ok — sourced: 'SERVICES AGREEMENT' (expected 'Services Agreement')
- parties: ok — sourced: 'Heron Robotics Ltd (the Provider) and Calder Mining plc (the Client)' (expected 'Heron Robotics')
- governing_law (not in the documents): left open
- written (2 sources, 4 unsupported): [[NEEDS INPUT: check against the documents]] The Services Agreement between Heron Robotics Ltd (the Provider) and Calder Mining plc (the Client) is in effect from 10 October 2026. The Provider shall supply the drilling automation services described in Schedule 1. [[NEEDS INPUT: check against the documents]] The term of the agreement is not specified in the provided text. Each party's total liability is limited to GBP 2,000,000. [[NEEDS INPUT: check against the documents]] The manner in which either party may terminate the agreement is not specified in the provided text. [[NEEDS INPUT: check against the documents]] The specific clauses governing termination are not included in the provided text.
- written (2 sources, 1 unsupported): The Provider shall supply the drilling automation services described in Schedule 1. Each party's total liability is limited to GBP 2,000,000. [[NEEDS INPUT: check against the documents]] The fees and payment terms are not specified in the provided text [NEEDS INPUT: fees and payment terms].
- written (1 sources, 1 unsupported): The Services Agreement limits the total liability of each party to GBP 2,000,000. [[NEEDS INPUT: check against the documents]] There are no specific exclusions of liability set out in the clauses provided. [[NEEDS INPUT: whether there are any other exclusions of liability not mentioned in the provided text]].
