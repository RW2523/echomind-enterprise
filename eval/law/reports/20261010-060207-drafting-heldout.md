# EchoMind Law drafting evaluation — heldout

- Commit: `97bc690+changes`
- Model: `nvidia/Qwen3-30B-A3B-FP4`
- Run: 20261010-060207

| Measure | Result |
|---|---|
| scenarios | 3 |
| field accuracy | 9/10 |
| absent left open | 2/2 |
| wrong values | 0 |
| invented figures | 0 |
| written passages | 4 |
| unsupported sentences | 24 |
| seconds per draft | 13.0 |

## demand-northwind (demand_letter, 17.2 s, 5 gaps)

- recipient: ok — sourced: 'Northwind Logistics Ltd' (expected 'Northwind Logistics')
- recipient_address: ok — sourced: 'Northwind Logistics Ltd, 2 Harbour Road, Hull HU1 1AA, United Kingdom' (expected '2 Harbour Road')
- agreement: ok — sourced: 'the Supply Agreement' (expected 'Supply Agreement')
- amount_claimed: ok — sourced: 'GBP 42,750' (expected '42,750')
- written (5 sources, 3 unsupported): Northwind Logistics Ltd is the Customer under the Supply Agreement, which was executed between Harrowgate Systems Inc. (the Supplier) and Northwind Logistics Ltd (the Customer). Notices to the Customer are to be sent to Northwind Logistics Ltd, 2 Harbour Road, Hull HU1 1AA, United Kingdom. Under the terms of the Supply Agreement, the Customer undertook to pay each invoice within thirty (30) days of its date. Invoices HS-1041 and HS-1042, issued on 3 June 2026 and 1 July 2026, remain unpaid, with the total outstanding amount being GBP 42,750,. [[NEEDS INPUT: check against the documents]] The Client has instructed that a claim is to be made for the unpaid invoices, with the date of the claim being 10 October 2026. [[NEEDS INPUT: check against the documents]] The Client has confirmed that the amount claimed is GBP 42,750. [[NEEDS INPUT: check against the documents]] The breach complained of is the failure by Northwind Logistics Ltd to pay the invoices within thirty (30) days of their respective dates, as required by the Supply Agreement.

## review-warehouse-lease (contract_review, 17.4 s, 23 gaps)

- agreement: **missed** — needs_input: None (expected 'Lease')
- parties: ok — sourced: 'Elmfield Estates Ltd and Alder Biotech AB' (expected 'Elmfield Estates')
- governing_law: ok — sourced: 'the law of England and Wales' (expected 'England and Wales')
- written (3 sources, 1 unsupported): The term of the agreement is ten (10) years, commencing on 1 April 2025. The Tenant may terminate the Lease by providing not less than six months' written notice, with the termination date being 31 March 2030. [[NEEDS INPUT: check against the documents]] Either party may terminate the Lease in accordance with the provisions of the agreement, but no specific further termination rights are outlined in the provided text [1: ""].

The governing law of the agreement is the law of England and Wales.
- written (0 sources, 15 unsupported): [[NEEDS INPUT: check against the documents]] [[NEEDS INPUT: check against the documents]] The rent under the Lease is GBP 84,000 per year, payable quarterly in advance. [[NEEDS INPUT: check against the documents]] [[NEEDS INPUT: check against the documents]] The term of the Lease is ten (10) years from 1 April 2025. [[NEEDS INPUT: check against the documents]] [[NEEDS INPUT: check against the documents]] The Tenant may terminate the Lease on 31 March 2030 by giving not less than six months' written notice. [[NEEDS INPUT: check against the documents]] [[NEEDS INPUT: check against the documents]] The Landlord's total liability under the Lease shall not exceed the rent paid in the preceding twelve months. [[NEEDS INPUT: check against the documents]] [[NEEDS INPUT: check against the documents]] This Lease is governed by the law of England and Wales. [[NEEDS INPUT: check against the documents]] The parties to the Lease are Elmfield Estates Ltd and Alder Biotech AB [source not available: "the Landlord" and "the Tenant" refer to these parties]. [[NEEDS INPUT: check against the documents]] [[NEEDS INPUT: check against the documents]] The matter number is EVAL-D2. [[NEEDS INPUT: check against the documents]] [[NEEDS INPUT: check against the documents]] The date of the Lease is 10 October 2026.
- written (0 sources, 5 unsupported): [[NEEDS INPUT: check against the documents]] [[NEEDS INPUT: check against the documents]] The Landlord's liability under this Lease is limited to the rent paid in the preceding twelve months. [[NEEDS INPUT: check against the documents]] There are no other exclusions or limitations of liability specified in the Lease. [[NEEDS INPUT: check against the documents]] [[NEEDS INPUT: check against the documents]] The Lease is governed by the law of England and Wales.

## nda-missing-period (nda, 4.4 s, 2 gaps)

- party_a: ok — sourced: 'Harrowgate Systems Inc.' (expected 'Harrowgate')
- party_b: ok — sourced: 'Elmfield Estates Ltd' (expected 'Elmfield')
- purpose: ok — sourced: 'assess a possible warehouse automation project' (expected 'warehouse automation')
- confidentiality_period (not in the documents): left open
- governing_law (not in the documents): left open
