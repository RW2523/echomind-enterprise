# EchoMind Law drafting evaluation — heldout

- Commit: `97bc690+changes`
- Model: `nvidia/Qwen3-30B-A3B-FP4`
- Run: 20261010-055810

| Measure | Result |
|---|---|
| scenarios | 3 |
| field accuracy | 8/10 |
| absent left open | 2/2 |
| wrong values | 0 |
| invented figures | 0 |
| written passages | 4 |
| unsupported sentences | 8 |
| seconds per draft | 12.1 |

## demand-northwind (demand_letter, 14.8 s, 3 gaps)

- recipient: ok — sourced: 'Northwind Logistics Ltd' (expected 'Northwind Logistics')
- recipient_address: ok — sourced: 'Northwind Logistics Ltd, 2 Harbour Road, Hull HU1 1AA, United Kingdom' (expected '2 Harbour Road')
- agreement: **missed** — needs_input: None (expected 'Supply Agreement')
- amount_claimed: ok — sourced: 'GBP 42,750' (expected '42,750')
- written (5 sources, 0 unsupported): The Supplier entered into a Supply Agreement with the Customer, Northwind Logistics Ltd. Notices to the Customer are to be sent to Northwind Logistics Ltd, 2 Harbour Road, Hull HU1 1AA, United Kingdom. Under the terms of the agreement, the Customer is required to pay each invoice within thirty (30) days of its date.  

The Customer has failed to pay two invoices, HS-1041 and HS-1042, issued on 3 June 2026 and 1 July 2026 respectively. The total amount outstanding is GBP 42,750. The Client has instructed that this matter be pursued, and the amount claimed is GBP 42,750 [CONFIRMED DETAILS: Amount claimed: GBP 42,750]. The claim is being made on 10 October 2026 [CONFIRMED DETAILS: Date: 10 October 2026].

## review-warehouse-lease (contract_review, 17.1 s, 2 gaps)

- agreement: **missed** — needs_input: None (expected 'Lease')
- parties: ok — sourced: 'Elmfield Estates Ltd and Alder Biotech AB' (expected 'Elmfield Estates')
- governing_law: ok — sourced: 'the law of England and Wales' (expected 'England and Wales')
- written (3 sources, 0 unsupported): The term of the agreement is ten (10) years from 1 April 2025. The Tenant may terminate this Lease on 31 March 2030 by giving not less than six months' written notice. The governing law of the agreement is the law of England and Wales.
- written (0 sources, 5 unsupported): The rent under the Lease is GBP 84,000 per year, payable quarterly in advance. The term of the Lease is ten (10) years from 1 April 2025. The Tenant may terminate the Lease on 31 March 2030 by giving not less than six months' written notice. The Landlord's total liability under the Lease shall not exceed the rent paid in the preceding twelve months. This Lease is governed by the law of England and Wales.
- written (2 sources, 3 unsupported): The Landlord's liability under this Lease is limited to the rent paid in the preceding twelve months. There are no other exclusions of liability specified in the Lease. The governing law of the Lease is the law of England and Wales. The matter number is EVAL-D2. The parties to the Lease are Elmfield Estates Ltd and Alder Biotech AB. The date of the Lease is 10 October 2026.

## nda-missing-period (nda, 4.5 s, 2 gaps)

- party_a: ok — sourced: 'Harrowgate Systems Inc.' (expected 'Harrowgate')
- party_b: ok — sourced: 'Elmfield Estates Ltd' (expected 'Elmfield')
- purpose: ok — sourced: 'to assess a possible warehouse automation project' (expected 'warehouse automation')
- confidentiality_period (not in the documents): left open
- governing_law (not in the documents): left open
