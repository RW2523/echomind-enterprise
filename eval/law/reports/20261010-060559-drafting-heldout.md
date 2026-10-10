# EchoMind Law drafting evaluation — heldout

- Commit: `97bc690+changes`
- Model: `nvidia/Qwen3-30B-A3B-FP4`
- Run: 20261010-060559

| Measure | Result |
|---|---|
| scenarios | 3 |
| field accuracy | 8/10 |
| absent left open | 2/2 |
| wrong values | 0 |
| invented figures | 0 |
| written passages | 4 |
| unsupported sentences | 4 |
| seconds per draft | 12.5 |

## demand-northwind (demand_letter, 17.9 s, 7 gaps)

- recipient: ok — sourced: 'Northwind Logistics Ltd' (expected 'Northwind Logistics')
- recipient_address: ok — sourced: 'Northwind Logistics Ltd, 2 Harbour Road, Hull HU1 1AA, United Kingdom' (expected '2 Harbour Road')
- agreement: **missed** — needs_input: None (expected 'Supply Agreement')
- amount_claimed: ok — sourced: 'GBP 42,750' (expected '42,750')
- written (5 sources, 4 unsupported): The Supplier entered into a Supply Agreement with the Customer, Northwind Logistics Ltd. Notices to the Customer are to be sent to the address of Northwind Logistics Ltd, 2 Harbour Road, Hull HU1 1AA, United Kingdom. Under the terms of the agreement, the Customer is required to pay each invoice within thirty (30) days of its date. Invoices HS-1041 and HS-1042, issued on 3 June 2026 and 1 July 2026, remain unpaid. The total outstanding amount is GBP 42,750. [[NEEDS INPUT: check against the documents]] The Client has confirmed that the amount claimed is GBP 42,750. [[NEEDS INPUT: check against the documents]] The claim is being made against Northwind Logistics Ltd, whose address is Northwind Logistics Ltd, 2 Harbour Road, Hull HU1 1AA, United Kingdom. [[NEEDS INPUT: check against the documents]] The date of the claim is 10 October 2026. The Customer has failed to pay the invoices within the stipulated period of thirty (30) days. [[NEEDS INPUT: check against the documents]] The breach consists of the failure to pay the invoices by the due date.

## review-warehouse-lease (contract_review, 15.2 s, 2 gaps)

- agreement: **missed** — needs_input: None (expected 'Lease')
- parties: ok — sourced: 'Elmfield Estates Ltd and Alder Biotech AB' (expected 'Elmfield Estates')
- governing_law: ok — sourced: 'the law of England and Wales' (expected 'England and Wales')
- written (3 sources, 0 unsupported): The term of the agreement is ten (10) years from 1 April 2025. The Tenant may terminate this Lease on 31 March 2030 by giving not less than six months' written notice. The agreement is governed by the law of England and Wales.
- written (5 sources, 0 unsupported): The Tenant shall pay an annual rent of GBP 84,000, payable quarterly in advance. The term of this Lease is ten (10) years from 1 April 2025. The Tenant may terminate this Lease on 31 March 2030 by giving not less than six months' written notice. The Landlord's total liability under this Lease shall not exceed the rent paid in the preceding twelve months. This Lease is governed by the law of England and Wales.
- written (2 sources, 0 unsupported): The Landlord's liability under this Lease is limited to the rent paid in the preceding twelve months. There are no other exclusions of liability specified in the Lease. The governing law of the Lease is the law of England and Wales.

## nda-missing-period (nda, 4.3 s, 2 gaps)

- party_a: ok — sourced: 'Harrowgate Systems Inc.' (expected 'Harrowgate')
- party_b: ok — sourced: 'Elmfield Estates Ltd' (expected 'Elmfield')
- purpose: ok — sourced: 'assess a possible warehouse automation project' (expected 'warehouse automation')
- confidentiality_period (not in the documents): left open
- governing_law (not in the documents): left open
