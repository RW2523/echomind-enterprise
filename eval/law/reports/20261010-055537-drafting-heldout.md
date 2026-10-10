# EchoMind Law drafting evaluation — heldout

- Commit: `97bc690+changes`
- Model: `nvidia/Qwen3-30B-A3B-FP4`
- Run: 20261010-055537

| Measure | Result |
|---|---|
| scenarios | 3 |
| field accuracy | 9/10 |
| absent left open | 2/2 |
| wrong values | 0 |
| invented figures | 0 |
| written passages | 4 |
| unsupported sentences | 3 |
| seconds per draft | 11.0 |

## demand-northwind (demand_letter, 14.0 s, 2 gaps)

- recipient: ok — sourced: 'Northwind Logistics Ltd' (expected 'Northwind Logistics')
- recipient_address: ok — sourced: 'Northwind Logistics Ltd, 2 Harbour Road, Hull HU1 1AA, United Kingdom' (expected '2 Harbour Road')
- agreement: ok — sourced: 'the Supply Agreement' (expected 'Supply Agreement')
- amount_claimed: ok — sourced: 'GBP 42,750' (expected '42,750')
- written (4 sources, 0 unsupported): The Supplier entered into a Supply Agreement with Northwind Logistics Ltd, which provides that notices to the Customer shall be sent to Northwind Logistics Ltd, 2 Harbour Road, Hull HU1 1AA, United Kingdom.  
Under the terms of the Supply Agreement, the Customer undertook to pay each invoice within thirty (30) days of its date.  
Invoices HS-1041 and HS-1042, issued on 3 June 2026 and 1 July 2026, remain unpaid.  
The total amount outstanding is GBP 42,750.  
The Supplier has not received payment for the invoices, and the Customer has therefore failed to comply with the payment terms of the Supply Agreement.

## review-warehouse-lease (contract_review, 14.7 s, 5 gaps)

- agreement: **missed** — needs_input: None (expected 'Lease')
- parties: ok — sourced: 'Elmfield Estates Ltd and Alder Biotech AB' (expected 'Elmfield Estates')
- governing_law: ok — sourced: 'the law of England and Wales' (expected 'England and Wales')
- written (2 sources, 0 unsupported): The term of the agreement is ten (10) years, commencing on [[NEEDS INPUT: confirm “1 April 2025” — not in the sources]]. The Tenant may terminate the Lease on 31 March 2030 by providing not less than six months' written notice.
- written (0 sources, 0 unsupported): The Tenant shall pay rent of GBP 84,000 per year, payable quarterly in advance [4.1: "The rent is GBP 84,000 per year, payable quarterly in advance"]. The term of this Lease is ten (10) years from [[NEEDS INPUT: confirm “1 April 2025” — not in the sources]] [3.1: "The term of this Lease is ten (10) years from [[NEEDS INPUT: confirm “1 April 2025” — not in the sources]]"]. The Tenant may terminate this Lease on 31 March 2030 by giving not less than six months' written notice [3.4: "The Tenant may terminate this Lease on 31 March 2030 by giving not less than six months' written notice"]. The Landlord's total liability under this Lease shall not exceed the rent paid in the preceding twelve months [9.2: "The Landlord's total liability under this Lease shall not exceed the rent paid in the preceding twelve months"]. This Lease is governed by the law of England and Wales [12.1: "This Lease is governed by the law of England and Wales"].
- written (0 sources, 3 unsupported): The Landlord's liability under this Lease is limited to the rent paid in the preceding twelve months. There is no indication in the Lease of any exclusion of liability beyond this limitation. The Lease is governed by the law of England and Wales.

## nda-missing-period (nda, 4.2 s, 2 gaps)

- party_a: ok — sourced: 'Harrowgate Systems Inc.' (expected 'Harrowgate')
- party_b: ok — sourced: 'Elmfield Estates Ltd' (expected 'Elmfield')
- purpose: ok — sourced: 'assess a possible warehouse automation project' (expected 'warehouse automation')
- confidentiality_period (not in the documents): left open
- governing_law (not in the documents): left open
