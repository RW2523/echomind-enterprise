# EchoMind Law drafting evaluation — heldout

- Commit: `9a3894f`
- Model: `nvidia/Qwen3-30B-A3B-FP4`
- Run: 20261010-132819

| Measure | Result |
|---|---|
| scenarios | 3 |
| field accuracy | 8/11 |
| absent left open | 1/1 |
| wrong values | 0 |
| invented figures | 0 |
| written passages | 2 |
| unsupported sentences | 5 |
| seconds per draft | 7.6 |

## engagement-corran (engagement_letter, 6.8 s, 6 gaps)

- client_address: **missed** — needs_input: None (expected '9 Mill Lane')
- client_contact: **missed** — needs_input: None (expected 'Imogen Hale')
- fee_arrangement: ok — sourced: 'GBP 25,000 plus VAT' (expected '25,000')
- written (2 sources, 1 unsupported): We have been instructed to act for Corran Foods Ltd in the dispute with its packaging supplier, with the scope of the engagement limited to the pre-action stage. The engagement is subject to a capped fee of GBP 25,000 plus VAT. [[NEEDS INPUT: check against the documents]] The matter is assigned the matter number EVAL-D1 and is governed by the terms set out in the engagement letter dated 10 October 2026.

## demand-brightwater (demand_letter, 10.8 s, 7 gaps)

- recipient: **missed** — needs_input: None (expected 'Brightwater Retail')
- recipient_address: ok — sourced: 'Brightwater Retail Ltd, 40 Canal Street, Manchester M1 3HE' (expected '40 Canal Street')
- agreement: ok — sourced: 'SOFTWARE LICENCE AGREEMENT' (expected 'Software Licence Agreement')
- amount_claimed: ok — sourced: 'SEK 310,000' (expected '310,000')
- written (4 sources, 4 unsupported): [[NEEDS INPUT: check against the documents]] Brightwater Retail Ltd is the Licensee under a Software Licence Agreement with Northbridge Analytics AB. Notices to the Licensee shall be sent to Brightwater Retail Ltd, 40 Canal Street, Manchester M1 3HE. The Licensee is required to pay the annual licence fee within thirty (30) days of invoice. [[NEEDS INPUT: check against the documents]] An invoice for the 2026 annual licence fee, numbered NA-2207, remains unpaid. The amount outstanding is SEK 310,000. The amount outstanding is SEK 310,000. [[NEEDS INPUT: check against the documents]] The client has confirmed that the amount claimed is SEK 310,000. [[NEEDS INPUT: check against the documents]] The claim was made on 10 October 2026.

## nda-no-law (nda, 5.3 s, 1 gaps)

- party_a: ok — sourced: 'Corran Foods Ltd' (expected 'Corran')
- party_b: ok — sourced: 'Alder Biotech AB' (expected 'Alder')
- purpose: ok — sourced: 'to evaluate a joint development of plant-based protein ingredients' (expected 'plant-based protein')
- confidentiality_period: ok — sourced: 'two (2) years' (expected 'two')
- governing_law (not in the documents): left open
