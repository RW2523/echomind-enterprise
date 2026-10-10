# EchoMind Law drafting evaluation — heldout

- Commit: `3446d1e`
- Model: `nvidia/Qwen3-30B-A3B-FP4`
- Run: 20261010-133601

| Measure | Result |
|---|---|
| scenarios | 3 |
| field accuracy | 7/11 |
| absent left open | 1/1 |
| wrong values | 1 |
| invented figures | 0 |
| written passages | 2 |
| unsupported sentences | 1 |
| seconds per draft | 7.7 |

## engagement-corran (engagement_letter, 6.8 s, 6 gaps)

- client_address: **missed** — needs_input: None (expected '9 Mill Lane')
- client_contact: **missed** — needs_input: None (expected 'Imogen Hale')
- fee_arrangement: ok — sourced: 'GBP 25,000 plus VAT' (expected '25,000')
- written (2 sources, 1 unsupported): The client has retained the firm to act in a dispute with its packaging supplier, as confirmed in the engagement letter. The scope of the legal services includes representation during the pre-action stage of the dispute, with a capped fee of GBP 25,000 plus VAT. [[NEEDS INPUT: check against the documents]] The matter is assigned the matter number EVAL-D1, and the fee arrangement is GBP 25,000 plus VAT.

## demand-brightwater (demand_letter, 11.6 s, 3 gaps)

- recipient: **missed** — needs_input: None (expected 'Brightwater Retail')
- recipient_address: ok — sourced: 'Brightwater Retail Ltd, 40 Canal Street, Manchester M1 3HE' (expected '40 Canal Street')
- agreement: **missed** — sourced: 'Northbridge Analytics AB' (expected 'Software Licence Agreement')
- amount_claimed: ok — sourced: 'SEK 310,000' (expected '310,000')
- written (5 sources, 0 unsupported): Brightwater Retail Ltd is the Licensee under a Software Licence Agreement dated 10 October 2026 with Northbridge Analytics AB. Notices to the Licensee are to be sent to the address 40 Canal Street, Manchester M1 3HE. Under the terms of the agreement, the Licensee is required to pay the annual licence fee within thirty (30) days of invoice. An invoice dated 10 October 2026 for the 2026 annual licence fee remains unpaid. The amount outstanding is SEK 310,000. The breach complained of is the failure by Brightwater Retail Ltd to pay the annual licence fee as required under the agreement.

## nda-no-law (nda, 4.7 s, 1 gaps)

- party_a: ok — sourced: 'Corran Foods Ltd' (expected 'Corran')
- party_b: ok — sourced: 'Alder Biotech AB' (expected 'Alder')
- purpose: ok — sourced: 'to evaluate a joint development of plant-based protein ingredients' (expected 'plant-based protein')
- confidentiality_period: ok — sourced: 'two (2) years' (expected 'two')
- governing_law (not in the documents): left open
