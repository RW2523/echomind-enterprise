# EchoMind Law playbook evaluation — heldout

- Commit: `c518b47+changes`
- Model: `nvidia/Qwen3-30B-A3B-FP4`
- Run: 20261010-063249

| Measure | Result |
|---|---|
| lines | 10 |
| accuracy | 9/10 |
| deviations caught | 4/4 |
| passed in error | 0 |
| by rule | 0 |
| quotes | 8 |
| seconds per check | 7.3 |

## licence-negotiated (9.6 s)

- payment: expected standard, got standard (model) — The clause matches the standard position on payment terms.
- liability_cap: expected fallback, got fallback (model) — The clause matches the fallback 1 position of capping liability at EUR 500,000.
- assignment: expected outside, got unclear (model) **✗** — The model named “fallback 1”, which this position does not have; a lawyer decides.
- data_protection: expected standard, got standard (model) — The clause matches the standard position on data protection.
- governing_law: expected standard, got standard (model) — The clause states that the agreement is governed by Swedish law, matching the standard position.

## licence-aggressive (5.1 s)

- payment: expected outside, got outside (model) — The clause specifies quarterly payments in arrears with a 60-day period, which differs from the standard's annual payments in advance.
- liability_cap: expected outside, got outside (model) — The clause specifies a different amount than the standard position.
- governing_law: expected outside, got outside (model) — The clause specifies Germany as the governing law, which is different from the standard's Sweden.
- assignment: expected not_found, got not_found (model) — No clause of this document deals with the topic.
- data_protection: expected not_found, got not_found (model) — No clause of this document deals with the topic.
