# EchoMind Law playbook evaluation — heldout

- Commit: `a710434+changes`
- Model: `nvidia/Qwen3-30B-A3B-FP4`
- Run: 20261010-113438

| Measure | Result |
|---|---|
| lines | 10 |
| accuracy | 9/10 |
| deviations caught | 4/4 |
| passed in error | 0 |
| by rule | 0 |
| quotes | 8 |
| seconds per check | 7.2 |

## licence-negotiated (9.5 s)

- payment: expected standard, got standard (model) — The clause matches the standard position on payment terms.
- liability_cap: expected fallback, got fallback (model) — The clause matches fallback 1 by capping liability at EUR 500,000.
- assignment: expected outside, got unclear (model) **✗** — The model named “fallback 1”, which this position does not have; a lawyer decides.
- data_protection: expected standard, got standard (model) — The clause matches the standard position on data protection.
- governing_law: expected standard, got standard (model) — The clause states the same governing law as the standard position.

## licence-aggressive (4.9 s)

- payment: expected outside, got outside (model) — Payment terms differ in frequency (quarterly vs. annual) and timing (in arrears vs. in advance).
- liability_cap: expected outside, got outside (model) — The clause specifies a different amount than the standard or fallback positions.
- governing_law: expected outside, got outside (model) — The clause specifies Germany as the governing law, which is different from the standard position of Sweden.
- assignment: expected not_found, got not_found (model) — No clause of this document deals with the topic.
- data_protection: expected not_found, got not_found (model) — No clause of this document deals with the topic.
