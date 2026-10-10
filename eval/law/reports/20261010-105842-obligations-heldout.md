# EchoMind Law obligations evaluation — heldout

- Commit: `a710434`
- Rules: `obligations-s3.6f`
- Run: 20261010-105842

| Measure | Result |
|---|---|
| documents | 3 |
| expected | 6 |
| found | 4/6 |
| timing in its words | 3/4 |
| dates right | 1/2 |
| dates guessed | 0 |
| extra | 2 |
| controls clean | 1/1 |
| all proposed | True |
| by kind | {'duty': '1/1', 'expiry': '1/2', 'notification': '1/1', 'payment': '1/1', 'termination': '0/1'} |
| ms per document | 100 |

## M1-consultancy-terms (19 ms)

- 1.1 expiry: found, date right
- 1.2 termination: **missed** ✗, **timing missing** ✗ (got None)
- 2.1 payment: found, timing in its words
- 3.1 duty: found, timing in its words

Not expected:
- 1.2 notice: end this Agreement

## M2-nda-us (18 ms)

- 1.1 expiry: **missed** ✗, **date wrong** ✗ (got None)
- 2.1 notification: found, timing in its words

Not expected:
- 1.1 duty: expire after the Effective Date

## M3-meeting-agenda-control (263 ms, control)

- (control: nothing expected)
