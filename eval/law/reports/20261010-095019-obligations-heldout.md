# EchoMind Law obligations evaluation — heldout

- Commit: `a4d20c9`
- Rules: `obligations-s3.6b`
- Run: 20261010-095019

| Measure | Result |
|---|---|
| documents | 3 |
| expected | 13 |
| found | 9/13 |
| timing in its words | 9/11 |
| dates right | 0/4 |
| dates guessed | 0 (corrected: first reported as 2, counting empty dates as guesses) |
| extra | 2 |
| controls clean | 1/1 |
| all proposed | True |
| by kind | {'duty': '2/3', 'expiry': '0/1', 'notice': '3/3', 'notification': '1/1', 'payment': '1/2', 'post_termination': '0/1', 'renewal': '1/1', 'termination': '1/1'} |
| ms per document | 18 |

## H1-us-saas (19 ms)

- 1.1 expiry: **missed** ✗, **date wrong** ✗ (got None)
- 1.2 renewal: found, timing in its words, **date wrong** ✗ (got None)
- 2.1 payment: found, timing in its words
- 2.2 payment: **missed** ✗, **timing missing** ✗ (got None)
- 3.1 notification: found, timing in its words
- 4.1 termination: found, timing in its words
- 4.2 post_termination: **missed** ✗, **timing missing** ✗ (got None)

Not expected:
- 2.2 notice: suspend the Service
- 4.2 duty: make Customer Data available for export

## H2-spa-completion-claims (18 ms)

- 5.1 duty: **missed** ✗, **date wrong** ✗ (got None)
- 5.2 duty: found, timing in its words
- 5.3 notice: found, timing in its words
- 8.1 notice: found, timing in its words, **date wrong** ✗ (got None)
- 8.2 notice: found, timing in its words
- 9.1 duty: found, timing in its words

## H3-internal-note-control (16 ms, control)

- (control: nothing expected)
