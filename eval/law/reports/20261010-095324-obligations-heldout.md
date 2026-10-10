# EchoMind Law obligations evaluation — heldout

- Commit: `aac96ba`
- Rules: `obligations-s3.6c`
- Run: 20261010-095324

| Measure | Result |
|---|---|
| documents | 3 |
| expected | 10 |
| found | 5/10 |
| timing in its words | 4/7 |
| dates right | 0/2 |
| dates guessed | 0 |
| extra | 0 |
| controls clean | 1/1 |
| all proposed | True |
| by kind | {'duty': '1/2', 'expiry': '0/1', 'notice': '1/1', 'notification': '1/1', 'payment': '0/2', 'post_termination': '1/2', 'termination': '1/1'} |
| ms per document | 12 |

## I1-employment-uk (14 ms)

- 1.2 expiry: **missed** ✗, **date wrong** ✗ (got None)
- 2.1 duty: **missed** ✗, **timing missing** ✗ (got None)
- 3.1 termination: found, timing in its words
- 3.2 post_termination: **missed** ✗
- 3.3 post_termination: found, timing in its words

## I2-loan-facility-us (12 ms)

- 2.1 payment: **missed** ✗, **date wrong** ✗ (got None)
- 2.2 payment: **missed** ✗, **timing missing** ✗ (got None)
- 2.3 notice: found, timing in its words
- 3.1 duty: found, **timing missing** ✗ (got 'twenty (120) days after the end of each fiscal year')
- 3.2 notification: found, timing in its words

## I3-press-release-control (11 ms, control)

- (control: nothing expected)
