# EchoMind Law obligations evaluation — golden

- Commit: `a5f81f5+changes`
- Rules: `obligations-s3.6`
- Run: 20261010-094644

| Measure | Result |
|---|---|
| documents | 4 |
| expected | 21 |
| found | 10/21 |
| timing in its words | 9/17 |
| dates right | 1/5 |
| dates guessed | 0 |
| extra | 2 |
| controls clean | 1/1 |
| all proposed | True |
| by kind | {'change_of_control': '1/1', 'confidentiality': '1/2', 'consent': '1/1', 'duty': '0/3', 'expiry': '0/3', 'notification': '1/2', 'payment': '1/3', 'post_termination': '1/1', 'renewal': '1/1', 'termination': '3/4'} |
| ms per document | 17 |

## G1-demo-msa (15 ms)

- 2.1 expiry: **missed** ✗, **date wrong** ✗ (got None)
- 2.2 renewal: found, timing in its words, date right
- 4.2 payment: found, timing in its words
- 5.1 confidentiality: found, timing in its words
- 6.1 notification: found, timing in its words
- 6.2 post_termination: found, timing in its words
- 12.1 termination: found, timing in its words
- 12.2 termination: found, timing in its words
- 14.1 consent: found
- 14.2 change_of_control: found, timing in its words
- 14.3 termination: found, timing in its words

## G2-office-lease (17 ms)

- 1.2 expiry: **missed** ✗, **date wrong** ✗ (got None)
- 2.1 payment: **missed** ✗, **timing missing** ✗ (got None)
- 2.2 payment: **missed** ✗, **timing missing** ✗ (got None)
- 3.1 termination: **missed** ✗, **timing missing** ✗ (got None), **date wrong** ✗ (got None)
- 4.1 duty: **missed** ✗, **timing missing** ✗ (got None)
- 5.1 duty: **missed** ✗, **timing missing** ✗ (got None)

Not expected:
- 3 termination: terminate this Lease

## G3-mutual-nda (21 ms)

- 1.1 expiry: **missed** ✗, **date wrong** ✗ (got None)
- 2.1 duty: **missed** ✗, **timing missing** ✗ (got None)
- 2.2 notification: **missed** ✗, **timing missing** ✗ (got None)
- 2.3 confidentiality: **missed** ✗, **timing missing** ✗ (got None)

Not expected:
- 2 confidentiality: return or destroy all Confidential Information

## G4-board-minutes-control (16 ms, control)

- (control: nothing expected)
