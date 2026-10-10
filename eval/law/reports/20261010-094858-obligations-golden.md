# EchoMind Law obligations evaluation — golden

- Commit: `a5f81f5+changes`
- Rules: `obligations-s3.6b`
- Run: 20261010-094858

| Measure | Result |
|---|---|
| documents | 4 |
| expected | 21 |
| found | 21/21 |
| timing in its words | 17/17 |
| dates right | 5/5 |
| dates guessed | 0 |
| extra | 0 |
| controls clean | 1/1 |
| all proposed | True |
| by kind | {'change_of_control': '1/1', 'confidentiality': '1/1', 'consent': '1/1', 'duty': '3/3', 'expiry': '3/3', 'notification': '2/2', 'payment': '3/3', 'post_termination': '1/1', 'renewal': '1/1', 'survival': '1/1', 'termination': '4/4'} |
| ms per document | 18 |

## G1-demo-msa (25 ms)

- 2.1 expiry: found, date right
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

## G2-office-lease (15 ms)

- 1.2 expiry: found, date right
- 2.1 payment: found, timing in its words
- 2.2 payment: found, timing in its words
- 3.1 termination: found, timing in its words, date right
- 4.1 duty: found, timing in its words
- 5.1 duty: found, timing in its words

## G3-mutual-nda (18 ms)

- 1.1 expiry: found, date right
- 2.1 duty: found, timing in its words
- 2.2 notification: found, timing in its words
- 2.3 survival: found, timing in its words

## G4-board-minutes-control (16 ms, control)

- (control: nothing expected)
