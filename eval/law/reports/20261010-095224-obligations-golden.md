# EchoMind Law obligations evaluation — golden

- Commit: `a4d20c9+changes`
- Rules: `obligations-s3.6c`
- Run: 20261010-095224

| Measure | Result |
|---|---|
| documents | 7 |
| expected | 34 |
| found | 34/34 |
| timing in its words | 28/28 |
| dates right | 10/10 |
| dates guessed | 0 |
| extra | 0 |
| controls clean | 2/2 |
| all proposed | True |
| by kind | {'change_of_control': '1/1', 'confidentiality': '1/1', 'consent': '1/1', 'duty': '6/6', 'expiry': '4/4', 'notice': '3/3', 'notification': '3/3', 'payment': '5/5', 'post_termination': '2/2', 'renewal': '2/2', 'survival': '1/1', 'termination': '5/5'} |
| ms per document | 18 |

## G1-demo-msa (24 ms)

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

## G2-office-lease (19 ms)

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

## G4-board-minutes-control (12 ms, control)

- (control: nothing expected)

## G5-us-saas (19 ms)

- 1.1 expiry: found, date right
- 1.2 renewal: found, timing in its words, date right
- 2.1 payment: found, timing in its words
- 2.2 payment: found, timing in its words
- 3.1 notification: found, timing in its words
- 4.1 termination: found, timing in its words
- 4.2 post_termination: found, timing in its words

## G6-spa-completion-claims (21 ms)

- 5.1 duty: found, date right
- 5.2 duty: found, timing in its words
- 5.3 notice: found, timing in its words
- 8.1 notice: found, timing in its words, date right
- 8.2 notice: found, timing in its words
- 9.1 duty: found, timing in its words, date right

## G7-internal-note-control (11 ms, control)

- (control: nothing expected)
