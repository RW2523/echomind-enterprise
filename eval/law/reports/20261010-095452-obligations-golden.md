# EchoMind Law obligations evaluation — golden

- Commit: `aac96ba+changes`
- Rules: `obligations-s3.6d`
- Run: 20261010-095452

| Measure | Result |
|---|---|
| documents | 10 |
| expected | 44 |
| found | 44/44 |
| timing in its words | 35/35 |
| dates right | 12/12 |
| dates guessed | 0 |
| extra | 0 |
| controls clean | 3/3 |
| all proposed | True |
| by kind | {'change_of_control': '1/1', 'confidentiality': '1/1', 'consent': '1/1', 'duty': '8/8', 'expiry': '5/5', 'notice': '4/4', 'notification': '4/4', 'payment': '7/7', 'post_termination': '4/4', 'renewal': '2/2', 'survival': '1/1', 'termination': '6/6'} |
| ms per document | 16 |

## G1-demo-msa (17 ms)

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

## G2-office-lease (14 ms)

- 1.2 expiry: found, date right
- 2.1 payment: found, timing in its words
- 2.2 payment: found, timing in its words
- 3.1 termination: found, timing in its words, date right
- 4.1 duty: found, timing in its words
- 5.1 duty: found, timing in its words

## G3-mutual-nda (17 ms)

- 1.1 expiry: found, date right
- 2.1 duty: found, timing in its words
- 2.2 notification: found, timing in its words
- 2.3 survival: found, timing in its words

## G4-board-minutes-control (16 ms, control)

- (control: nothing expected)

## G5-us-saas (21 ms)

- 1.1 expiry: found, date right
- 1.2 renewal: found, timing in its words, date right
- 2.1 payment: found, timing in its words
- 2.2 payment: found, timing in its words
- 3.1 notification: found, timing in its words
- 4.1 termination: found, timing in its words
- 4.2 post_termination: found, timing in its words

## G6-spa-completion-claims (22 ms)

- 5.1 duty: found, date right
- 5.2 duty: found, timing in its words
- 5.3 notice: found, timing in its words
- 8.1 notice: found, timing in its words, date right
- 8.2 notice: found, timing in its words
- 9.1 duty: found, timing in its words, date right

## G7-internal-note-control (11 ms, control)

- (control: nothing expected)

## G8-employment-uk (19 ms)

- 1.2 expiry: found, date right
- 2.1 duty: found, timing in its words
- 3.1 termination: found, timing in its words
- 3.2 post_termination: found
- 3.3 post_termination: found, timing in its words

## G9-loan-facility-us (13 ms)

- 2.1 payment: found, date right
- 2.2 payment: found, timing in its words
- 2.3 notice: found, timing in its words
- 3.1 duty: found, timing in its words
- 3.2 notification: found, timing in its words

## G10-press-release-control (11 ms, control)

- (control: nothing expected)
