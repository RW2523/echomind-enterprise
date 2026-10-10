# EchoMind Law obligations evaluation — golden

- Commit: `aaf24ef+changes`
- Rules: `obligations-s3.6e`
- Run: 20261010-095831

| Measure | Result |
|---|---|
| documents | 13 |
| expected | 56 |
| found | 56/56 |
| timing in its words | 43/43 |
| dates right | 15/15 |
| dates guessed | 0 |
| extra | 0 |
| controls clean | 4/4 |
| all proposed | True |
| by kind | {'change_of_control': '1/1', 'confidentiality': '1/1', 'consent': '1/1', 'duty': '11/11', 'expiry': '6/6', 'notice': '6/6', 'notification': '4/4', 'payment': '9/9', 'post_termination': '6/6', 'renewal': '2/2', 'survival': '1/1', 'termination': '8/8'} |
| ms per document | 17 |

## G1-demo-msa (18 ms)

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

## G2-office-lease (21 ms)

- 1.2 expiry: found, date right
- 2.1 payment: found, timing in its words
- 2.2 payment: found, timing in its words
- 3.1 termination: found, timing in its words, date right
- 4.1 duty: found, timing in its words
- 5.1 duty: found, timing in its words

## G3-mutual-nda (25 ms)

- 1.1 expiry: found, date right
- 2.1 duty: found, timing in its words
- 2.2 notification: found, timing in its words
- 2.3 survival: found, timing in its words

## G4-board-minutes-control (17 ms, control)

- (control: nothing expected)

## G5-us-saas (21 ms)

- 1.1 expiry: found, date right
- 1.2 renewal: found, timing in its words, date right
- 2.1 payment: found, timing in its words
- 2.2 payment: found, timing in its words
- 3.1 notification: found, timing in its words
- 4.1 termination: found, timing in its words
- 4.2 post_termination: found, timing in its words

## G6-spa-completion-claims (16 ms)

- 5.1 duty: found, date right
- 5.2 duty: found, timing in its words
- 5.3 notice: found, timing in its words
- 8.1 notice: found, timing in its words, date right
- 8.2 notice: found, timing in its words
- 9.1 duty: found, timing in its words, date right

## G7-internal-note-control (16 ms, control)

- (control: nothing expected)

## G8-employment-uk (13 ms)

- 1.2 expiry: found, date right
- 2.1 duty: found, timing in its words
- 3.1 termination: found, timing in its words
- 3.2 post_termination: found
- 3.3 post_termination: found, timing in its words

## G9-loan-facility-us (15 ms)

- 2.1 payment: found, date right
- 2.2 payment: found, timing in its words
- 2.3 notice: found, timing in its words
- 3.1 duty: found, timing in its words
- 3.2 notification: found, timing in its words

## G10-press-release-control (18 ms, control)

- (control: nothing expected)

## G11-distribution-uk (13 ms)

- 1.1 expiry: found, date right
- 1.1 termination: found, timing in its words
- 2.1 duty: found, timing in its words
- 2.2 payment: found, timing in its words
- 2.3 notice: found, timing in its words
- 3.1 termination: found, timing in its words
- 4.1 post_termination: found
- 4.2 post_termination: found, timing in its words

## G12-escrow-us (13 ms)

- 1.1 duty: found, date right
- 1.2 notice: found, date right
- 1.3 duty: found, timing in its words
- 2.1 payment: found, timing in its words

## G13-matter-summary-control (18 ms, control)

- (control: nothing expected)
