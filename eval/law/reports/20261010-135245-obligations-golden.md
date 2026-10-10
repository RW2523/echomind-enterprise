# EchoMind Law obligations evaluation — golden

- Commit: `df3b8a5`
- Rules: `obligations-s3.6g`
- Model pass: `nvidia/Qwen3-30B-A3B-FP4` (obligations-model-s3.6e2)
- Run: 20261010-135245

| Measure | Result |
|---|---|
| documents | 16 |
| expected | 68 |
| found | 67/68 |
| timing in its words | 50/51 |
| dates right | 18/18 |
| dates guessed | 0 |
| extra | 4 |
| controls clean | 5/5 |
| all proposed | True |
| by kind | {'change_of_control': '1/1', 'confidentiality': '1/1', 'consent': '2/2', 'duty': '15/16', 'expiry': '8/8', 'notice': '6/6', 'notification': '5/5', 'payment': '11/11', 'post_termination': '6/6', 'renewal': '2/2', 'survival': '1/1', 'termination': '9/9'} |
| ms per document | 597 |

## G1-demo-msa (3690 ms)

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

Not expected:
- 22.1 notice: address notice to the General Counsel of the receiving party
- 4.1 payment: pay a platform subscription fee
- 4.2 payment: invoice monthly in advance

## G2-office-lease (25 ms)

- 1.2 expiry: found, date right
- 2.1 payment: found, timing in its words
- 2.2 payment: found, timing in its words
- 3.1 termination: found, timing in its words, date right
- 4.1 duty: found, timing in its words
- 5.1 duty: found, timing in its words

## G3-mutual-nda (14 ms)

- 1.1 expiry: found, date right
- 2.1 duty: found, timing in its words
- 2.2 notification: found, timing in its words
- 2.3 survival: found, timing in its words

## G4-board-minutes-control (19 ms, control)

- (control: nothing expected)

## G5-us-saas (22 ms)

- 1.1 expiry: found, date right
- 1.2 renewal: found, timing in its words, date right
- 2.1 payment: found, timing in its words
- 2.2 payment: found, timing in its words
- 3.1 notification: found, timing in its words
- 4.1 termination: found, timing in its words
- 4.2 post_termination: found, timing in its words

## G6-spa-completion-claims (15 ms)

- 5.1 duty: found, date right
- 5.2 duty: found, timing in its words
- 5.3 notice: found, timing in its words
- 8.1 notice: found, timing in its words, date right
- 8.2 notice: found, timing in its words
- 9.1 duty: found, timing in its words, date right

## G7-internal-note-control (225 ms, control)

- (control: nothing expected)

## G8-employment-uk (22 ms)

- 1.2 expiry: found, date right
- 2.1 duty: found, timing in its words
- 3.1 termination: found, timing in its words
- 3.2 post_termination: found
- 3.3 post_termination: found, timing in its words

## G9-loan-facility-us (21 ms)

- 2.1 payment: found, date right
- 2.2 payment: found, timing in its words
- 2.3 notice: found, timing in its words
- 3.1 duty: found, timing in its words
- 3.2 notification: found, timing in its words

## G10-press-release-control (225 ms, control)

- (control: nothing expected)

## G11-distribution-uk (21 ms)

- 1.1 expiry: found, date right
- 1.1 termination: found, timing in its words
- 2.1 duty: found, timing in its words
- 2.2 payment: found, timing in its words
- 2.3 notice: found, timing in its words
- 3.1 termination: found, timing in its words
- 4.1 post_termination: found
- 4.2 post_termination: found, timing in its words

## G12-escrow-us (21 ms)

- 1.1 duty: found, date right
- 1.2 notice: found, date right
- 1.3 duty: found, timing in its words
- 2.1 payment: found, timing in its words

## G13-matter-summary-control (12 ms, control)

- (control: nothing expected)

## G14-lease-uk (1455 ms)

- 1.1 expiry: found, date right
- 2.1 payment: found, timing in its words
- 2.2 duty: found, date right
- 3.1 duty: found, timing in its words
- 3.2 duty: **missed** ✗, **timing missing** ✗ (got None)
- 4.1 consent: found

Not expected:
- 3.2 notice: carry out any repairs specified in a notice from the Landlord

## G15-supply-us (3548 ms)

- 1.1 expiry: found, date right
- 2.1 duty: found, timing in its words
- 2.2 duty: found, timing in its words
- 3.1 notification: found, timing in its words
- 4.1 termination: found, timing in its words
- 4.2 payment: found, timing in its words

## G16-engagement-letter-control (221 ms, control)

- (control: nothing expected)
