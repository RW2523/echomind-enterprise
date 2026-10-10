# EchoMind Law obligations evaluation — heldout

- Commit: `e16681f`
- Rules: `obligations-s3.6e`
- Run: 20261010-100341

| Measure | Result |
|---|---|
| documents | 3 |
| expected | 12 |
| found | 10/12 |
| timing in its words | 7/8 |
| dates right | 2/3 |
| dates guessed | 0 |
| extra | 1 |
| controls clean | 1/1 |
| all proposed | True |
| by kind | {'consent': '1/1', 'duty': '4/5', 'expiry': '1/2', 'notification': '1/1', 'payment': '2/2', 'termination': '1/1'} |
| ms per document | 1738 |

## L1-lease-uk (1381 ms)

- 1.1 expiry: **missed** ✗, **date wrong** ✗ (got None)
- 2.1 payment: found, timing in its words
- 2.2 duty: found, date right
- 3.1 duty: found, timing in its words
- 3.2 duty: **missed** ✗, **timing missing** ✗ (got None)
- 4.1 consent: found

Not expected:
- 3.2 notice: carry out any repairs specified in a notice from the Landlord

## L2-supply-us (3580 ms)

- 1.1 expiry: found, date right
- 2.1 duty: found, timing in its words
- 2.2 duty: found, timing in its words
- 3.1 notification: found, timing in its words
- 4.1 termination: found, timing in its words
- 4.2 payment: found, timing in its words

## L3-engagement-letter-control (253 ms, control)

- (control: nothing expected)
