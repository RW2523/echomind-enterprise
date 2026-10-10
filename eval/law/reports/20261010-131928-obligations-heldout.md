# EchoMind Law obligations evaluation — heldout

- Commit: `689331f`
- Rules: `obligations-s3.6g`
- Run: 20261010-131928

| Measure | Result |
|---|---|
| documents | 3 |
| expected | 7 |
| found | 7/7 |
| timing in its words | 4/4 |
| dates right | 2/2 |
| dates guessed | 0 |
| extra | 0 |
| controls clean | 1/1 |
| all proposed | True |
| by kind | {'duty': '1/1', 'expiry': '1/1', 'notice': '1/1', 'payment': '3/3', 'termination': '1/1'} |
| ms per document | 372 |

## P-reseller (1084 ms)

- 1.1 expiry: found, date right
- 2.1 duty: found, timing in its words
- 2.2 payment: found, timing in its words
- 3.1 notice: found, timing in its words
- 4.1 termination: found

## P-guarantee (14 ms)

- 1.1 payment: found, date right
- 2.1 payment: found, timing in its words

## P-minutes-control (17 ms, control)

- (control: nothing expected)
