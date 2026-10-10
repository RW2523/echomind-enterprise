# EchoMind Law obligations evaluation — heldout

- Commit: `aaf24ef`
- Rules: `obligations-s3.6d`
- Run: 20261010-095547

| Measure | Result |
|---|---|
| documents | 3 |
| expected | 12 |
| found | 5/12 |
| timing in its words | 4/8 |
| dates right | 0/3 |
| dates guessed | 0 |
| extra | 1 |
| controls clean | 1/1 |
| all proposed | True |
| by kind | {'duty': '1/3', 'expiry': '0/1', 'notice': '1/2', 'payment': '1/2', 'post_termination': '1/2', 'termination': '1/2'} |
| ms per document | 13 |

## K1-distribution-uk (13 ms)

- 1.1 expiry: **missed** ✗, **date wrong** ✗ (got None)
- 1.1 termination: **missed** ✗, **timing missing** ✗ (got None)
- 2.1 duty: **missed** ✗, **timing missing** ✗ (got None)
- 2.2 payment: found, timing in its words
- 2.3 notice: found, timing in its words
- 3.1 termination: found, timing in its words
- 4.1 post_termination: found
- 4.2 post_termination: **missed** ✗, **timing missing** ✗ (got None)

Not expected:
- 1.1 notice: remain in force for an initial period of five (5) years from the date of this Agreement and thereafter until terminated

## K2-escrow-us (11 ms)

- 1.1 duty: **missed** ✗, **date wrong** ✗ (got None)
- 1.2 notice: **missed** ✗, **date wrong** ✗ (got None)
- 1.3 duty: found, timing in its words
- 2.1 payment: **missed** ✗, **timing missing** ✗ (got None)

## K3-matter-summary-control (16 ms, control)

- (control: nothing expected)
