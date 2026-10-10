# EchoMind Law definitions and cross-references evaluation — heldout

- Commit: `5dc27a4`
- Checker: `definitions-s3.5c`
- Run: 20261010-085712

| Measure | Result |
|---|---|
| documents | 3 |
| planted | 12 |
| caught | 12/12 |
| fixes right | 3/3 |
| false findings | 1 |
| controls clean | 1/1 |
| by type | {'broken': '2/2', 'duplicate': '1/1', 'gap': '3/3', 'inconsistent': '2/2', 'outdated': '1/1', 'repeat': '1/1', 'undefined': '1/1', 'unused': '1/1'} |
| ms per check | 23 |

## P1-asset-purchase (26 ms)

- duplicate `Assets`: found
- unused `Excluded Liabilities`: found
- inconsistent `assets`: found, fix right
- undefined `Completion Accounts`: found
- broken `Schedule 4`: found
- repeat `Clause 5 appears twice`: found
- gap `Clause 3 is missing`: found
- gap `Clause 6.3 is missing`: found

Not expected:
- repeat `5.1`: Clause 5.1 appears twice.

## P2-lease-markdown (17 ms)

- inconsistent `term`: found, fix right
- outdated `Clause 7 (Insurance)`: found, fix right
- broken `clause 9.4`: found
- gap `Clauses 3 to 6 are missing`: found

## P3-clean-nda-control (25 ms, control)

- (control: nothing expected)
