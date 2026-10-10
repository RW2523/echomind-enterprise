# EchoMind Law definitions and cross-references evaluation — heldout

- Commit: `e26268c`
- Checker: `definitions-s3.5`
- Run: 20261010-084909

| Measure | Result |
|---|---|
| documents | 3 |
| planted | 12 |
| caught | 8/12 |
| fixes right | 2/2 |
| false findings | 1 |
| controls clean | 1/1 |
| by type | {'broken': '2/2', 'duplicate': '1/1', 'gap': '1/2', 'inconsistent': '1/1', 'outdated': '1/1', 'undefined': '1/4', 'unused': '1/1'} |
| ms per check | 18 |

## H1-software-licence (17 ms)

- duplicate `Software`: found
- unused `Documentation`: found
- undefined `Licence Fee`: found
- inconsistent `licensee`: found, fix right
- broken `Article 7`: found
- outdated `clause 3 (Fees)`: found, fix right
- gap `Article 5 is missing`: **missed** ✗

Not expected:
- gap `6`: Clause 5 is missing: the numbering goes from 4 to 6.

## H2-shareholders-markdown (15 ms)

- undefined `Board`: **missed** ✗
- undefined `Directors`: **missed** ✗
- undefined `Shareholder Consent`: **missed** ✗
- broken `Schedule 3`: found
- gap `Items between (ii) and (iv)`: found

## H3-clean-employment-control (22 ms, control)

- (control: nothing expected)
