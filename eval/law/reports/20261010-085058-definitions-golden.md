# EchoMind Law definitions and cross-references evaluation — golden

- Commit: `e26268c+changes`
- Checker: `definitions-s3.5b`
- Run: 20261010-085058

| Measure | Result |
|---|---|
| documents | 4 |
| planted | 26 |
| caught | 26/26 |
| fixes right | 4/4 |
| false findings | 2 |
| controls clean | 1/1 |
| by type | {'broken': '6/6', 'duplicate': '1/1', 'gap': '9/9', 'inconsistent': '3/3', 'outdated': '1/1', 'repeat': '1/1', 'undefined': '3/3', 'unused': '2/2'} |
| ms per check | 18 |

## G1-services-planted (24 ms)

- duplicate `Fees`: found
- unused `Territory`: found
- undefined `Commencement Date`: found
- inconsistent `fees`: found, fix right
- inconsistent `confidential information`: found, fix right
- outdated `clause 4 (Liability)`: found, fix right
- broken `clause 12`: found
- gap `Clause 3 is missing`: found
- gap `Clauses 5 to 8 are missing`: found
- gap `Clause 4.2 is missing`: found
- gap `Item (b) of clause 9.2 is missing`: found

## G2-demo-msa-markdown (18 ms)

- broken `Schedule 1`: found
- broken `Schedule 2`: found
- broken `Schedule 4`: found
- undefined `Confidential Information`: found
- gap `Clauses 10 and 11 are missing`: found
- gap `Clause 13 is missing`: found
- gap `Clauses 15 to 20 are missing`: found

Not expected:
- undefined `Client under Clause`: “Client under Clause” reads as a defined term but is not defined in this document.

## G3-lease-interpretation-at-end (14 ms)

- undefined `Rent Commencement Date`: found
- inconsistent `premises`: found, fix right
- unused `Regulations`: found
- broken `clause 2.5 (Alterations)`: found
- broken `Schedule 3`: found
- repeat `Clause 6.2 appears twice`: found
- gap `Clauses 4 and 5 are missing`: found
- gap `Clauses 7 to 19 are missing`: found

Not expected:
- undefined `Building Regulations`: “Building Regulations” reads as a defined term but is not defined in this document.

## G4-clean-nda-control (15 ms, control)

- (control: nothing expected)
