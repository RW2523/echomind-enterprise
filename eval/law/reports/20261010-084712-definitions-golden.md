# EchoMind Law definitions and cross-references evaluation — golden

- Commit: `a4bdc49+changes`
- Checker: `definitions-s3.5`
- Run: 20261010-084712

| Measure | Result |
|---|---|
| documents | 4 |
| planted | 26 |
| caught | 26/26 |
| fixes right | 4/4 |
| false findings | 7 |
| controls clean | 0/1 |
| by type | {'broken': '6/6', 'duplicate': '1/1', 'gap': '9/9', 'inconsistent': '3/3', 'outdated': '1/1', 'repeat': '1/1', 'undefined': '3/3', 'unused': '2/2'} |
| ms per check | 23 |

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

## G2-demo-msa-markdown (27 ms)

- broken `Schedule 1`: found
- broken `Schedule 2`: found
- broken `Schedule 4`: found
- undefined `Confidential Information`: found
- gap `Clauses 10 and 11 are missing`: found
- gap `Clause 13 is missing`: found
- gap `Clauses 15 to 20 are missing`: found

Not expected:
- inconsistent `service`: “service” is used where the defined term is “Service”.
- broken `Schedule 1`: This document has no Schedule 1.

## G3-lease-interpretation-at-end (22 ms)

- undefined `Rent Commencement Date`: found
- inconsistent `premises`: found, fix right
- unused `Regulations`: found
- broken `clause 2.5 (Alterations)`: found
- broken `Schedule 3`: found
- repeat `Clause 6.2 appears twice`: found
- gap `Clauses 4 and 5 are missing`: found
- gap `Clauses 7 to 19 are missing`: found

## G4-clean-nda-control (18 ms, control)

- (control: nothing expected)

Not expected:
- unused `Corran`: “Corran” is defined but never used.
- unused `Bellhaven`: “Bellhaven” is defined but never used.
- undefined `Parties`: “Parties” reads as a defined term but is not defined in this document.
- undefined `Party's Confidential Information`: “Party's Confidential Information” reads as a defined term but is not defined in this document.
- gap `2.2(i)`: Items (d) to (h) of clause 2.2 are missing: the items go from (c) to (i).
