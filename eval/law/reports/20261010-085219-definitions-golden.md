# EchoMind Law definitions and cross-references evaluation — golden

- Commit: `e26268c+changes`
- Checker: `definitions-s3.5b`
- Run: 20261010-085219

| Measure | Result |
|---|---|
| documents | 7 |
| planted | 38 |
| caught | 37/38 |
| fixes right | 6/6 |
| false findings | 0 |
| controls clean | 2/2 |
| by type | {'broken': '8/8', 'duplicate': '2/2', 'gap': '11/11', 'inconsistent': '4/4', 'outdated': '2/2', 'repeat': '1/1', 'undefined': '6/7', 'unused': '3/3'} |
| ms per check | 23 |

## G1-services-planted (28 ms)

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

## G2-demo-msa-markdown (31 ms)

- broken `Schedule 1`: found
- broken `Schedule 2`: found
- broken `Schedule 4`: found
- undefined `Confidential Information`: found
- gap `Clauses 10 and 11 are missing`: found
- gap `Clause 13 is missing`: found
- gap `Clauses 15 to 20 are missing`: found

## G3-lease-interpretation-at-end (18 ms)

- undefined `Rent Commencement Date`: found
- inconsistent `premises`: found, fix right
- unused `Regulations`: found
- broken `clause 2.5 (Alterations)`: found
- broken `Schedule 3`: found
- repeat `Clause 6.2 appears twice`: found
- gap `Clauses 4 and 5 are missing`: found
- gap `Clauses 7 to 19 are missing`: found

## G4-clean-nda-control (22 ms, control)

- (control: nothing expected)

## G5-software-licence (21 ms)

- duplicate `Software`: found
- unused `Documentation`: found
- undefined `Licence Fee`: found
- inconsistent `licensee`: found, fix right
- broken `Article 7`: found
- outdated `clause 3 (Fees)`: found, fix right
- gap `Article 5 is missing`: found

## G6-shareholders-markdown (16 ms)

- undefined `Board`: **missed** ✗
- undefined `Directors`: found
- undefined `Shareholder Consent`: found
- broken `Schedule 3`: found
- gap `Items between (ii) and (iv)`: found

## G7-clean-employment-control (23 ms, control)

- (control: nothing expected)
