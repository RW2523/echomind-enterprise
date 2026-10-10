# EchoMind Law definitions and cross-references evaluation — golden

- Commit: `ed9f657+changes`
- Checker: `definitions-s3.5c`
- Run: 20261010-085549

| Measure | Result |
|---|---|
| documents | 10 |
| planted | 55 |
| caught | 52/55 |
| fixes right | 8/8 |
| false findings | 0 |
| controls clean | 2/2 |
| by type | {'broken': '11/11', 'duplicate': '3/3', 'gap': '14/14', 'inconsistent': '6/6', 'outdated': '2/2', 'repeat': '2/2', 'undefined': '8/11', 'unused': '6/6'} |
| ms per check | 18 |

## G1-services-planted (20 ms)

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

## G2-demo-msa-markdown (28 ms)

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

## G4-clean-nda-control (12 ms, control)

- (control: nothing expected)

## G5-software-licence (13 ms)

- duplicate `Software`: found
- unused `Documentation`: found
- undefined `Licence Fee`: found
- inconsistent `licensee`: found, fix right
- broken `Article 7`: found
- outdated `clause 3 (Fees)`: found, fix right
- gap `Article 5 is missing`: found

## G6-shareholders-markdown (14 ms)

- undefined `Board`: **missed** ✗
- undefined `Directors`: found
- undefined `Shareholder Consent`: found
- broken `Schedule 3`: found
- gap `Items between (ii) and (iv)`: found

## G7-clean-employment-control (15 ms, control)

- (control: nothing expected)

## G8-consultancy-uk (14 ms)

- duplicate `Fee`: found
- unused `Expenses`: found
- inconsistent `deliverables`: found, fix right
- undefined `Completion Date`: found
- undefined `Background IP`: **missed** ✗
- undefined `Statement of Work`: **missed** ✗
- broken `clause 6 (Set-off)`: found
- broken `Clause 7`: found
- gap `Clause 3.2 is missing`: found
- gap `Clause 4 is missing`: found

## G9-loan-note-markdown (22 ms)

- unused `Noteholder`: found
- undefined `Interest Payment Date`: found
- repeat `Clause 2.2 appears twice`: found
- inconsistent `maturity date`: found, fix right
- gap `Item (c) of clause 3.2 is missing`: found
- broken `paragraph 3 of Schedule 2`: found

## G10-clean-us-consulting-control (19 ms)

- unused `Confidential Information`: found
