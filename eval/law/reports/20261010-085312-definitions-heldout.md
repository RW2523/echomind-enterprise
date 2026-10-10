# EchoMind Law definitions and cross-references evaluation — heldout

- Commit: `ed9f657`
- Checker: `definitions-s3.5b`
- Run: 20261010-085312

| Measure | Result |
|---|---|
| documents | 3 |
| planted | 16 |
| caught | 13/16 |
| fixes right | 2/2 |
| false findings | 5 |
| controls clean | 0/1 |
| by type | {'broken': '2/3', 'duplicate': '1/1', 'gap': '3/3', 'inconsistent': '2/2', 'repeat': '1/1', 'undefined': '2/4', 'unused': '2/2'} |
| ms per check | 18 |

## N1-consultancy-uk (17 ms)

- duplicate `Fee`: found
- unused `Expenses`: found
- inconsistent `deliverables`: found, fix right
- undefined `Completion Date`: found
- undefined `Background IP`: **missed** ✗
- undefined `Statement of Work`: **missed** ✗
- broken `clause 6 (Set-off)`: found
- broken `Clause 7`: **missed** ✗
- gap `Clause 3.2 is missing`: found
- gap `Clause 4 is missing`: found

Not expected:
- broken `7`: This document has no clause 7.

## N2-loan-note-markdown (22 ms)

- unused `Noteholder`: found
- undefined `Interest Payment Date`: found
- repeat `Clause 2.2 appears twice`: found
- inconsistent `maturity date`: found, fix right
- gap `Item (c) of clause 3.2 is missing`: found
- broken `paragraph 3 of Schedule 2`: found

Not expected:
- before_definition `Note`: “Note” is used before it is defined (in clause 1.2).

## N3-clean-us-consulting-control (16 ms, control)

- (control: nothing expected)

Not expected:
- undefined `Brightline Data`: “Brightline Data” reads as a defined term but is not defined in this document.
- unused `Confidential Information`: “Confidential Information” is defined but never used.
- duplicate `Confidential Information`: “Confidential Information” is defined twice, with different meanings.
