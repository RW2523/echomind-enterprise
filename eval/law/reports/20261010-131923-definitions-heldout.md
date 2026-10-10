# EchoMind Law definitions and cross-references evaluation — heldout

- Commit: `689331f`
- Checker: `definitions-s3.5c`
- Run: 20261010-131923

| Measure | Result |
|---|---|
| documents | 3 |
| planted | 11 |
| caught | 9/11 |
| fixes right | 2/2 |
| false findings | 2 |
| controls clean | 1/1 |
| by type | {'broken': '1/1', 'gap': '4/4', 'inconsistent': '1/1', 'outdated': '1/1', 'repeat': '1/1', 'undefined': '0/2', 'unused': '1/1'} |
| ms per check | 19 |

## R1-dpa (26 ms)

- inconsistent `personal data`: found, fix right
- unused `Supervisory Authority`: found
- undefined `Sub-processor`: **missed** ✗
- outdated `clause 6 (Audits)`: found, fix right
- broken `clause 14`: found
- gap `Clauses 3 to 5 are missing`: found
- gap `Clause 8 is missing`: found

Not expected:
- inconsistent `personal data`: “personal data” is used where the defined term is “Personal Data”.

## R2-sha (15 ms)

- undefined `Investor Consent`: **missed** ✗
- repeat `Clause 4.2 appears twice`: found
- gap `Item (c) of clause 4.1 is missing`: found
- gap `Clauses 2 and 3 are missing`: found

Not expected:
- inconsistent `shareholders`: “shareholders” is used where the defined term is “Shareholders”.

## R3-clean-control (16 ms, control)

- (control: nothing expected)
