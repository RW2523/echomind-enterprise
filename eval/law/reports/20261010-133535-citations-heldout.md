# EchoMind Law citation checker evaluation (T07) — heldout

- Commit: `3446d1e`
- Checker: `citations-s3.4g`
- Run: 20261010-133535

| Measure | Result |
|---|---|
| items | 24 |
| correct | 24/24 |
| references | 4/4 |
| missing caught | 2/2 |
| false missing | 0 |
| quotations | 4/4 |
| altered caught | 1/1 |
| false alarms | 0 |
| authorities found | 16/16 |
| malformed caught | 4/4 |
| wrongly flagged | 0 |
| all unverified | True |
| external lookups | 0 |
| extra items | 0 |
| ms per check | 30 |

## S1-executive-exit-uk (33 ms)

- quote `For twelve months after the Termination Date the Executive shall not solicit any Client`: expected altered, clause 12.2 of Service Agreement; got altered, clause 12.2 of Service Agreement
- quote `for up to three months`: expected accurate, clause 18.1 of Service Agreement; got accurate, clause 18.1 of Service Agreement
- reference `Clause 22.5 of the Service Agreement`: expected missing; got missing — The Service Agreement has no clause 22.5.
- authority `section 98 of the Employment Rights Act 1996`: expected well-formed; got unverified, well-formed
- authority `Contracts (Rights of Third Parties) Act 1999`: expected well-formed; got unverified, well-formed
- authority `[2019] UKSC 32`: expected well-formed; got unverified, well-formed
- authority `[1916] 1 AC 688`: expected well-formed; got unverified, well-formed
- authority `[2003] UKSC 14`: expected malformed; got unverified, malformed: UKSC gave no judgments before 2009

## S2-distribution-memo-eu (19 ms)

- authority `Regulation (EU) 2022/720`: expected well-formed; got unverified, well-formed
- authority `Directive 2014/104/EU`: expected well-formed; got unverified, well-formed
- authority `Case C-230/16`: expected well-formed; got unverified, well-formed
- authority `Regulation (EU) 2031/45`: expected malformed; got unverified, malformed: the year 2031 is in the future
- authority `Case T-112/29`: expected malformed; got unverified, malformed: the year 29 is in the future
- reference `paragraph 1.1`: expected resolved, clause 1.1 of self; got resolved, clause 1.1 of this document
- reference `Paragraph 3.4`: expected missing; got missing — This document has no paragraph 3.4.

## S3-msa-memo-us (37 ms)

- quote `within forty-five (45) days of receipt`: expected accurate, clause 5.2 of Master Services Agreement; got accurate, clause 5.2 of Master Services Agreement
- quote `In no event shall either party's aggregate liability exceed the fees paid in the twelve (12) months preceding the claim`: expected accurate, clause 9.1 of Master Services Agreement; got accurate, clause 9.1 of Master Services Agreement
- reference `Section 13.1 of the Master Services Agreement`: expected resolved, clause 13.1 of Master Services Agreement; got resolved, clause 13.1 of Master Services Agreement
- authority `9 U.S.C. § 2`: expected well-formed; got unverified, well-formed
- authority `29 C.F.R. Part 1630`: expected well-formed; got unverified, well-formed
- authority `Fed. R. Civ. P. 26(b)(1)`: expected well-formed; got unverified, well-formed
- authority `245 A.3d 101 (Del. Ch. 2021)`: expected well-formed; got unverified, well-formed
- authority `54 U.S.C. § 100101`: expected well-formed; got unverified, well-formed
- authority `63 U.S.C. § 12`: expected malformed; got unverified, malformed: the US Code has no title 63
