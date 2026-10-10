# EchoMind Law citation checker evaluation (T07) — golden

- Commit: `1d2875a+changes`
- Checker: `citations-s3.4`
- Run: 20261010-074818

| Measure | Result |
|---|---|
| items | 30 |
| correct | 30/30 |
| references | 10/10 |
| missing caught | 4/4 |
| false missing | 0 |
| quotations | 6/6 |
| altered caught | 2/2 |
| false alarms | 0 |
| authorities found | 14/14 |
| malformed caught | 3/3 |
| wrongly flagged | 0 |
| all unverified | True |
| external lookups | 0 |
| extra items | 0 |
| ms per check | 36 |

## G1-supply-memo (54 ms)

- quote `within fourteen (14) days of each Order`: expected accurate, clause 5.1 of Supply Agreement; got accurate, clause 5.1 of Supply Agreement
- reference `clause 5.2 of the Supply Agreement`: expected resolved, clause 5.2 of Supply Agreement; got resolved, clause 5.2 of Supply Agreement
- reference `Clause 5.3 of the Supply Agreement`: expected missing; got missing — The Supply Agreement has no clause 5.3.
- quote `preceding six months`: expected altered, clause 9.1 of Supply Agreement; got altered, clause 9.1 of Supply Agreement
- reference `Schedule 1 of the Supply Agreement`: expected resolved, clause Schedule 1 of Supply Agreement; got resolved, clause Schedule 1 of Supply Agreement
- reference `Schedule 2 of the Supply Agreement`: expected missing; got missing — The Supply Agreement has no Schedule 2.

## G2-services-self-references (20 ms)

- reference `clause 8`: expected resolved, clause 8 of self; got resolved, clause 8 of this document
- reference `Schedule 1`: expected resolved, clause Schedule 1 of self; got resolved, clause Schedule 1 of this document
- reference `clause 2.4`: expected missing; got missing — This document has no clause 2.4.
- reference `clause 8`: expected resolved, clause 8 of self; got resolved, clause 8 of this document
- reference `Clause 2.1`: expected resolved, clause 2.1 of self; got resolved, clause 2.1 of this document
- reference `Clause 9.3`: expected missing; got missing — This document has no clause 9.3.

## G3-brief-authorities (22 ms)

- authority `42 U.S.C. § 1983`: expected well-formed; got unverified, well-formed
- authority `28 U.S.C. § 1331`: expected well-formed; got unverified, well-formed
- authority `Palmer v. Oakridge Freight, 544 F.3d 210 (7th Cir. 2008)`: expected well-formed; got unverified, well-formed
- authority `87 F. Supp. 3d 455 (S.D.N.Y. 2015)`: expected well-formed; got unverified, well-formed
- authority `310 F.4th 88 (12th Cir. 2022)`: expected malformed; got unverified, malformed: there is no 12th Cir.
- authority `45 F.6th 12 (9th Cir. 2019)`: expected malformed; got unverified, malformed: F.6th is not a known reporter
- authority `[2017] UKSC 59`: expected well-formed; got unverified, well-formed
- authority `[2008] UKHL 48`: expected well-formed; got unverified, well-formed
- authority `[2016] UKHL 7`: expected malformed; got unverified, malformed: the House of Lords stopped sitting in 2009
- authority `(1854) 9 Exch 341`: expected well-formed; got unverified, well-formed
- authority `Regulation (EU) 2016/679`: expected well-formed; got unverified, well-formed
- authority `Directive 2011/83/EU`: expected well-formed; got unverified, well-formed
- authority `Case C-311/18`: expected well-formed; got unverified, well-formed
- authority `29 C.F.R. § 1910.1200`: expected well-formed; got unverified, well-formed

## G4-lease-letter-quotes (47 ms)

- quote `shall pay the Rent quarterly`: expected accurate, clause 3.1 of Lease; got accurate, clause 3.1 of Lease
- quote `The Landlord shall keep the structure`: expected accurate, clause 4.2 of Lease; got accurate, clause 4.2 of Lease — The words are in clause 4.2, not clause 4.1.
- quote `reviewed every five years`: expected altered, clause 3.2 of Lease; got altered, clause 3.2 of Lease
- quote `assign this lease freely`: expected not_found; got not_found — These words were not found in the documents it points to that you can open.
