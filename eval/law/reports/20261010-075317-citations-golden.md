# EchoMind Law citation checker evaluation (T07) — golden

- Commit: `c8e8abc+changes`
- Checker: `citations-s3.4b`
- Run: 20261010-075317

| Measure | Result |
|---|---|
| items | 55 |
| correct | 55/55 |
| references | 22/22 |
| missing caught | 9/9 |
| false missing | 0 |
| quotations | 9/9 |
| altered caught | 3/3 |
| false alarms | 0 |
| authorities found | 24/24 |
| malformed caught | 5/5 |
| wrongly flagged | 0 |
| all unverified | True |
| external lookups | 0 |
| extra items | 0 |
| ms per check | 35 |

## G1-supply-memo (47 ms)

- quote `within fourteen (14) days of each Order`: expected accurate, clause 5.1 of Supply Agreement; got accurate, clause 5.1 of Supply Agreement
- reference `clause 5.2 of the Supply Agreement`: expected resolved, clause 5.2 of Supply Agreement; got resolved, clause 5.2 of Supply Agreement
- reference `Clause 5.3 of the Supply Agreement`: expected missing; got missing — The Supply Agreement has no clause 5.3.
- quote `preceding six months`: expected altered, clause 9.1 of Supply Agreement; got altered, clause 9.1 of Supply Agreement
- reference `Schedule 1 of the Supply Agreement`: expected resolved, clause Schedule 1 of Supply Agreement; got resolved, clause Schedule 1 of Supply Agreement
- reference `Schedule 2 of the Supply Agreement`: expected missing; got missing — The Supply Agreement has no Schedule 2.

## G2-services-self-references (33 ms)

- reference `clause 8`: expected resolved, clause 8 of self; got resolved, clause 8 of this document
- reference `Schedule 1`: expected resolved, clause Schedule 1 of self; got resolved, clause Schedule 1 of this document
- reference `clause 2.4`: expected missing; got missing — This document has no clause 2.4.
- reference `clause 8`: expected resolved, clause 8 of self; got resolved, clause 8 of this document
- reference `Clause 2.1`: expected resolved, clause 2.1 of self; got resolved, clause 2.1 of this document
- reference `Clause 9.3`: expected missing; got missing — This document has no clause 9.3.

## G3-brief-authorities (20 ms)

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

## G4-lease-letter-quotes (43 ms)

- quote `shall pay the Rent quarterly`: expected accurate, clause 3.1 of Lease; got accurate, clause 3.1 of Lease
- quote `The Landlord shall keep the structure`: expected accurate, clause 4.2 of Lease; got accurate, clause 4.2 of Lease — The words are in clause 4.2, not clause 4.1.
- quote `reviewed every five years`: expected altered, clause 3.2 of Lease; got altered, clause 3.2 of Lease
- quote `assign this lease freely`: expected not_found; got not_found — These words were not found in the documents it points to that you can open.

## G5-spa-disclosure (43 ms)

- quote `within eighteen (18) months after Completion`: expected accurate, clause 6.3 of Share Purchase Agreement; got accurate, clause 6.3 of Share Purchase Agreement
- reference `clause 7.2`: expected resolved, clause 7.2 of Share Purchase Agreement; got resolved, clause 7.2 of Share Purchase Agreement
- quote `matters disclosed in the Disclosure Letter`: expected altered, clause 6.2 of Share Purchase Agreement; got altered, clause 6.2 of Share Purchase Agreement
- reference `paragraph 1 of Schedule 4`: expected resolved, clause 1 of Share Purchase Agreement; got resolved, clause 1 of Share Purchase Agreement
- reference `Schedule 4 of the Share Purchase Agreement`: expected resolved, clause Schedule 4 of Share Purchase Agreement; got resolved, clause Schedule 4 of Share Purchase Agreement
- reference `Schedule 5 of the Share Purchase Agreement`: expected missing; got missing — The Share Purchase Agreement has no Schedule 5.
- quote `the Purchase Price`: expected accurate, clause 7.1 of Share Purchase Agreement; got accurate, clause 7.1 of Share Purchase Agreement
- reference `clause 7.4 of the Share Purchase Agreement`: expected missing; got missing — The Share Purchase Agreement has no clause 7.4.

## G6-licence-internal (35 ms)

- reference `Article 3`: expected resolved, clause Article 3 of self; got resolved, clause Article 3 of this document
- reference `Annex A`: expected resolved, clause Annex A of self; got resolved, clause Annex A of this document
- reference `Annex B`: expected missing; got missing — This document has no Annex B.
- reference `Article 5`: expected missing; got missing — This document has no article 5.
- reference `Section 3.1`: expected resolved, clause 3.1 of self; got resolved, clause 3.1 of this document
- reference `Section 2.1`: expected resolved, clause 2.1 of self; got resolved, clause 2.1 of this document
- reference `Article 9`: expected missing; got missing — This document has no article 9.

## G7-opinion-authorities (27 ms)

- authority `11 U.S.C. § 362(a)`: expected well-formed; got unverified, well-formed
- authority `15 U.S.C. §§ 1`: expected well-formed; got unverified, well-formed
- authority `901 F.3d 1100 (Fed. Cir. 2018)`: expected well-formed; got unverified, well-formed
- authority `22 F.4th 301 (D.C. Cir. 2021)`: expected well-formed; got unverified, well-formed
- authority `55 F.4th 120 (11th Cir. 2034)`: expected malformed; got unverified, malformed: the year 2034 is in the future
- authority `[2022] EWCA Civ 1234`: expected well-formed; got unverified, well-formed
- authority `[2019] EWHC 2468 (Ch)`: expected well-formed; got unverified, well-formed
- authority `[2012] UKHL 15`: expected malformed; got unverified, malformed: the House of Lords stopped sitting in 2009
- authority `Regulation (EC) No 593/2008`: expected well-formed; got unverified, well-formed
- authority `Case T-201/04`: expected well-formed; got unverified, well-formed
