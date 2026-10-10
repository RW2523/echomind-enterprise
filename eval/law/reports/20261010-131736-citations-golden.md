# EchoMind Law citation checker evaluation (T07) — golden

- Commit: `41eafcf+changes`
- Checker: `citations-s3.4e`
- Run: 20261010-131736

| Measure | Result |
|---|---|
| items | 83 |
| correct | 83/83 |
| references | 33/33 |
| missing caught | 13/13 |
| false missing | 0 |
| quotations | 12/12 |
| altered caught | 4/4 |
| false alarms | 0 |
| authorities found | 38/38 |
| malformed caught | 8/8 |
| wrongly flagged | 0 |
| all unverified | True |
| external lookups | 0 |
| extra items | 0 |
| ms per check | 33 |

## G1-supply-memo (57 ms)

- quote `within fourteen (14) days of each Order`: expected accurate, clause 5.1 of Supply Agreement; got accurate, clause 5.1 of Supply Agreement
- reference `clause 5.2 of the Supply Agreement`: expected resolved, clause 5.2 of Supply Agreement; got resolved, clause 5.2 of Supply Agreement
- reference `Clause 5.3 of the Supply Agreement`: expected missing; got missing — The Supply Agreement has no clause 5.3.
- quote `preceding six months`: expected altered, clause 9.1 of Supply Agreement; got altered, clause 9.1 of Supply Agreement
- reference `Schedule 1 of the Supply Agreement`: expected resolved, clause Schedule 1 of Supply Agreement; got resolved, clause Schedule 1 of Supply Agreement
- reference `Schedule 2 of the Supply Agreement`: expected missing; got missing — The Supply Agreement has no Schedule 2.

## G2-services-self-references (37 ms)

- reference `clause 8`: expected resolved, clause 8 of self; got resolved, clause 8 of this document
- reference `Schedule 1`: expected resolved, clause Schedule 1 of self; got resolved, clause Schedule 1 of this document
- reference `clause 2.4`: expected missing; got missing — This document has no clause 2.4.
- reference `clause 8`: expected resolved, clause 8 of self; got resolved, clause 8 of this document
- reference `Clause 2.1`: expected resolved, clause 2.1 of self; got resolved, clause 2.1 of this document
- reference `Clause 9.3`: expected missing; got missing — This document has no clause 9.3.

## G3-brief-authorities (29 ms)

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

## G4-lease-letter-quotes (31 ms)

- quote `shall pay the Rent quarterly`: expected accurate, clause 3.1 of Lease; got accurate, clause 3.1 of Lease
- quote `The Landlord shall keep the structure`: expected accurate, clause 4.2 of Lease; got accurate, clause 4.2 of Lease — The words are in clause 4.2, not clause 4.1.
- quote `reviewed every five years`: expected altered, clause 3.2 of Lease; got altered, clause 3.2 of Lease
- quote `assign this lease freely`: expected not_found; got not_found — These words were not found in the documents it points to that you can open.

## G5-spa-disclosure (27 ms)

- quote `within eighteen (18) months after Completion`: expected accurate, clause 6.3 of Share Purchase Agreement; got accurate, clause 6.3 of Share Purchase Agreement
- reference `clause 7.2`: expected resolved, clause 7.2 of Share Purchase Agreement; got resolved, clause 7.2 of Share Purchase Agreement
- quote `matters disclosed in the Disclosure Letter`: expected altered, clause 6.2 of Share Purchase Agreement; got altered, clause 6.2 of Share Purchase Agreement
- reference `paragraph 1 of Schedule 4`: expected resolved, clause 1 of Share Purchase Agreement; got resolved, clause 1 of Share Purchase Agreement
- reference `Schedule 4 of the Share Purchase Agreement`: expected resolved, clause Schedule 4 of Share Purchase Agreement; got resolved, clause Schedule 4 of Share Purchase Agreement
- reference `Schedule 5 of the Share Purchase Agreement`: expected missing; got missing — The Share Purchase Agreement has no Schedule 5.
- quote `the Purchase Price`: expected accurate, clause 7.1 of Share Purchase Agreement; got accurate, clause 7.1 of Share Purchase Agreement
- reference `clause 7.4 of the Share Purchase Agreement`: expected missing; got missing — The Share Purchase Agreement has no clause 7.4.

## G6-licence-internal (25 ms)

- reference `Article 3`: expected resolved, clause Article 3 of self; got resolved, clause Article 3 of this document
- reference `Annex A`: expected resolved, clause Annex A of self; got resolved, clause Annex A of this document
- reference `Annex B`: expected missing; got missing — This document has no Annex B.
- reference `Article 5`: expected missing; got missing — This document has no article 5.
- reference `Section 3.1`: expected resolved, clause 3.1 of self; got resolved, clause 3.1 of this document
- reference `Section 2.1`: expected resolved, clause 2.1 of self; got resolved, clause 2.1 of this document
- reference `Article 9`: expected missing; got missing — This document has no article 9.

## G7-opinion-authorities (21 ms)

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

## G8-facility-advice-uk (51 ms)

- quote `shall not exceed 3.75:1`: expected altered, clause 22.1 of Facility Agreement; got altered, clause 22.1 of Facility Agreement
- quote `shall not be less than 4.00:1`: expected accurate, clause 22.2 of Facility Agreement; got accurate, clause 22.2 of Facility Agreement
- reference `Clause 22.5 of the Facility Agreement`: expected missing; got missing — The Facility Agreement has no clause 22.5.
- quote `within three Business Days of its due date`: expected accurate, clause 27.1 of Facility Agreement; got accurate, clause 27.1 of Facility Agreement
- reference `paragraph 2 of Schedule 3`: expected resolved, clause 2 of Facility Agreement; got resolved, clause 2 of Facility Agreement
- reference `Schedule 3 of the Facility Agreement`: expected resolved, clause Schedule 3 of Facility Agreement; got resolved, clause Schedule 3 of Facility Agreement
- authority `Companies Act 2006`: expected well-formed; got unverified, well-formed
- authority `Limitation Act 1980`: expected well-formed; got unverified, well-formed
- authority `[2023] EWHC 101 (KB)`: expected well-formed; got unverified, well-formed
- authority `[1998] EWCA Civ 12`: expected malformed; got unverified, malformed: EWCA Civ gave no judgments before 2001

## G9-purchase-agreement-internal-us (23 ms)

- reference `Section 3.2`: expected resolved, clause 3.2 of self; got resolved, clause 3.2 of this document
- reference `Exhibit A`: expected resolved, clause Exhibit A of self; got resolved, clause Exhibit A of this document
- reference `Section 3.3(b)`: expected resolved, clause 3.3(b) of self; got resolved, clause 3.3(b) of this document
- reference `Exhibit C`: expected missing; got missing — This document has no Exhibit C.
- reference `Section 7.1`: expected resolved, clause 7.1 of self; got resolved, clause 7.1 of this document
- reference `Section 7.4`: expected missing; got missing — This document has no section 7.4.

## G10-litigation-memo-us (31 ms)

- authority `28 U.S.C. § 1441(a)`: expected well-formed; got unverified, well-formed
- authority `Fed. R. Civ. P. 81(c)`: expected well-formed; got unverified, well-formed
- authority `571 U.S. 402 (2014)`: expected well-formed; got unverified, well-formed
- authority `140 S. Ct. 1201 (2020)`: expected well-formed; got unverified, well-formed
- authority `812 F. App'x 330 (3d Cir. 2020)`: expected well-formed; got unverified, well-formed
- authority `61 Cal. Rptr. 3d 97 (Cal. Ct. App. 2007)`: expected well-formed; got unverified, well-formed
- authority `288 So. 4th 15 (Fla. 2019)`: expected malformed; got unverified, malformed: So. 4th is not a known reporter
- authority `77 F.3d 501 (5th Cir. 2029)`: expected malformed; got unverified, malformed: the year 2029 is in the future
- authority `Regulation (EU) No 1215/2012`: expected well-formed; got unverified, well-formed
- authority `Directive (EU) 2019/1937`: expected well-formed; got unverified, well-formed
- reference `paragraph 2.1`: expected resolved, clause 2.1 of self; got resolved, clause 2.1 of this document
- reference `Paragraph 5.2`: expected missing; got missing — This document has no paragraph 5.2.
