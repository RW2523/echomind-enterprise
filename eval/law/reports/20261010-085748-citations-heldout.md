# EchoMind Law citation checker evaluation (T07) — heldout

- Commit: `5dc27a4`
- Checker: `citations-s3.4d`
- Run: 20261010-085748

| Measure | Result |
|---|---|
| items | 28 |
| correct | 27/28 |
| references | 11/11 |
| missing caught | 4/4 |
| false missing | 0 |
| quotations | 3/3 |
| altered caught | 1/1 |
| false alarms | 0 |
| authorities found | 13/14 |
| malformed caught | 3/3 |
| wrongly flagged | 0 |
| all unverified | True |
| external lookups | 0 |
| extra items | 0 |
| ms per check | 36 |

## N1-facility-advice-uk (46 ms)

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

## N2-purchase-agreement-internal-us (34 ms)

- reference `Section 3.2`: expected resolved, clause 3.2 of self; got resolved, clause 3.2 of this document
- reference `Exhibit A`: expected resolved, clause Exhibit A of self; got resolved, clause Exhibit A of this document
- reference `Section 3.3(b)`: expected resolved, clause 3.3(b) of self; got resolved, clause 3.3(b) of this document
- reference `Exhibit C`: expected missing; got missing — This document has no Exhibit C.
- reference `Section 7.1`: expected resolved, clause 7.1 of self; got resolved, clause 7.1 of this document
- reference `Section 7.4`: expected missing; got missing — This document has no section 7.4.

## N3-litigation-memo-us (27 ms)

- authority `28 U.S.C. § 1441(a)`: expected well-formed; got unverified, well-formed
- authority `Fed. R. Civ. P. 81(c)`: expected well-formed; got not found by the parser **✗**
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
