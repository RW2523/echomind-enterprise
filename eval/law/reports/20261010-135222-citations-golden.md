# EchoMind Law citation checker evaluation (T07) — golden

- Commit: `df3b8a5`
- Checker: `citations-s3.4g`
- Run: 20261010-135222

| Measure | Result |
|---|---|
| items | 124 |
| correct | 124/124 |
| references | 42/42 |
| missing caught | 17/17 |
| false missing | 0 |
| quotations | 19/19 |
| altered caught | 6/6 |
| false alarms | 0 |
| authorities found | 63/63 |
| malformed caught | 15/15 |
| wrongly flagged | 0 |
| all unverified | True |
| external lookups | 0 |
| extra items | 0 |
| ms per check | 31 |

## G1-supply-memo (52 ms)

- quote `within fourteen (14) days of each Order`: expected accurate, clause 5.1 of Supply Agreement; got accurate, clause 5.1 of Supply Agreement
- reference `clause 5.2 of the Supply Agreement`: expected resolved, clause 5.2 of Supply Agreement; got resolved, clause 5.2 of Supply Agreement
- reference `Clause 5.3 of the Supply Agreement`: expected missing; got missing — The Supply Agreement has no clause 5.3.
- quote `preceding six months`: expected altered, clause 9.1 of Supply Agreement; got altered, clause 9.1 of Supply Agreement
- reference `Schedule 1 of the Supply Agreement`: expected resolved, clause Schedule 1 of Supply Agreement; got resolved, clause Schedule 1 of Supply Agreement
- reference `Schedule 2 of the Supply Agreement`: expected missing; got missing — The Supply Agreement has no Schedule 2.

## G2-services-self-references (25 ms)

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

## G4-lease-letter-quotes (34 ms)

- quote `shall pay the Rent quarterly`: expected accurate, clause 3.1 of Lease; got accurate, clause 3.1 of Lease
- quote `The Landlord shall keep the structure`: expected accurate, clause 4.2 of Lease; got accurate, clause 4.2 of Lease — The words are in clause 4.2, not clause 4.1.
- quote `reviewed every five years`: expected altered, clause 3.2 of Lease; got altered, clause 3.2 of Lease
- quote `assign this lease freely`: expected not_found; got not_found — These words were not found in the documents it points to that you can open.

## G5-spa-disclosure (47 ms)

- quote `within eighteen (18) months after Completion`: expected accurate, clause 6.3 of Share Purchase Agreement; got accurate, clause 6.3 of Share Purchase Agreement
- reference `clause 7.2`: expected resolved, clause 7.2 of Share Purchase Agreement; got resolved, clause 7.2 of Share Purchase Agreement
- quote `matters disclosed in the Disclosure Letter`: expected altered, clause 6.2 of Share Purchase Agreement; got altered, clause 6.2 of Share Purchase Agreement
- reference `paragraph 1 of Schedule 4`: expected resolved, clause 1 of Share Purchase Agreement; got resolved, clause 1 of Share Purchase Agreement
- reference `Schedule 4 of the Share Purchase Agreement`: expected resolved, clause Schedule 4 of Share Purchase Agreement; got resolved, clause Schedule 4 of Share Purchase Agreement
- reference `Schedule 5 of the Share Purchase Agreement`: expected missing; got missing — The Share Purchase Agreement has no Schedule 5.
- quote `the Purchase Price`: expected accurate, clause 7.1 of Share Purchase Agreement; got accurate, clause 7.1 of Share Purchase Agreement
- reference `clause 7.4 of the Share Purchase Agreement`: expected missing; got missing — The Share Purchase Agreement has no clause 7.4.

## G6-licence-internal (22 ms)

- reference `Article 3`: expected resolved, clause Article 3 of self; got resolved, clause Article 3 of this document
- reference `Annex A`: expected resolved, clause Annex A of self; got resolved, clause Annex A of this document
- reference `Annex B`: expected missing; got missing — This document has no Annex B.
- reference `Article 5`: expected missing; got missing — This document has no article 5.
- reference `Section 3.1`: expected resolved, clause 3.1 of self; got resolved, clause 3.1 of this document
- reference `Section 2.1`: expected resolved, clause 2.1 of self; got resolved, clause 2.1 of this document
- reference `Article 9`: expected missing; got missing — This document has no article 9.

## G7-opinion-authorities (18 ms)

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

## G8-facility-advice-uk (31 ms)

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

## G9-purchase-agreement-internal-us (22 ms)

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

## G11-skeleton-uk (29 ms)

- reference `clause 15.3 of the Supply Contract`: expected resolved, clause 15.3 of Supply Contract; got resolved, clause 15.3 of Supply Contract
- quote `shall not be liable for any indirect loss`: expected altered, clause 9.1 of Supply Contract; got altered, clause 9.1 of Supply Contract
- reference `clause 21.4 of the Supply Contract`: expected missing; got missing — The Supply Contract has no clause 21.4.
- authority `CPR r. 3.4(2)(a)`: expected well-formed; got unverified, well-formed
- authority `CPR Part 24`: expected well-formed; got unverified, well-formed
- authority `CPR Part 112`: expected malformed; got unverified, malformed: the Civil Procedure Rules have no Part 112
- authority `[2021] EWCA Civ 1405`: expected well-formed; got unverified, well-formed
- authority `[2015] UKSC 36`: expected well-formed; got unverified, well-formed
- authority `(1854) 9 Exch 341`: expected well-formed; got unverified, well-formed
- authority `[2007] UKSC 12`: expected malformed; got unverified, malformed: UKSC gave no judgments before 2009

## G12-motion-us (23 ms)

- authority `Fed. R. Civ. P. 12(b)(6)`: expected well-formed; got unverified, well-formed
- authority `Fed. R. Evid. 702`: expected well-formed; got unverified, well-formed
- authority `28 U.S.C. § 1404(a)`: expected well-formed; got unverified, well-formed
- authority `621 F.3d 77 (2d Cir. 2010)`: expected well-formed; got unverified, well-formed
- authority `88 F.4th 501 (3d Cir. 2031)`: expected malformed; got unverified, malformed: the year 2031 is in the future
- authority `Fed. R. App. P. 52`: expected malformed; got unverified, malformed: there is no Rule 52 of the Federal Rules of Appellate Procedure
- reference `Section 2`: expected resolved, clause 2 of self; got resolved, clause 2 of this document
- reference `Section 9`: expected missing; got missing — This document has no section 9.

## G13-lease-note-clean (37 ms)

- quote `The Tenant shall pay the rent quarterly in advance`: expected accurate, clause 4.1 of Lease; got accurate, clause 4.1 of Lease
- quote `good and substantial repair`: expected accurate, clause 6.1 of Lease; got accurate, clause 6.1 of Lease

## G14-distribution-advice-uk (50 ms)

- reference `Clause 2.1 of the Distribution Agreement`: expected resolved, clause 2.1 of Distribution Agreement; got resolved, clause 2.1 of Distribution Agreement
- quote `The Distributor shall purchase Products worth at least EUR 750,000 in each Contract Year`: expected altered, clause 8.3 of Distribution Agreement; got altered, clause 8.3 of Distribution Agreement
- quote `immediately by written notice if the Distributor fails to meet the minimum purchase obligation`: expected accurate, clause 17.2 of Distribution Agreement; got accurate, clause 17.2 of Distribution Agreement
- reference `Clause 19.1 of the Distribution Agreement`: expected missing; got missing — The Distribution Agreement has no clause 19.1.
- authority `Late Payment of Commercial Debts (Interest) Act 1998`: expected well-formed; got unverified, well-formed
- authority `[2019] EWHC 1784 (Comm)`: expected well-formed; got unverified, well-formed
- authority `[2016] UKSC 11`: expected well-formed; got unverified, well-formed
- authority `[1999] EWCA Civ 310`: expected malformed; got unverified, malformed: EWCA Civ gave no judgments before 2001
- authority `Regulation (EU) 2022/720`: expected well-formed; got unverified, well-formed

## G15-opposition-brief-us (35 ms)

- reference `Section 6.4 of the Asset Purchase Agreement`: expected resolved, clause 6.4 of Asset Purchase Agreement; got resolved, clause 6.4 of Asset Purchase Agreement
- reference `Section 11.1 of the Asset Purchase Agreement`: expected resolved, clause 11.1 of Asset Purchase Agreement; got resolved, clause 11.1 of Asset Purchase Agreement
- reference `Section 14.2 of the Asset Purchase Agreement`: expected missing; got missing — The Asset Purchase Agreement has no section 14.2.
- authority `Fed. R. Civ. P. 56(a)`: expected well-formed; got unverified, well-formed
- authority `Fed. R. Evid. 1006`: expected well-formed; got unverified, well-formed
- authority `15 U.S.C. § 78j(b)`: expected well-formed; got unverified, well-formed
- authority `17 C.F.R. § 240.10b-5`: expected well-formed; got unverified, well-formed
- authority `412 F.3d 201 (4th Cir. 2005)`: expected well-formed; got unverified, well-formed
- authority `Fed. R. Crim. P. 74`: expected malformed; got unverified, malformed: there is no Rule 74 of the Federal Rules of Criminal Procedure
- authority `91 F.5th 12 (2d Cir. 2024)`: expected malformed; got unverified, malformed: F.5th is not a known reporter

## G16-licence-note-clean (25 ms)

- quote `a non-exclusive licence to use the Software in the United Kingdom`: expected accurate, clause 3.1 of Licence Agreement; got accurate, clause 3.1 of Licence Agreement
- quote `Each party shall keep the other party's Confidential Information secret for five years after termination`: expected accurate, clause 10.2 of Licence Agreement; got accurate, clause 10.2 of Licence Agreement
