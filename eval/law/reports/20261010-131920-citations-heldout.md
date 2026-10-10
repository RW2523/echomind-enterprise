# EchoMind Law citation checker evaluation (T07) — heldout

- Commit: `689331f`
- Checker: `citations-s3.4e`
- Run: 20261010-131920

| Measure | Result |
|---|---|
| items | 20 |
| correct | 19/20 |
| references | 4/4 |
| missing caught | 2/2 |
| false missing | 0 |
| quotations | 3/3 |
| altered caught | 1/1 |
| false alarms | 0 |
| authorities found | 12/13 |
| malformed caught | 4/4 |
| wrongly flagged | 0 |
| all unverified | True |
| external lookups | 0 |
| extra items | 1 |
| ms per check | 27 |

## Q1-skeleton-uk (28 ms)

- reference `clause 15.3 of the Supply Contract`: expected resolved, clause 15.3 of Supply Contract; got resolved, clause 15.3 of Supply Contract
- quote `shall not be liable for any indirect loss`: expected altered, clause 9.1 of Supply Contract; got altered, clause 9.1 of Supply Contract
- reference `clause 21.4 of the Supply Contract`: expected missing; got missing — The Supply Contract has no clause 21.4.
- authority `CPR r. 3.4(2)(a)`: expected well-formed; got not found by the parser **✗**
- authority `CPR Part 24`: expected well-formed; got unverified, well-formed
- authority `CPR Part 112`: expected malformed; got unverified, malformed: the Civil Procedure Rules have no Part 112
- authority `[2021] EWCA Civ 1405`: expected well-formed; got unverified, well-formed
- authority `[2015] UKSC 36`: expected well-formed; got unverified, well-formed
- authority `(1854) 9 Exch 341`: expected well-formed; got unverified, well-formed
- authority `[2007] UKSC 12`: expected malformed; got unverified, malformed: UKSC gave no judgments before 2009

Also found (not in the expectations):
- authority `CPR r. 3.4(2)`: unverified

## Q2-motion-us (31 ms)

- authority `Fed. R. Civ. P. 12(b)(6)`: expected well-formed; got unverified, well-formed
- authority `Fed. R. Evid. 702`: expected well-formed; got unverified, well-formed
- authority `28 U.S.C. § 1404(a)`: expected well-formed; got unverified, well-formed
- authority `621 F.3d 77 (2d Cir. 2010)`: expected well-formed; got unverified, well-formed
- authority `88 F.4th 501 (3d Cir. 2031)`: expected malformed; got unverified, malformed: the year 2031 is in the future
- authority `Fed. R. App. P. 52`: expected malformed; got unverified, malformed: there is no Rule 52 of the Federal Rules of Appellate Procedure
- reference `Section 2`: expected resolved, clause 2 of self; got resolved, clause 2 of this document
- reference `Section 9`: expected missing; got missing — This document has no section 9.

## Q3-lease-note-clean (23 ms)

- quote `The Tenant shall pay the rent quarterly in advance`: expected accurate, clause 4.1 of Lease; got accurate, clause 4.1 of Lease
- quote `good and substantial repair`: expected accurate, clause 6.1 of Lease; got accurate, clause 6.1 of Lease
