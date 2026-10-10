# EchoMind Law citation checker evaluation (T07) — heldout

- Commit: `9a3894f`
- Checker: `citations-s3.4f`
- Run: 20261010-132919

| Measure | Result |
|---|---|
| items | 21 |
| correct | 20/21 |
| references | 5/5 |
| missing caught | 2/2 |
| false missing | 0 |
| quotations | 4/4 |
| altered caught | 1/1 |
| false alarms | 0 |
| authorities found | 11/12 |
| malformed caught | 3/3 |
| wrongly flagged | 0 |
| all unverified | True |
| external lookups | 0 |
| extra items | 0 |
| ms per check | 32 |

## R1-distribution-advice-uk (35 ms)

- reference `Clause 2.1 of the Distribution Agreement`: expected resolved, clause 2.1 of Distribution Agreement; got resolved, clause 2.1 of Distribution Agreement
- quote `The Distributor shall purchase Products worth at least EUR 750,000 in each Contract Year`: expected altered, clause 8.3 of Distribution Agreement; got altered, clause 8.3 of Distribution Agreement
- quote `immediately by written notice if the Distributor fails to meet the minimum purchase obligation`: expected accurate, clause 17.2 of Distribution Agreement; got accurate, clause 17.2 of Distribution Agreement
- reference `Clause 19.1 of the Distribution Agreement`: expected missing; got missing — The Distribution Agreement has no clause 19.1.
- authority `Late Payment of Commercial Debts (Interest) Act 1998`: expected well-formed; got not found by the parser **✗**
- authority `[2019] EWHC 1784 (Comm)`: expected well-formed; got unverified, well-formed
- authority `[2016] UKSC 11`: expected well-formed; got unverified, well-formed
- authority `[1999] EWCA Civ 310`: expected malformed; got unverified, malformed: EWCA Civ gave no judgments before 2001
- authority `Regulation (EU) 2022/720`: expected well-formed; got unverified, well-formed

## R2-opposition-brief-us (23 ms)

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

## R3-licence-note-clean (37 ms)

- quote `a non-exclusive licence to use the Software in the United Kingdom`: expected accurate, clause 3.1 of Licence Agreement; got accurate, clause 3.1 of Licence Agreement
- quote `Each party shall keep the other party's Confidential Information secret for five years after termination`: expected accurate, clause 10.2 of Licence Agreement; got accurate, clause 10.2 of Licence Agreement
