# EchoMind Law citation checker evaluation (T07) — heldout

- Commit: `c14e77b`
- Checker: `citations-s3.4`
- Run: 20261010-075004

| Measure | Result |
|---|---|
| items | 25 |
| correct | 20/25 |
| references | 10/12 |
| missing caught | 5/5 |
| false missing | 1 |
| quotations | 0/3 |
| altered caught | 0/1 |
| false alarms | 0 |
| authorities found | 10/10 |
| malformed caught | 2/2 |
| wrongly flagged | 0 |
| all unverified | True |
| external lookups | 0 |
| extra items | 5 |
| ms per check | 58 |

## H1-spa-disclosure (115 ms)

- reference `section 6.3 of the Share Purchase Agreement`: expected resolved, clause 6.3 of Share Purchase Agreement; got not found by the parser **✗**
- quote `within eighteen (18) months after Completion`: expected accurate, clause 6.3 of Share Purchase Agreement; got not found by the parser **✗**
- reference `clause 7.2`: expected resolved, clause 7.2 of Share Purchase Agreement; got missing — This document has no clause 7.2. **✗**
- quote `matters disclosed in the Disclosure Letter`: expected altered, clause 6.2 of Share Purchase Agreement; got not found by the parser **✗**
- reference `Schedule 4 of the Share Purchase Agreement`: expected resolved, clause Schedule 4 of Share Purchase Agreement; got resolved, clause Schedule 4 of Share Purchase Agreement
- reference `Schedule 5 of the Share Purchase Agreement`: expected missing; got missing — The Share Purchase Agreement has no Schedule 5.
- quote `the Purchase Price`: expected accurate, clause 7.1 of Share Purchase Agreement; got not found by the parser **✗**
- reference `clause 7.4 of the Share Purchase Agreement`: expected missing; got missing — The Share Purchase Agreement has no clause 7.4.

Also found (not in the expectations):
- quote `).

2. Time limits
2.1 Under section 6.3 of the Share Purchase Agreement, claims must be notified`: not_found
- quote `.
2.2 By clause 7.2 of the SPA, the Seller is not liable for small claims.

3. Warranties
3.1 Clause 6.2 of the Share Purchase Agreement qualifies the Warranties by`: not_found
- quote `.
3.2 The tax warranty is paragraph 1 of Schedule 4 of the Share Purchase Agreement; the property warranties are in Schedule 5 of the Share Purchase Agreement.
3.3 Clause 7.1 of the Share Purchase Agreement caps liability at`: not_found
- reference `paragraph 1`: resolved
- reference `Clause 7.1 of the Share Purchase Agreement`: resolved

## H2-licence-internal (33 ms)

- reference `Article 3`: expected resolved, clause Article 3 of self; got resolved, clause Article 3 of this document
- reference `Annex A`: expected resolved, clause Annex A of self; got resolved, clause Annex A of this document
- reference `Annex B`: expected missing; got missing — This document has no Annex B.
- reference `Article 5`: expected missing; got missing — This document has no article 5.
- reference `Section 3.1`: expected resolved, clause 3.1 of self; got resolved, clause 3.1 of this document
- reference `Section 2.1`: expected resolved, clause 2.1 of self; got resolved, clause 2.1 of this document
- reference `Article 9`: expected missing; got missing — This document has no article 9.

## H3-opinion-authorities (25 ms)

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
