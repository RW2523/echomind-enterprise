# SYN-02 bible — Larkspur Health Partners: Ostrander vendor ransomware incident

Answer-key fact sheet. Every fact below is what the documents say; where documents disagree, the "Traps" section
says which one prevails. All names, companies, addresses, domains and figures are invented.

## 1. Parties and people

### Larkspur Health Partners, Inc. (client; HIPAA Covered Entity; Named Insured)
- New York corporation; group of eight outpatient clinics; HQ 400 Tillary Court, Brooklyn, NY 11201; main line (718) 555-0110.
- Email domain `larkspurhealth.example`. Notice mailboxes: `legalnotices@`, `privacy@`, `security@`. 24x7 security hotline (718) 555-0190 (MSA Exhibit C Part 4).
- Dr. Celia Fontaine — Chief Executive Officer and director (signed the MSA on January 26, 2022). cfontaine@
- Lionel Marchbanks — General Counsel (signed the BAA on January 26, 2022; countersigned the engagement letter March 17, 2025). lmarchbanks@, (718) 555-0115
- Annette Rhee — Privacy Officer (author of patient letter template v4; HIPAA incident file PRIV-2025-007). arhee@, (718) 555-0124
- Tomasz Wierzbicki — Chief Information Security Officer (CISO). twierzbicki@, (718) 555-0131, cell (917) 555-0186
- Paul Kowalczyk — Chief Financial Officer (signed the First Amendment on June 21, 2023).
- Beatrice Nolan — Corporate Secretary (not a director; signed the March 20, 2025 minutes).
- Board of Directors (7): Margaret Lindqvist (Chair), Dr. Celia Fontaine, Sunita Varadarajan, Leon Abernathy, Dr. Raymond Ibsen, Gerald Pomeroy, Harold Nakamura. Quorum = majority of the whole Board (4).

### The eight clinics
| Code | Clinic | Location |
|---|---|---|
| LHP-01 | Larkspur Flatbush Clinic | Brooklyn |
| LHP-02 | Larkspur Bay Ridge Clinic | Brooklyn |
| LHP-03 | Larkspur Williamsburg Clinic | Brooklyn |
| LHP-04 | Larkspur Astoria Clinic | Queens |
| LHP-05 | Larkspur Jamaica Clinic | Queens |
| LHP-06 | Larkspur Fordham Clinic | Bronx |
| LHP-07 | Larkspur St. George Clinic | Staten Island |
| LHP-08 | Larkspur Yonkers Clinic | Yonkers (Westchester County) |

### Ostrander Billing Solutions LLC (vendor; HIPAA Business Associate; adverse)
- Delaware limited liability company; principal office 410 Wexley Commons Drive, Suite 300, Hartford, CT 06106; main (860) 555-0170.
- Named inconsistently on purpose: "Ostrander Billing Solutions LLC" (MSA, First Amendment), "Ostrander Billing Solutions, LLC" (BAA, notice letter), "OBS", "Ostrander", "the Vendor", "Business Associate".
- Thomas R. Vandermeer — President & CEO (signed the MSA on January 28, 2022).
- Bryce Kettering — Chief Operating Officer (signed the BAA on January 28, 2022 and the First Amendment on June 27, 2023; signed the March 14, 2025 notice letter; author of the June 27, 2025 email). bkettering@ostranderbilling.example, (860) 555-0172
- Nadia Okafor — Director of Information Security (phoned Wierzbicki on March 10, 2025 at about 4:15 p.m.). nokafor@, (860) 555-0174
- Lauren Szabo — Deputy General Counsel. lszabo@
- Ostrander's own incident response firm: Ironvale Response Group (named in the notice letter and Cobaltline reports only).

### Calder & Wynne LLP (Larkspur's counsel)
- 1180 Hollis Avenue, 22nd Floor, New York, NY 10017; main (212) 555-0100; domain `calderwynne.example`.
- Alice Moreno — Partner (signed the engagement letter). amoreno@, (212) 555-0151
- Kenji Arai — senior associate, Corporate & Regulatory; leads the matter day to day; signed the notice of circumstances; sent the indemnity demand. karai@, (212) 555-0148
- Omar Haddad — paralegal. ohaddad@, (212) 555-0157
- Client-matter number 30517.001.

### Cobaltline Forensics LLC (forensic firm, retained by Calder & Wynne for Larkspur)
- 31 Varick Mews, 9th Floor, New York, NY 10013; UK office Cobaltline Forensics (UK) Ltd, 4 Tanner's Wharf, Manchester M4 7HQ.
- Imogen Thwaite — Principal Forensic Analyst, Manchester (writes UK dates: "9 March 2025", "09/03/2025"). +44 161 496 0732
- Derek Solano — Managing Director, New York.
- Engagement CF-2025-0317, engaged March 17, 2025 by Calder & Wynne on behalf of Larkspur. Reports CF-2025-0317-PR-01 (preliminary, draft v0.3) and CF-2025-0317-FR-01 v1.0 (final).

### Halvard Mutual Insurance Company (cyber insurer; adverse on coverage)
- Cyber & Technology Claims, 600 Larchmont Wharf, Boston, MA 02210; cyberclaims@halvardmutual.example; 24/7 incident hotline (800) 555-0188.
- Priya Venkataraman — Senior Claims Specialist (signed the reservation of rights letter). (617) 555-0129
- Claim number HMC-CL-2025-03391.
- Producer/broker: Pinecrest Risk Advisors, LLC, New York — Graham Tolliver, (212) 555-0163.

### Other vendors (named in passing)
- Vantorra Identity Services, LLC — credit monitoring and identity restoration; USD 1.50 per individual per month.
- Brightwater Mailing Services — notification print and mail; USD 1.35 per letter.
- Harborline Contact Services — call center; USD 41,300.00 (single figure in the demand).
- Patient call center toll-free number (855) 555-0137.

## 2. Contracts

### 2.1 Master Services Agreement (doc 01) — Contract No. LHP-OBS-2022-01
- Effective Date February 1, 2022. Signed by Fontaine (January 26, 2022) and Vandermeer (January 28, 2022).
- Initial Term three years (to January 31, 2025); automatic one-year renewals unless 90 days' notice of non-renewal (Section 3). At the incident the MSA was in its first Renewal Term (February 1, 2025 – January 31, 2026).
- Section 1 definitions: "Client Data" includes PHI; "Security Incident" = actual or reasonably suspected unauthorized access, acquisition, use, disclosure, encryption, corruption or loss of Client Data, or compromise of any Vendor System that stores, processes or transmits Client Data; "discovery" (Section 1.10) = the first day on which a Security Incident is known to Vendor or, by exercising reasonable diligence, would have been known to Vendor; "Business Day" = Monday–Friday excluding New York bank holidays.
- Section 4 Fees: per Exhibit A; monthly invoices; net 45 days.
- Section 9 Data Security: 9.1 security program per Exhibit C; 9.2 encryption of all Client Data in transit (TLS 1.2+) and at rest "wherever stored, including databases, file servers, file shares, backup media, data exports and staging or test environments" (AES-256 or equivalent); 9.3 access controls, MFA, patching per Exhibit C; 9.4 (original) written notice of a Security Incident without undue delay and within seventy-two (72) hours after discovery; telephone contact is not notice; notice must contain listed content; 9.5 investigation, remediation, twice-weekly updates for 30 days, cooperation, access for Client's forensic examiner, copies of reports; 9.6 Client controls third-party notifications; 9.7 audits (annual SOC 2 Type II or equivalent certification); 9.8 return/destruction (NIST SP 800-88); 9.9 Vendor bears its own incident costs. Section 7.3 and 21.4: a Security Incident (including ransomware) is not force majeure. Section 10.2(e) warrants the vendor security questionnaire of November 18, 2021. Section 14.3: Client may terminate immediately for a material breach of Section 9/BAA.
- Section 11 Indemnification: 11.1(a) IP infringement; 11.1(b) gross negligence, willful misconduct, violation of law; 11.1(c) any breach of Section 9 or the BAA and any Security Incident involving Client Data in Vendor's (or Subcontractors') possession, including third-party claims and, whether or not a third-party claim is asserted, (i) notification costs, (ii) credit monitoring and identity restoration for not less than 12 months "or such longer period as is required by applicable law, demanded by a Governmental Authority or reasonably determined by Client to be appropriate", (iii) call center, (iv) forensic investigation, (v) fines and penalties to the extent permitted by law, (vi) reasonable attorneys' fees. 11.2 Client indemnity. 11.3 procedure (prompt written notice; failure relieves only to extent of prejudice).
- Section 12 Limitation of Liability: 12.1 consequential-loss exclusion; 12.2 general cap = total Fees paid or payable during the twelve (12) months immediately preceding the event giving rise to the claim; 12.3 cap carve-outs: (a) Client payment obligations, (b) Section 11.1(a), (c) gross negligence, fraud, willful misconduct, (d) breach of Section 8 other than a breach arising from a Security Incident.
- Section 11.4: amounts under 11.1(c) payable within thirty (30) days of Client's written demand with supporting documentation.
- Section 13 Insurance: CGL USD 1,000,000 per occurrence / USD 2,000,000 aggregate; E&O USD 2,000,000 per claim; Vendor cyber liability USD 3,000,000 (raised to USD 5,000,000 by the First Amendment). (The USD 2,000,000 CGL/E&O figures are a distractor for the USD 2,000,000 Data Breach Cap.)
- Section 18 Dispute Resolution: 18.1 executive escalation within fifteen (15) Business Days of a dispute notice; 18.2 courts of New York County. Section 19 New York law.
- Section 20 Notices: in writing; effective (a) on personal delivery, (b) one Business Day after deposit with a national overnight courier, (c) on transmission by email if sent before 5:00 p.m. recipient's local time on a Business Day and a copy is sent the same day by courier (otherwise next Business Day). Client notices: Attn General Counsel, legalnotices@larkspurhealth.example, copy to Privacy Officer. Vendor notices: Attn Chief Operating Officer, legal@ostranderbilling.example.
- Section 22.3 Order of precedence: (i) the BAA, solely with respect to use, disclosure and safeguarding of PHI; (ii) amendments (later prevails over earlier); (iii) body of the MSA; (iv) Exhibit C; (v) Exhibit B; (vi) Exhibit A; (vii) SOWs.
- Exhibit A (original fees, fixed for the Initial Term): platform fee USD 31,500/month; claims USD 1.25/claim (min 5,000/month); statements USD 0.55; denials 4.0% of recoveries; credentialing USD 250/provider/month; implementation USD 45,000 one-time.
- Exhibit B Service Levels: platform availability 99.5% monthly; claims submitted within 2 Business Days of charge entry (98%); remittance posting within 3 Business Days (97%); service credits 2%–10% of monthly platform fee, capped at 15%; chronic failure = 3 misses in 6 months.
- Exhibit C Security Requirements: Part 1 program (annual SOC 2 Type II or equivalent certification approved by Client); Part 2 technical: 2.1 encryption (cross-refers 9.2), 2.2 MFA for all remote access, 2.3 privileged access, 2.4 patching — critical (CVSS 9.0+) within fourteen (14) days of patch availability, high within thirty (30) days, 2.5 EDR on all servers, 2.6 logging retained one (1) year, 2.7 immutable backups, 2.8 segmentation, 2.9 data minimisation/retention: exports deleted within thirty (30) days; Part 3 personnel; Part 4 incident notification contacts.

### 2.2 Business Associate Agreement (doc 02)
- Effective February 1, 2022; Larkspur ("Covered Entity") and "Ostrander Billing Solutions, LLC" ("Business Associate"). Signed by Marchbanks (January 26, 2022) and Kettering (January 28, 2022). 1.2(k) defines Business Day; 2.5 presumption of Breach; 2.7 subcontractors; 2.8–2.10 access/amendment/accounting within 10 Business Days; 6.2 termination for cause (30-day cure); 6.4 survival.
- 1.2(d) discovery = first day known to BA or by reasonable diligence would have been known (mirrors 45 C.F.R. § 164.410(a)(2)).
- 2.3 report non-permitted uses/disclosures and Security Incidents without unreasonable delay.
- 2.4 report a Breach of Unsecured PHI "without unreasonable delay and in no case later than ten (10) business days after discovery", in writing, with the content of 45 C.F.R. § 164.410(c).
- 2.13 and 5.3 cooperation; 5.1 Covered Entity controls notifications; 5.2 BA reimburses Covered Entity's reasonable costs of notification (printing, postage, call center, notices to Secretary/media/state) — "in addition to and not in limitation of Section 11 of the Underlying Agreement".
- 7.4 relationship to the MSA: BAA controls on conflict regarding PHI, PROVIDED that more stringent obligations of the Underlying Agreement (expressly including shorter notice periods) apply in addition and are not a conflict.
- 7.6 BA is an independent contractor, not an agent. 7.7 New York law.

### 2.3 First Amendment (doc 03)
- Effective July 1, 2023; signed by Kowalczyk (June 21, 2023) and Kettering (June 27, 2023).
- Section 1: MSA 9.4 replaced — notice within forty-eight (48) hours after discovery (content and "telephone is not notice" language kept).
- Section 2: MSA 12.2 prefaced "Except as provided in Sections 12.3 and 12.4,".
- Section 3: new 12.4 "Data Breach Cap": Vendor's aggregate liability for (a) any breach of Section 9 or of the BAA and (b) indemnification under Section 11.1(c) shall not exceed USD 2,000,000, in lieu of (not in addition to) Section 12.2; Data Breach Costs are direct damages not excluded by 12.1; 12.3 carve-outs still apply. Defines "Data Breach Costs".
- Section 4: MSA 13.1 Vendor cyber cover raised to USD 5,000,000.
- Section 5: Exhibit A table replaced from July 1, 2023: platform fee USD 38,750/month; claims USD 1.40 (min 6,000/month); statements USD 0.58; denials 3.5%; credentialing USD 275/provider/month; NEW prior authorization support USD 4,200/month. Fees fixed through January 31, 2026.
- Section 6: amendment prevails over the MSA on conflict; otherwise MSA unchanged.

### 2.4 Cyber policy (doc 08) — Halvard Mutual Policy No. HMC-CY-2024-08817, Form HMC-CY 100 NY (01/24)
- Named Insured Larkspur Health Partners, Inc.; Policy Period October 1, 2024 to October 1, 2025 (12:01 a.m. at Named Insured's address). Premium USD 187,450 (Endorsement No. 3 return premium USD 6,250). Retroactive Date and Continuity Date October 1, 2021.
- Declarations Item 3: Policy Aggregate Limit USD 5,000,000. A Incident Response USD 5,000,000, sublimit Notification and Credit Monitoring Costs USD 750,000; B Network Security and Privacy Liability USD 5,000,000; C Regulatory Proceedings USD 5,000,000, sublimit Regulatory Fines and Penalties USD 1,000,000; D Business Interruption and Data Restoration USD 5,000,000; E Dependent Business Incident USD 1,500,000 (Declarations figure; reduced to USD 500,000 by Endorsement No. 3); F Cyber Extortion USD 1,000,000. All sublimits are part of the aggregate.
- Item 4: Retention USD 100,000 each Incident or Claim (one Retention for Related Incidents); Waiting Period 12 hours (D and E).
- Section I Insuring Agreements A–F. A covers Incident Response Costs from a Security Failure or Privacy Event; "Privacy Event" includes Personal Information in the care, custody or control of a Service Provider. E covers Dependent Business Loss from a Dependent Business Incident (interruption of a Service Provider's Computer System caused by a Security Failure).
- Section II Definitions incl. "Control Group" (CEO, CFO, General Counsel, CISO, Privacy Officer, Risk Manager), "Insured's Computer System" (excludes any Service Provider's Computer System), "Service Provider".
- Section III Exclusions: A bodily injury/property damage; B contractual liability (exceptions: liability absent contract; contractual duty to protect Personal Information; Incident Response Costs; PCI assessments); C prior known circumstances (Control Group knowledge before the Continuity Date); D failure to maintain security (critical patch not applied within 45 days to the Insured's Computer System, known to the Control Group) — "applies solely to the Insured's Computer System and shall not apply to any Service Provider's Computer System"; E war/infrastructure; F intentional acts; G antitrust; H securities; I patent/trade secret; J unsolicited communications; K governmental seizure; L contractual penalties/service credits.
- Section V.A Notice: condition precedent; written notice as soon as practicable and in no event later than sixty (60) days after the Insured's Chief Information Security Officer or General Counsel first becomes aware of the Claim or Incident. Endorsement No. 1 adds: failure to give timely notice shall not invalidate coverage unless it prejudiced the Insurer (New York Insurance Law § 3420).
- Section V.C Consent: Incident Response Costs require the Insurer's prior written consent (not unreasonably withheld), except costs incurred within thirty (30) days after the Incident is first Discovered.
- Section VII.A Subrogation: Insured must not waive rights of recovery after an Incident; pre-Incident written contractual waivers/limitations are permitted.
- Other policy facts: renewal of HMC-CY-2023-07264; Declarations countersigned September 24, 2024 by Martin L. Ashcombe (Authorized Representative); policy signed by Eleanor V. Whitfield (President) and Raymond J. Ostergaard (Secretary); Application includes ransomware supplemental questionnaire dated August 30, 2024. "Discovered" (II.15) = first time a Control Group member becomes aware. "Credit Monitoring Costs" (II.7) = up to 24 months or longer if required by law/regulator. "Incident Response Costs" (II.19) must be incurred within 12 months after Discovery and include breach counsel, forensics, Notification Costs, Credit Monitoring Costs, PR, and costs to determine indemnification rights against a Service Provider. "Notification Costs" (II.23) include the call center. Section IV.B: the USD 750,000 sublimit covers Notification Costs and Credit Monitoring Costs combined. Section IV.E: Incident Response Costs are allocated to Insuring Agreement A whether the Incident occurred on the Insured's or a Service Provider's Computer System. Section V.A: a hotline call is not notice. Section V.C: Insurer may limit counsel/forensic reimbursement to panel rates. Exclusion III.D uses a 45-day window (the MSA's Exhibit C uses 14 days — different instruments, not a conflict).
- Endorsement No. 1 — New York Amendatory Endorsement, Form HMC-CY 501 NY (01/24) (notice prejudice per Insurance Law § 3420; 15-day cancellation notice for non-payment; suit limitation per NY law).
- Endorsement No. 2 — Sanctions Limitation and Cyber Extortion Payment Conditions Endorsement, Form HMC-CY 214 (06/23): sanctions; conditions to extortion payments; Insuring Agreement F does not apply to threats against a Service Provider. Nothing about the dependent business sublimit.
- Endorsement No. 3 — Dependent Business Incident Sublimit Amendment, Form HMC-CY 330 (03/24): Item 3(E) replaced with USD 500,000; effective October 1, 2024; issued November 14, 2024; return premium USD 6,250; issued at the Named Insured's request after review of its Service Provider schedule of October 22, 2024; applies only to Dependent Business Loss under Insuring Agreement E; "Nothing in this endorsement reduces or otherwise affects" limits or sublimits for A, B, C, D or F, "including where the Loss ... arises from an Incident on a Service Provider's Computer System".

## 3. Chronology (all Eastern time unless stated; DST began 2:00 a.m. March 9, 2025)
| Date | Event | Source |
|---|---|---|
| 01/26/2022, 01/28/2022 | MSA signed (Fontaine; Vandermeer) | 01 |
| 02/01/2022 | MSA and BAA effective | 01, 02 |
| 06/21/2023, 06/27/2023 | First Amendment signed (Kowalczyk; Kettering) | 03 |
| 07/01/2023 | First Amendment effective (48 h notice; Data Breach Cap) | 03 |
| 10/01/2024 | Halvard policy period begins | 08 |
| 11/14/2024 | Endorsement No. 3 issued (effective 10/01/2024) | 08 |
| 12/10/2024 | Appliance manufacturer advisory SA-2024-117 (CVSS 9.8) and fixed firmware published | 07 |
| 12/24/2024 | Exhibit C 14-day critical patch deadline | 01, 07 |
| 02/01/2025 | MSA first Renewal Term begins | 01 |
| 02/17/2025 07:41 UTC (02:41 EST) | Earliest intrusion: exploit of OBS-VPN-EDGE-01 (final report) | 07 |
| 02/17/2025 08:03 UTC (03:03 EST) | Threat actor reads the appliance configuration file holding stored credentials (incl. svc_rcmsync) | 07 |
| 02/19/2025 04:26 UTC (02/18 23:26 EST) | Test authentication with harvested credentials | 07 |
| 02/21/2025 14:22 UTC (09:22 EST) | First interactive VPN session as svc_rcmsync — the preliminary report's "earliest intrusion" | 06, 07 |
| 02/22/2025 03:17 UTC (02/21 22:17 EST) | Credential-dumping tool on APP-RCM-02; domain administrator credentials obtained | 06, 07 |
| 02/24/2025 02:40 UTC (02/23 21:40 EST) | First interactive logon to FS-BILL-03 with domain administrator account | 06, 07 |
| 03/03–03/07/2025 | Staging of archives ARCH-01 to ARCH-08 | 07 |
| 03/05–03/07/2025 | Exfiltration channel 1: 41.0 GB | 06, 07 |
| 03/08/2025 | Exfiltration channel 2: 2.6 GB (found only in the final report) | 07 |
| 03/09/2025 (Sunday) 09:58 UTC / 05:58 EDT | BLACKTARN encryption begins | 06, 07 |
| 03/09/2025 10:14 UTC / 06:14 EDT | Ostrander on-call engineer alerted; ransom notes on FS-BILL-03 and APP-RCM servers = DISCOVERY | 07 |
| 03/09/2025 11:05 EDT | Ostrander isolates hosted environment | 07 |
| 03/10/2025 (Monday) ~4:15 p.m. | Okafor phones Wierzbicki (CISO) — Larkspur's first awareness | 04, 05, 10, 15 |
| 03/10/2025 7:48 p.m. | Wierzbicki email to Marchbanks, Rhee, cc Fontaine | 05 |
| 03/11/2025 06:14 EDT | 48-hour MSA notice deadline (First Amendment) | derived |
| 03/11/2025 | Marchbanks contacts Alice Moreno; VPN appliance patched (and later replaced) | 05, 07 |
| 03/12/2025 06:14 EDT | 72-hour deadline under original 9.4 (superseded) | derived |
| 03/12/2025 | Ostrander says it "determined" Larkspur systems were affected | 04 |
| 03/13/2025 | Engagement letter dated; Ostrander platform restored (~2 p.m.) | 14, 05, 04 |
| 03/14/2025 (Friday) 3:12 p.m. | Ostrander written notice letter received by email | 04, 05 |
| 03/14/2025 6:21 p.m. | Marchbanks email: board meeting 3/20 | 05 |
| 03/17/2025 | Engagement letter countersigned; Cobaltline engaged; hard copy of notice stamped received | 14, 06, 04 |
| 03/20/2025 4:00 p.m. | Special board meeting; Special Committee formed | 10 |
| 03/21/2025 | BAA 10-business-day Breach report deadline; Larkspur's reservation-of-rights letter to Ostrander (referenced only) | derived, 09 |
| 04/04/2025 | Cobaltline preliminary report (DRAFT v0.3) | 06 |
| 04/09/2025 | End of the 30-day window after Discovery (03/10) in which Incident Response Costs need no prior consent (policy V.C) | derived |
| 04/16/2025 | Ostrander confirms in writing that FS-BILL-03 had no encryption at rest (E-10) | 07 |
| 04/17/2025 | Board meeting approves the March 20 minutes as corrected | 10 |
| 04/22/2025 | Ostrander produces firewall egress logs from 14 Feb 2025 | 07 |
| 05/02/2025 | Fontaine approves 24 months of credit monitoring | 13 |
| 05/05/2025 | Patient letter template v4 (draft, tracked changes) | 13 |
| 05/09/2025 | Day 60 after CISO awareness (3/10) — policy notice deadline | derived |
| 05/12/2025 (Monday) | Notice of circumstances to Halvard (day 63 from 3/10; day 59 from 3/14) | 09 |
| 05/13/2025 | Day 60 counted from 3/14 (counsel's framing) | derived |
| 05/27/2025 (Tuesday) | Halvard reservation of rights letter | 15 |
| 05/29/2025 | ROR hard copy stamped received at Calder & Wynne; Marchbanks forwards the internal chain to Arai | 15, 05 |
| 05/30/2025 | Cobaltline final forensic report v1.0 | 07 |
| 06/02/2025 | Privacy Office prints the by-clinic table | 12 |
| 06/13/2025 | Arai indemnity demand to Ostrander (letter + email) | 16 |
| 06/27/2025 | Kettering email disputing the demand | 16 |
| 07/09 or 07/10/2025 | Executive meeting dates proposed by Kettering (Section 18.1) | 16 |
| 07/18/2025 5:00 p.m. | Expiry of Ostrander's without-prejudice offer | 16 |

## 4. Forensic figures
| Item | Preliminary (04/04/2025, doc 06) | Final (05/30/2025, doc 07) — prevails |
|---|---|---|
| Affected individuals | approx. 18,400 (estimate; ARCH-06 to ARCH-08 not parsed; de-duplication incomplete) | 23,912 |
| Earliest intrusion | 21 February 2025 (14:22 UTC) | 17 February 2025 (07:41 UTC) |
| Exfiltrated (total, all Ostrander clients) | approx. 41 GB (41.0 GB) | 43.6 GB (41.0 + 2.6) |
| Larkspur-attributable data | not determined | 17.2 GB in 1,284 files |
| Root cause | working hypothesis: compromised credentials of svc_rcmsync | unpatched VPN appliance OBS-VPN-EDGE-01 (SA-2024-117) |
| Encryption at rest of FS-BILL-03 | "under review" | not enabled — billing exports in plaintext (breach of MSA 9.2) |
| Dwell time | 16 days (02/21–03/09) | 20 days (02/17–03/09) |

Preliminary-report extras: 31,206 Larkspur patient records extracted from ARCH-01 to ARCH-05 (→ ~18,400 after first-pass de-duplication); evidence E-01 to E-09; outstanding requests R-01 to R-05; EDR telemetry for 22 of 61 servers; VPN logs only from 20/02/2025 (rotation); Larkspur SFTP account disabled 13/03/2025.

Other final-report facts: new evidence E-10 (16/04/2025 encryption confirmation), E-11 (22/04/2025 firewall logs), E-12 and E-13 (24/04/2025 patch records, forwarded VPN logs), E-14 (29/04/2025 EDR), E-15 (25/04/2025 retention schedule), E-16 (12/05/2025 MPI extract), E-17 (19/05/2025 Ironvale memo); ARCH-06 to ARCH-08 parsed 02/05/2025; MPI reconciliation 21/05/2025; change record CHG-24-1187 raised 16/12/2024 for the "Q1 2025 maintenance window"; FS-BILL-03 commissioned August 2023, never encrypted; EDR on 47 of 61 servers (FS-BILL-03 among the 14 without); Larkspur files 612 claim (9.8 GB) + 438 statement (5.1 GB) + 234 eligibility (2.3 GB) = 1,284 files / 17.2 GB, dated 02/10/2023 to 08/03/2025; payment data 1,038 card (type, last 4, expiry) + 148 bank account = 1,186; Ironvale memorandum of 12/03/2025 is the source of Ostrander's "March 12 determination"; leak-site monitoring from 18/03/2025, nothing found as at 28/05/2025; databases DB-RCM-01/02 encrypted (transparent database encryption); FS-BILL-03 held exports from October 2023 to March 2025 despite the 30-day retention rule (Exhibit C 2.9); patch available 69 days before the exploit, 55 days past the 14-day deadline; patch applied March 11, 2025 (91 days after the advisory). Exfiltration destinations 203.0.113.47 and 198.51.100.23. No evidence of publication on a leak site as of report date.

### Affected individuals by clinic (docs 07 Appendix B, 11, 12)
| Code | Total | Name/address | DOB | SSN | Health insurance ID | Dx/procedure codes | Payment card/bank |
|---|---|---|---|---|---|---|---|
| LHP-01 Flatbush | 4,318 | 4,318 | 4,301 | 1,942 | 3,876 | 3,512 | 214 |
| LHP-02 Bay Ridge | 3,207 | 3,207 | 3,198 | 1,410 | 2,915 | 2,640 | 167 |
| LHP-03 Williamsburg | 2,954 | 2,954 | 2,940 | 1,288 | 2,611 | 2,398 | 141 |
| LHP-04 Astoria | 3,671 | 3,671 | 3,660 | 1,733 | 3,302 | 2,987 | 188 |
| LHP-05 Jamaica | 2,489 | 2,489 | 2,481 | 1,105 | 2,204 | 2,011 | 122 |
| LHP-06 Fordham | 3,102 | 3,102 | 3,090 | 1,467 | 2,833 | 2,560 | 156 |
| LHP-07 St. George | 1,876 | 1,876 | 1,870 | 802 | 1,690 | 1,522 | 93 |
| LHP-08 Yonkers | 2,295 | 2,295 | 2,288 | 1,016 | 2,047 | 1,860 | 105 |
| TOTAL | 23,912 | 23,912 | 23,828 | 10,763 | 21,478 | 19,490 | 1,186 |
Column totals verified: 4,318+3,207+2,954+3,671+2,489+3,102+1,876+2,295 = 23,912.

## 5. Arithmetic
- MSA notice (48 h, First Amendment): discovery 03/09/2025 06:14 EDT + 48 h = 03/11/2025 06:14 EDT. Written notice 03/14/2025 3:12 p.m. = about 129 hours (5 days 8 h 58 min) after discovery, about 81 hours late.
- Original 72 h (superseded): deadline 03/12/2025 06:14 EDT.
- Ostrander's own theory: 48 h from its March 12 "determination" → March 14 (timely in its view).
- BAA 2.4: 10 business days after Sunday 03/09/2025 = Mar 10, 11, 12, 13, 14, 17, 18, 19, 20, 21 → 03/21/2025.
- MSA Section 20(c): email sent 3:12 p.m. on a Business Day with same-day courier copy → effective 03/14/2025 (hard copy stamped 03/17/2025 is irrelevant to the effective date).
- Policy notice: 60 days after 03/10/2025 = 05/09/2025 (Friday). Notice 05/12/2025 = day 63 (3 days late). Counting from 03/14/2025: day 60 = 05/13/2025, so 05/12 would be day 59.
- Policy V.C consent window: Discovery 03/10/2025 + 30 days = 04/09/2025; costs incurred after that and before consent are the subject of the ROR's paragraph 7.
- Section 18.1 window counted from 06/27/2025: 15 Business Days (July 4 is a bank holiday) = 07/21/2025; the proposed July 9/10 dates fall inside it.
- Patch: advisory 12/10/2024 + 14 days = 12/24/2024; exploit 02/17/2025 = 69 days after advisory, 55 days after the deadline; patched 03/11/2025 = 91 days after advisory.
- Dwell: 02/17 → 03/09 = 20 days; preliminary 02/21 → 03/09 = 16 days.
- Exfiltration: 41.0 + 2.6 = 43.6 GB.
- Engagement budget (doc 14), hours Moreno 60+25+15 = 100, Arai 140+70+45 = 255, Haddad 50+20+10 = 80: Phase 1 Moreno 60 h x 1,150 = 69,000; Arai 140 h x 785 = 109,900; Haddad 50 h x 340 = 17,000 → 195,900. Phase 2: 25 x 1,150 = 28,750; 70 x 785 = 54,950; 20 x 340 = 6,800 → 90,500. Phase 3: 15 x 1,150 = 17,250; 45 x 785 = 35,325; 10 x 340 = 3,400 → 55,975. Total at standard rates 342,375; 10% discount 34,237.50 → 308,137.50; plus estimated disbursements 7,500 → 315,637.50.
- Credit monitoring (Vantorra, USD 1.50/individual/month): 23,912 x 1.50 x 24 = 860,832.00; 12 months = 430,416.00. At the preliminary 18,400: 24 months = 662,400; 12 months = 331,200 (template comment).
- Mailing: 23,912 x 1.35 = 32,281.20. Call center: 41,300.00 (stated as a single figure).
- Cobaltline (214,750.00) and Calder & Wynne fees through 05/31/2025 (168,420.00) are stated only as totals in the June 13 demand (doc 16); no document breaks them down.
- Indemnity demand 06/13/2025: 214,750.00 + 168,420.00 + 32,281.20 + 41,300.00 + 860,832.00 = 1,317,583.20 (below the USD 2,000,000 Data Breach Cap; headroom 682,416.80).
- Ostrander's general-cap figure (doc 16) — invoices March 2024–February 2025: 61,204.50; 59,870.25; 62,415.80; 60,988.10; 63,107.45; 61,552.90; 60,214.35; 64,031.70; 62,876.15; 65,340.60; 63,918.25; 62,709.95 = 748,230.00.
- Ostrander's without-prejudice offer: 32,281.20 + 430,416.00 = 462,697.20.
- Derived only (no document states it): notification-type costs under the policy's USD 750,000 sublimit = 32,281.20 + 41,300.00 + 860,832.00 = 934,413.20, i.e. 184,413.20 above that sublimit.

## 6. What each document says
- 01 MSA (executed): Sections 1–22 and Exhibits A–C as in 2.1; 72-hour notice in the original 9.4; general cap only.
- 02 BAA (executed): as in 2.2; 10 business days for Breach reports; 7.4 more-stringent-obligations proviso.
- 03 First Amendment: 48 hours; Section 12.4 Data Breach Cap USD 2,000,000; new fees; Vendor insurance USD 5,000,000.
- 04 Ostrander notice letter dated March 14, 2025 (Kettering): discovery of ransomware "on Sunday, March 9, 2025"; claims it "determined" on March 12 that Larkspur systems were affected and that notice within 48 hours of that determination is timely under MSA 9.4 and BAA 2.4; refers to Okafor's March 10 call to Wierzbicki; "no evidence that Larkspur data was exfiltrated"; services restored March 13; Ironvale engaged; federal law enforcement notified; stamped received March 17, 2025 (hard copy).
- 05 Internal emails (forwarded by Marchbanks to Arai, cc Moreno and Haddad, on May 29, 2025 9:07 AM): Wierzbicki's March 10 7:48 PM email reporting the ~4:15 p.m. call on his cell; Rhee's March 11 8:05 AM deadline notes (48 h; BAA 3/21; BAA 7.4; independent contractor); Marchbanks March 11 9:32 AM (spoke with Moreno; will handle carrier notice once Ostrander writes; asks if broker is still Pinecrest); Fontaine March 12 6:40 PM (about 6,200 claims queued); Wierzbicki March 13 7:36 PM (platform back ~2 p.m.; Ironvale named; SFTP account disabled, 41 users re-enrolled in MFA); Rhee March 14 4:58 PM (letter arrived 3:12 pm; "about 14,000 patients" — wrong; file PRIV-2025-007); Marchbanks March 14 6:21 PM (no numbers in email; "Calder Wynne" engaged by letter dated yesterday; board 3/20 4 pm; will speak to Pinecrest next week). Forward note admits the 60 days were counted from Ostrander's letter and that 14,000 was a guess (prelim 18,400; final due "tomorrow").
- 06 Cobaltline preliminary (DRAFT, privileged), 4 April 2025: 18,400; 21 Feb 2025; ~41 GB; credential hypothesis; limitations.
- 07 Cobaltline final, 30 May 2025: supersedes preliminary; 23,912; 17 Feb 2025; 43.6 GB; root cause; FS-BILL-03 unencrypted; Appendix B clinic table.
- 08 Halvard policy: as in 2.4.
- 09 Notice of circumstances, May 12, 2025 (Arai): says Larkspur "first became aware" on March 14, 2025 on receipt of Ostrander's letter and that notice is within Section V.A; encloses the Ostrander letter (which mentions the March 10 call); preliminary 18,400 and February 21; seeks A, B, C, E; asks consent under V.C for Calder & Wynne, Cobaltline, Vantorra, Brightwater, Harborline; mentions Larkspur's March 21, 2025 reservation-of-rights letter to Ostrander.
- 10 Board minutes March 20, 2025 (signed Nolan): typed attendance shows Pomeroy present, Ibsen absent; handwritten correction: Pomeroy absent (excused), Ibsen present by telephone from 4:20 p.m.; quorum present; resolutions: Special Committee (Varadarajan chair, Abernathy, Nakamura) with authority up to USD 350,000; ratify Calder & Wynne engagement letter of March 13, 2025 and Cobaltline retention through counsel (March 17); litigation hold; GC to notify cyber insurer "as soon as practicable". Vote at about 4:52 p.m.; 6 directors present and voting in favor.
- 11/12 Affected individuals by clinic: the table in section 4 (CSV not importable; PDF print).
- 13 Patient letter template v4 (draft, Rhee, 05/05/2025, signed for Fontaine): credit monitoring {12 → 24} months (tracked change); comment that Fontaine approved 24 months on 5/2 and Vantorra charges $1.50 per person per month (about $662,400 for 24 months vs $331,200 for 12 at ~18,400); intrusion window "between February 21, 2025 and March 9, 2025" with a comment that it comes from the preliminary report and must not be mailed until the final report; "a third-party vendor" replaced by Ostrander's name; "Larkspur's own systems were not affected" replaced; enrollment deadline "July 31, 2025" replaced by "90 days after mailing"; identity theft insurance up to $1,000,000; call line (855) 555-0137, 9 a.m.–9 p.m. ET; regulator-notification sentence inserted as a tracked insertion (intention only).
- 14 Engagement letter March 13, 2025 (Moreno; countersigned Marchbanks March 17): incident reported to Larkspur "on or about March 10, 2025"; work began on oral instruction of March 11, 2025; three phases; rates Moreno 1,150, Arai 785, Haddad 340 (other partners 995–1,250, associates 560–825, specialists 295); 10% discount; budget 315,637.50; no retainer; monthly invoices net 30; vendor charges over USD 5,000 need approval; insurer-notice warning (paragraph 9); Halvard Re Ltd. 2023 conflict disclosure and consent; Cobaltline retained through the firm; file kept 7 years.
- 15 Halvard ROR May 27, 2025 (Venkataraman; stamped received May 29, 2025): late notice (63 days after March 10) subject to Endorsement No. 1 / § 3420; dependent business sublimit USD 500,000 "pursuant to Endorsement No. 2" applied to all Loss; Exclusion III.D for the unpatched VPN; III.B; III.C; V.C pre-consent costs; sublimits 750,000 / 1,000,000; retention 100,000; subrogation; document requests.
- 16 Kettering email June 27, 2025 4:42 PM (.eml, not importable; FRE 408 / CPLR 4547 header): disputes demand; says "the MSA says 12 months" of credit monitoring; says notice timely (phone call March 10; March 12 determination; BAA controls via MSA 22.3); FS-BILL-03 was "staging"; misquotes Section 12.4 as covering only "breach of Section 9"; general cap 748,230.00; 24 months unreasonable; credit monitoring consequential; offers 462,697.20 without prejudice; asks for Section 18.1 executive meeting. Quotes Arai's June 13 demand (1,317,583.20).

## 7. Traps (deliberate inconsistencies) and what prevails
1. Notice period 72 h (MSA 9.4, doc 01) vs 48 h (First Amendment, doc 03). The First Amendment prevails (effective July 1, 2023; MSA 22.3(ii); Amendment Section 6).
2. Discovery date: Ostrander's letter (doc 04) measures from a March 12 "determination"; MSA 1.10 / BAA 1.2(d) define discovery as first known or reasonably knowable, and the final report (doc 07) shows ransom notes on Larkspur-processing servers at 06:14 EDT on March 9, 2025. Discovery = March 9, 2025; notice was late (deadline March 11, 2025 06:14 EDT).
3. "BAA controls" (docs 04, 16): MSA 22.3 gives the BAA priority only on conflict regarding PHI, and BAA 7.4 says shorter MSA notice periods apply in addition. The 48-hour MSA requirement still applies; the BAA's 10-business-day deadline (March 21, 2025) is a separate floor.
4. Telephone notice: Ostrander relies on the March 10 call (doc 16). MSA 9.4 (as amended) says telephone contact is not notice; written notice under Section 20 is required. Written notice effective March 14, 2025 (email before 5 p.m. with courier copy); the March 17 RECEIVED stamp on doc 04 is the hard copy only.
5. Affected individuals approx. 18,400 (docs 06, 09, 15, the doc 13 comment and the doc 05 forward) vs 23,912 (docs 07, 11, 12, 16). Final prevails ("supersedes ... in its entirety").
6. "About 14,000 patients" (Rhee, March 14 email, doc 05) is wrong; 23,912 prevails. Marchbanks' May 29 forward says so.
7. Earliest intrusion February 21, 2025 (doc 06; also doc 09 and the template doc 13) vs February 17, 2025 (doc 07). Final prevails; the February 21 date is the first interactive VPN session, not the first intrusion.
8. Exfiltration ~41 GB (doc 06) vs 43.6 GB (doc 07); "no evidence of exfiltration" (doc 04). Final prevails.
9. Root cause: credential compromise hypothesis (doc 06) vs unpatched VPN appliance (doc 07). Final prevails.
10. UK dates in Cobaltline tables ("09/03/2025" = 9 March 2025, not September 3; "17/02/2025"; "22/04/2025").
11. Larkspur's awareness for the policy notice condition: doc 09 says March 14, 2025 (letter receipt); docs 05, 10, 04 (enclosed with doc 09 itself) and 15 show the CISO knew on March 10, 2025. March 10 prevails as the fact; notice on May 12 was day 63, three days after the May 9 deadline. Effect is softened by Endorsement No. 1 (prejudice requirement) — a coverage argument, not a change of fact.
12. Dependent business sublimit: Declarations USD 1,500,000 vs Endorsement No. 3 USD 500,000 — the endorsement prevails.
13. The ROR (doc 15) cites "Endorsement No. 2" for the USD 500,000 sublimit; the correct endorsement is No. 3 (No. 2 is the Sanctions Limitation and Cyber Extortion Payment Conditions Endorsement).
14. The ROR applies the USD 500,000 sublimit to "all Loss arising from the Ostrander incident"; Endorsement No. 3 limits it to Insuring Agreement E and says it does not affect A, B, C, D or F, and Section IV.E allocates Incident Response Costs to A even when the Incident is on a Service Provider's system. Policy text prevails over the ROR's characterization.
15. The ROR invokes Exclusion III.D (failure to maintain security) for Ostrander's unpatched VPN and paraphrases it without its "Insured's Computer System" limitation; III.D applies solely to the Insured's Computer System, not to a Service Provider's, and also requires Control Group knowledge. Policy text prevails.
16. Cap: Kettering (doc 16) says the general cap of Section 12.2 applies (USD 748,230.00) and quotes 12.4 as covering only "breach of Section 9"; First Amendment 12.4 covers breach of Section 9 or the BAA and indemnity under 11.1(c), in lieu of 12.2, with Data Breach Costs as direct damages. The First Amendment text prevails: the applicable cap is USD 2,000,000.
17. "Staging server" argument (doc 16) vs MSA 9.2 text (expressly covers file servers, exports and staging environments). MSA text prevails; the final report finds FS-BILL-03 unencrypted.
18. Credit monitoring 12 months (Kettering; the deleted text in template v4) vs 24 months (template v4 tracked change; demand). MSA 11.1(c)(ii) permits a longer period reasonably determined by Client; the 24-month figure is Larkspur's current position (template v4, demand), the dispute is open.
19. Board attendance: typed list (Pomeroy present, Ibsen absent) vs handwritten correction (Pomeroy absent, Ibsen by phone from 4:20 p.m.). The signed correction prevails: present = Lindqvist, Fontaine, Varadarajan, Abernathy, Nakamura, Ibsen (6 of 7).
20. Name variants of Ostrander and of Larkspur's counsel ("Calder Wynne" in one email) — same entities.
21. Insurance figure distractor: the Declarations say "Notification and Credit Monitoring Costs USD 750,000" — a sublimit within A, not a separate limit; the USD 5,000,000 aggregate is the overall maximum.
22. Kettering says "the MSA says 12 months" of credit monitoring; MSA 11.1(c)(ii) says not less than 12 months or a longer period required by law, demanded by a regulator or reasonably determined by Client. MSA text prevails.
23. Time zones: the Cobaltline reports are in UTC; some events fall on a different calendar day in Eastern time (e.g. 22/02/2025 03:17 UTC = 21/02/2025 22:17 EST). The intrusion date of 17 February 2025 is the same in both (07:41 UTC = 02:41 EST).

## 8. Not stated (no document states these)
1. Whether Ostrander paid a ransom, or the amount of any ransom demanded.
2. The actual date patient notification letters were mailed, and how many were mailed in any wave.
3. Whether, or when, Larkspur actually notified HHS OCR, the New York Attorney General, the Department of State or the State Police, or the media (the draft template only says "We are also notifying", as a tracked insertion); and whether any regulator opened an investigation or imposed a fine.
4. The number or percentage of affected individuals who are New York residents (or residents of any other state).
5. The name of Ostrander's own cyber insurer, and whether Ostrander actually carries the USD 5,000,000 cover required by the First Amendment.
6. Halvard's final coverage decision, any payment by Halvard, or whether Halvard consented to Calder & Wynne, Cobaltline or the other providers.
7. Whether Larkspur terminated, or gave notice of non-renewal of, the MSA.
8. Larkspur's own verified total of Fees paid to Ostrander in the 12 months before March 9, 2025 (only Ostrander's table exists).
9. Whether the Section 18.1 executive meeting took place or the dispute settled.
10. The credit monitoring enrollment rate.
11. Larkspur's actual business interruption loss (Insuring Agreement E amount) — only "about 6,200 claims queued" on March 12.
12. A CVE identifier for the VPN vulnerability (only manufacturer advisory SA-2024-117) and the appliance manufacturer's name.
13. Whether the Special Committee met after March 20, 2025 or what it decided.
14. The text of Larkspur's March 21, 2025 reservation-of-rights letter to Ostrander and of the June 13, 2025 demand letter itself (only the cover email is quoted).
15. Any breakdown of the Cobaltline (USD 214,750.00), Calder & Wynne (USD 168,420.00) or Harborline (USD 41,300.00) amounts into invoices, months or rates.
