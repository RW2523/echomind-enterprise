> FICTIONAL DEMO DOCUMENT. All parties, people, amounts and events are invented. Not legal advice.

# QUARTERLY SERVICE LEVEL REPORT — Q3 2026

Prepared by Harrowgate Systems Inc. ("Provider") for Voss-Lindqvist AB ("Client") under paragraph 5 of Schedule 2 to the Master Services Agreement dated March 2, 2026 (reference HSI-MSA-2026-0147).

Reporting period: July 1, 2026 to September 30, 2026

Issued: October 7, 2026

Prepared by: Priya Raman, Service Delivery Manager, Harrowgate Systems Inc.

Distribution: Petra Voss, General Counsel, Voss-Lindqvist AB; Jonas Ekberg, Head of Platform Engineering, Voss-Lindqvist AB; Thomas Reyes, Vice President, Commercial Accounts, Harrowgate Systems Inc.

## 1. Summary

1.1 The Availability Target under the Agreement is 99.5% in each calendar month.

1.2 Availability met the Availability Target in July 2026 and September 2026. Availability in August 2026 was 98.71%, which is below the Availability Target, because of two incidents described in section 3.

1.3 No Personal Data Breach affecting Client Data occurred during the reporting period.

## 2. Monthly Availability

Availability is calculated under paragraph 2 of Schedule 2: ((Total Minutes − Downtime) ÷ Total Minutes) × 100, rounded to two decimal places. Scheduled Maintenance is excluded from Downtime.

| Month | Total Minutes | Scheduled Maintenance (minutes, excluded) | Downtime (minutes) | Availability | Availability Target met? |
|---|---|---|---|---|---|
| July 2026 | 44,640 | 240 | 80 | 99.82% | Yes |
| August 2026 | 44,640 | 180 | 576 | 98.71% | No |
| September 2026 | 43,200 | 240 | 155 | 99.64% | Yes |

For information only, Availability across the whole quarter was 99.39% (811 minutes of Downtime in 132,480 minutes). The Agreement measures Availability by calendar month, not by quarter.

## 3. Incidents Counted as Downtime

| Reference | Date | Window (UTC) | Duration (minutes) | Cause |
|---|---|---|---|---|
| INC-2026-0719 | July 19, 2026 | 03:10 to 04:30 | 80 | Certificate rotation failure on the API gateway |
| INC-2026-0812 | August 12, 2026 | 14:06 to 20:58 | 412 | Storage cluster failover did not complete in the primary EU region |
| INC-2026-0827 | August 27, 2026 | 09:31 to 12:15 | 164 | Faulty network configuration change in the event-streaming tier |
| INC-2026-0904 | September 4, 2026 | 10:02 to 11:12 | 70 | Database connection-pool exhaustion |
| INC-2026-0922 | September 22, 2026 | 16:45 to 18:10 | 85 | Degraded performance after a dependency upgrade |

3.1 INC-2026-0812 was a Severity 1 incident. Provider's root-cause analysis was sent to Client on August 19, 2026. The storage failover procedure has been rebuilt and was re-tested successfully on September 15, 2026.

3.2 INC-2026-0827 was a Severity 1 incident. Provider has added an automated pre-deployment check for network configuration changes.

## 4. Support

| Severity | Tickets opened | Response time met |
|---|---|---|
| Severity 1 | 2 | 2 of 2 |
| Severity 2 | 6 | 6 of 6 |
| Severity 3 | 19 | 18 of 19 |
| Severity 4 | 31 | 31 of 31 |

## 5. Security and Data Protection

5.1 Provider's ISO/IEC 27001 certification remained current throughout the reporting period.

5.2 No new sub-processor was appointed during the reporting period.

## 6. Service Credits

6.1 A Service Credit applies for August 2026, the only month in the reporting period in which Availability was below the Availability Target. The Service Credit is calculated under paragraph 3 of Schedule 2 and will be credited against the next invoice issued after this report, as paragraph 5 of Schedule 2 provides.

6.2 No Service Credit applies for July 2026 or September 2026.

## 7. Scheduled Maintenance for Q4 2026

7.1 Provider plans Scheduled Maintenance on October 17, 2026, November 14, 2026 and December 12, 2026, each between 22:00 and 02:00 Central European Time.

Priya Raman
Service Delivery Manager, Harrowgate Systems Inc.
October 7, 2026
