# EchoMind Law research evaluation — heldout9

- Commit: `3eaaa9a+changes`
- Answer model: `nvidia/Qwen3-30B-A3B-FP4`
- Embeddings: nomic-embed-text; reranker: ms-marco-MiniLM-L-6-v2
- Run: 20261104-141106

| Measure | Result |
|---|---|
| items | 96 |
| answer accuracy | 26/54 |
| abstention accuracy | 24/42 |
| false abstentions | 22 |
| t06 leaks | 0 |
| trap values in answers | 5 |
| citation precision | 0.951 |
| quote verification rate | 0.986 |
| t05 conflicts found | 0/0 |
| r7 multi source | 3/3 |
| statuses | {'partial': 37, 'abstained': 51, 'answered': 7, 'unverified': 1} |
| latency total p50 s | 5.64 |
| latency total p95 s | 11.81 |
| latency first paragraph p50 s | 5.64 |
| latency first paragraph p95 s | 11.81 |
| runs | 3 |
| spread | {'answer_accuracy': [0.444, 0.556], 'abstention_accuracy': [0.5, 0.643], 'citation_precision': [0.902, 1.0], 'quote_verification_rate': [0.957, 1.0], 't06_leaks': [0, 0]} |
| never correct | ['h9-si-okafor-rate', 'h9-va-sept-venue', 'h9-va-side-end', 'h9-va-schedule1', 'h9-rp-nda-purpose-only', 'h9-rp-lease-fm-delay', 'h9-rp-msa-conf-use', 'h9-xdoc-lease-courts', 'h9-list-board-decisions', 'h9-list-msa-schedules', 'h9-list-landlord-obligations'] |

| Run | answer accuracy | abstention accuracy | citation precision | quote verification rate | t06 leaks |
|---|---|---|---|---|---|
| 1 | 10/18 | 8/14 | 0.902 | 1.0 | 0 |
| 2 | 8/18 | 9/14 | 1.0 | 1.0 | 0 |
| 3 | 8/18 | 7/14 | 0.95 | 0.957 | 0 |

Changed between runs (correct/runs): h9-si-breach-undue-delay 1/3, h9-si-coc-advance 1/3, h9-va-completion-actual 2/3, h9-va-holm-relative 1/3, h9-rp-nda-advisers-strict 2/3, h9-rp-cw-complaints-email 1/3, h9-xdoc-kestrel-coc 1/3, h9-xdoc-nda-breach-hours 2/3

| Item | Run | Expected | Status | Correct | Precision | Quotes verified | Seconds |
|---|---|---|---|---|---|---|---|
| h9-si-nda-carveout-count | 1 | answer | partial | yes | 1.0 | 2/2 | 11.62 |
| h9-si-okafor-rate | 1 | answer | abstained | **no** | — | 0/0 | 6.24 |
| h9-si-voss-attendance | 1 | answer | partial | yes | 1.0 | 1/1 | 5.26 |
| h9-si-breach-undue-delay | 1 | answer | partial | yes | 1.0 | 1/1 | 7.91 |
| h9-si-coc-advance | 1 | answer | partial | yes | 0.6666666666666666 | 3/3 | 17.34 |
| h9-si-lease-quarter-days | 1 | answer | partial | yes | 1.0 | 1/1 | 5.23 |
| h9-si-amend-invoicing | 1 | answer | partial | yes | 1.0 | 2/2 | 9.92 |
| h9-si-side-agreed | 1 | answer | partial | yes | 0.25 | 8/8 | 16.07 |
| h9-va-meeting-end | 1 | abstain | abstained | yes | — | 0/0 | 7.56 |
| h9-va-sept-venue | 1 | abstain | partial | **no** | — | 2/2 | 10.84 |
| h9-va-side-end | 1 | abstain | partial | **no** | — | 1/1 | 5.64 |
| h9-va-completion-actual | 1 | abstain | abstained | yes | — | 0/0 | 5.62 |
| h9-va-ekberg-since | 1 | abstain | abstained | yes | — | 0/0 | 3.21 |
| h9-va-holm-relative | 1 | abstain | partial | **no** | — | 1/1 | 5.1 |
| h9-va-schedule1 | 1 | abstain | partial | **no** | — | 2/2 | 9.47 |
| h9-va-estimate-hours | 1 | abstain | abstained | yes | — | 0/0 | 6.22 |
| h9-rp-nda-purpose-only | 1 | answer | abstained | **no** | — | 0/0 | 10.17 |
| h9-rp-nda-advisers-strict | 1 | answer | partial | yes | 1.0 | 1/1 | 9.58 |
| h9-rp-lease-assign-reasonable | 1 | answer | partial | yes | 1.0 | 1/1 | 7.37 |
| h9-rp-lease-fm-delay | 1 | answer | abstained | **no** | — | 0/0 | 11.81 |
| h9-rp-cw-complaints-email | 1 | answer | partial | **no** | 1.0 | 1/1 | 6.77 |
| h9-rp-msa-conf-use | 1 | answer | abstained | **no** | — | 0/0 | 5.46 |
| h9-xdoc-kestrel-return | 1 | abstain | abstained | yes | — | 0/0 | 7.82 |
| h9-xdoc-lease-courts | 1 | abstain | abstained | **no** | — | 0/0 | 5.41 |
| h9-xdoc-kestrel-coc | 1 | abstain | abstained | **no** | — | 0/0 | 4.96 |
| h9-xdoc-nda-breach-hours | 1 | abstain | abstained | yes | — | 0/0 | 5.77 |
| h9-list-board-decisions | 1 | answer | abstained | **no** | — | 0/0 | 2.62 |
| h9-list-msa-schedules | 1 | answer | abstained | **no** | — | 0/0 | 3.74 |
| h9-list-landlord-obligations | 1 | answer | abstained | **no** | — | 0/0 | 7.32 |
| h9-list-kestrel-docs | 1 | answer | answered | yes | 1.0 | 2/2 | 11.82 |
| h9-t06-instalment-investigation | 1 | abstain | abstained | yes | — | 0/0 | 5.43 |
| h9-t06-june-merger | 1 | abstain | abstained | yes | — | 0/0 | 4.22 |
| h9-si-nda-carveout-count | 2 | answer | answered | yes | 1.0 | 1/1 | 7.77 |
| h9-si-okafor-rate | 2 | answer | abstained | **no** | — | 0/0 | 7.75 |
| h9-si-voss-attendance | 2 | answer | partial | yes | 1.0 | 2/2 | 7.53 |
| h9-si-breach-undue-delay | 2 | answer | partial | **no** | 1.0 | 1/1 | 7.36 |
| h9-si-coc-advance | 2 | answer | abstained | **no** | — | 0/0 | 13.49 |
| h9-si-lease-quarter-days | 2 | answer | partial | yes | 1.0 | 1/1 | 4.51 |
| h9-si-amend-invoicing | 2 | answer | partial | yes | 1.0 | 2/2 | 8.59 |
| h9-si-side-agreed | 2 | answer | partial | yes | 1.0 | 1/1 | 4.55 |
| h9-va-meeting-end | 2 | abstain | abstained | yes | — | 0/0 | 5.31 |
| h9-va-sept-venue | 2 | abstain | answered | **no** | — | 1/1 | 3.82 |
| h9-va-side-end | 2 | abstain | partial | **no** | — | 3/3 | 6.83 |
| h9-va-completion-actual | 2 | abstain | abstained | yes | — | 0/0 | 3.11 |
| h9-va-ekberg-since | 2 | abstain | abstained | yes | — | 0/0 | 2.05 |
| h9-va-holm-relative | 2 | abstain | abstained | yes | — | 0/0 | 3.52 |
| h9-va-schedule1 | 2 | abstain | partial | **no** | — | 1/1 | 4.41 |
| h9-va-estimate-hours | 2 | abstain | abstained | yes | — | 0/0 | 4.27 |
| h9-rp-nda-purpose-only | 2 | answer | abstained | **no** | — | 0/0 | 7.97 |
| h9-rp-nda-advisers-strict | 2 | answer | partial | yes | 1.0 | 1/1 | 11.8 |
| h9-rp-lease-assign-reasonable | 2 | answer | partial | yes | 1.0 | 3/3 | 11.51 |
| h9-rp-lease-fm-delay | 2 | answer | abstained | **no** | — | 0/0 | 7.03 |
| h9-rp-cw-complaints-email | 2 | answer | answered | **no** | 1.0 | 1/1 | 3.34 |
| h9-rp-msa-conf-use | 2 | answer | abstained | **no** | — | 0/0 | 7.21 |
| h9-xdoc-kestrel-return | 2 | abstain | abstained | yes | — | 0/0 | 8.52 |
| h9-xdoc-lease-courts | 2 | abstain | abstained | **no** | — | 0/0 | 6.22 |
| h9-xdoc-kestrel-coc | 2 | abstain | abstained | **no** | — | 0/0 | 5.35 |
| h9-xdoc-nda-breach-hours | 2 | abstain | abstained | yes | — | 0/0 | 2.77 |
| h9-list-board-decisions | 2 | answer | abstained | **no** | — | 0/0 | 1.41 |
| h9-list-msa-schedules | 2 | answer | partial | **no** | 1.0 | 1/1 | 4.6 |
| h9-list-landlord-obligations | 2 | answer | abstained | **no** | — | 0/0 | 5.96 |
| h9-list-kestrel-docs | 2 | answer | partial | yes | 1.0 | 2/2 | 10.1 |
| h9-t06-instalment-investigation | 2 | abstain | abstained | yes | — | 0/0 | 2.52 |
| h9-t06-june-merger | 2 | abstain | abstained | yes | — | 0/0 | 3.07 |
| h9-si-nda-carveout-count | 3 | answer | partial | yes | 1.0 | 1/1 | 6.21 |
| h9-si-okafor-rate | 3 | answer | abstained | **no** | — | 0/0 | 5.95 |
| h9-si-voss-attendance | 3 | answer | partial | yes | 1.0 | 1/1 | 3.99 |
| h9-si-breach-undue-delay | 3 | answer | partial | **no** | 1.0 | 1/1 | 6.7 |
| h9-si-coc-advance | 3 | answer | abstained | **no** | — | 0/0 | 11.1 |
| h9-si-lease-quarter-days | 3 | answer | partial | yes | 1.0 | 1/1 | 4.53 |
| h9-si-amend-invoicing | 3 | answer | partial | yes | 1.0 | 2/2 | 9.36 |
| h9-si-side-agreed | 3 | answer | partial | yes | 0.5 | 2/2 | 12.86 |
| h9-va-meeting-end | 3 | abstain | abstained | yes | — | 0/0 | 4.92 |
| h9-va-sept-venue | 3 | abstain | partial | **no** | — | 1/2 | 10.59 |
| h9-va-side-end | 3 | abstain | partial | **no** | — | 4/4 | 11.21 |
| h9-va-completion-actual | 3 | abstain | partial | **no** | — | 1/1 | 5.94 |
| h9-va-ekberg-since | 3 | abstain | abstained | yes | — | 0/0 | 1.72 |
| h9-va-holm-relative | 3 | abstain | partial | **no** | — | 1/1 | 3.72 |
| h9-va-schedule1 | 3 | abstain | unverified | **no** | — | 0/0 | 5.37 |
| h9-va-estimate-hours | 3 | abstain | abstained | yes | — | 0/0 | 3.58 |
| h9-rp-nda-purpose-only | 3 | answer | abstained | **no** | — | 0/0 | 6.19 |
| h9-rp-nda-advisers-strict | 3 | answer | abstained | **no** | — | 0/0 | 5.41 |
| h9-rp-lease-assign-reasonable | 3 | answer | partial | yes | 1.0 | 1/1 | 5.05 |
| h9-rp-lease-fm-delay | 3 | answer | abstained | **no** | — | 0/0 | 6.16 |
| h9-rp-cw-complaints-email | 3 | answer | answered | yes | 1.0 | 1/1 | 2.93 |
| h9-rp-msa-conf-use | 3 | answer | abstained | **no** | — | 0/0 | 2.35 |
| h9-xdoc-kestrel-return | 3 | abstain | abstained | yes | — | 0/0 | 4.24 |
| h9-xdoc-lease-courts | 3 | abstain | abstained | **no** | — | 0/0 | 3.64 |
| h9-xdoc-kestrel-coc | 3 | abstain | abstained | yes | — | 0/0 | 1.13 |
| h9-xdoc-nda-breach-hours | 3 | abstain | answered | **no** | — | 1/1 | 2.62 |
| h9-list-board-decisions | 3 | answer | abstained | **no** | — | 0/0 | 0.96 |
| h9-list-msa-schedules | 3 | answer | partial | **no** | 1.0 | 2/2 | 4.7 |
| h9-list-landlord-obligations | 3 | answer | abstained | **no** | — | 0/0 | 3.4 |
| h9-list-kestrel-docs | 3 | answer | answered | yes | 1.0 | 2/2 | 6.63 |
| h9-t06-instalment-investigation | 3 | abstain | abstained | yes | — | 0/0 | 1.59 |
| h9-t06-june-merger | 3 | abstain | abstained | yes | — | 0/0 | 0.9 |
