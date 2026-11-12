# Retrieval, stage by stage (item1-stems-headings)

Commit f8d4de8; reranker cross-encoder/ms-marco-MiniLM-L-6-v2 (floor -10.5); 151 of 153 single-turn items with located evidence. Rerank p50 0.458 s, p95 1.093 s per search.

| Stage | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| dense | 0.480 | 0.681 | 0.373 | 0.428 |
| sparse | 0.642 | 0.787 | 0.567 | 0.605 |
| keyword | 0.509 | 0.662 | 0.419 | 0.463 |
| fused | 0.643 | 0.772 | 0.503 | 0.552 |
| reranked | 0.571 | 0.743 | 0.459 | 0.513 |
| floored | 0.571 | 0.743 | 0.459 | 0.513 |
| final | 0.667 | 0.703 | 0.514 | 0.544 |
| hybrid | 0.597 | 0.763 | 0.493 | 0.541 |
| hybrid_sparse | 0.667 | 0.815 | 0.537 | 0.587 |

Search's final passages hold 0.720 of the evidence; the answer model's whole context holds 0.809 (7.255 sources, 5241.523 characters on average).

| Kind | Items | Final recall | Context recall |
|---|---|---|---|
| amendment | 4 | 0.750 | 0.750 |
| calculation | 16 | 0.830 | 0.896 |
| chronology | 2 | 0.500 | 0.500 |
| conflict | 14 | 0.619 | 0.762 |
| cross_doc | 16 | 0.677 | 0.760 |
| definitions | 2 | 0.250 | 0.500 |
| fact | 45 | 0.822 | 0.911 |
| list | 9 | 0.444 | 0.500 |
| not_stated | 21 | 0.690 | 0.738 |
| role_party | 22 | 0.727 | 0.864 |
