# Retrieval, stage by stage (item8-rerank-top)

Commit b33911a; reranker cross-encoder/ms-marco-MiniLM-L-6-v2 (floor -10.5); 151 of 153 single-turn items with located evidence. Rerank p50 0.743 s, p95 1.474 s per search.

| Stage | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| dense | 0.480 | 0.681 | 0.373 | 0.428 |
| sparse | 0.642 | 0.787 | 0.567 | 0.605 |
| keyword | 0.509 | 0.662 | 0.419 | 0.463 |
| fused | 0.643 | 0.772 | 0.503 | 0.552 |
| reranked | 0.643 | 0.772 | 0.503 | 0.552 |
| floored | 0.643 | 0.772 | 0.503 | 0.552 |
| final | 0.643 | 0.698 | 0.495 | 0.526 |
| hybrid | 0.643 | 0.772 | 0.503 | 0.552 |
| hybrid_sparse | 0.636 | 0.781 | 0.550 | 0.590 |

Search's final passages hold 0.708 of the evidence; the answer model's whole context holds 0.827 (7.562 sources, 5340.366 characters on average).

| Kind | Items | Final recall | Context recall |
|---|---|---|---|
| amendment | 4 | 1.000 | 1.000 |
| calculation | 16 | 0.747 | 0.844 |
| chronology | 2 | 0.500 | 0.500 |
| conflict | 14 | 0.667 | 0.774 |
| cross_doc | 16 | 0.635 | 0.750 |
| definitions | 2 | 0.250 | 0.500 |
| fact | 45 | 0.767 | 0.856 |
| list | 9 | 0.444 | 0.722 |
| not_stated | 21 | 0.786 | 0.833 |
| role_party | 22 | 0.682 | 0.909 |
