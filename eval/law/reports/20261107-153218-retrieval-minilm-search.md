# Retrieval, stage by stage (minilm-search)

Commit 50d0af6; reranker cross-encoder/ms-marco-MiniLM-L-6-v2 (floor -10.5); 151 of 153 single-turn items with located evidence. Rerank p50 0.750 s, p95 2.483 s per search.

| Stage | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| dense | 0.480 | 0.681 | 0.373 | 0.428 |
| sparse | 0.618 | 0.789 | 0.553 | 0.591 |
| keyword | 0.509 | 0.662 | 0.419 | 0.463 |
| fused | 0.636 | 0.742 | 0.485 | 0.533 |
| reranked | 0.574 | 0.736 | 0.457 | 0.510 |
| floored | 0.574 | 0.733 | 0.456 | 0.509 |
| final | 0.574 | 0.583 | 0.439 | 0.458 |
| hybrid | 0.591 | 0.746 | 0.484 | 0.529 |
| hybrid_sparse | 0.631 | 0.785 | 0.532 | 0.576 |

Search's final passages hold – of the evidence; the answer model's whole context holds – (0.0 sources, 0.0 characters on average).

| Kind | Items | Final recall | Context recall |
|---|---|---|---|
| amendment | 4 | – | – |
| calculation | 16 | – | – |
| chronology | 2 | – | – |
| conflict | 14 | – | – |
| cross_doc | 16 | – | – |
| definitions | 2 | – | – |
| fact | 45 | – | – |
| list | 9 | – | – |
| not_stated | 21 | – | – |
| role_party | 22 | – | – |
