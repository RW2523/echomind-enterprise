# Retrieval, stage by stage (baseline)

Commit 50d0af6; reranker cross-encoder/ms-marco-MiniLM-L-6-v2 (floor -10.5); 151 of 153 single-turn items with located evidence. Rerank p50 0.695 s, p95 2.082 s per search.

| Stage | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| dense | 0.480 | 0.681 | 0.373 | 0.428 |
| sparse | 0.618 | 0.789 | 0.553 | 0.591 |
| keyword | 0.509 | 0.662 | 0.419 | 0.463 |
| fused | 0.636 | 0.742 | 0.485 | 0.533 |
| reranked | 0.574 | 0.736 | 0.457 | 0.510 |
| final | 0.574 | 0.583 | 0.439 | 0.458 |

Search's final passages hold 0.607 of the evidence; the answer model's whole context holds 0.751 (7.392 sources, 5306.235 characters on average).

| Kind | Items | Final recall | Context recall |
|---|---|---|---|
| amendment | 4 | 0.500 | 0.500 |
| calculation | 16 | 0.705 | 0.833 |
| chronology | 2 | 0.500 | 0.500 |
| conflict | 14 | 0.571 | 0.750 |
| cross_doc | 16 | 0.458 | 0.781 |
| definitions | 2 | 0.000 | 0.500 |
| fact | 45 | 0.789 | 0.867 |
| list | 9 | 0.444 | 0.500 |
| not_stated | 21 | 0.548 | 0.643 |
| role_party | 22 | 0.500 | 0.727 |

Failed items of 20261107-150837-chat-v5-rescored.json, by the first stage that lost the evidence:

| Stage | Items |
|---|---|
| in context: answered wrong or incomplete | 14: m101-11, m101-19, m101-21, m101-24, m114-03, m114-09, m114-13, m114-19, m114-20, mat1042-21, m301-06, m302-07, m306-09, m306-12 |
| ranked out by the reranker (below the top passages or the floor) | 11: m101-17, m114-05, mat1042-14, mat1042-18, m301-09, m302-04, m302-11, m304-04, m305-03, m305-07, m306-02 |
| partly in context | 7: m101-04, m101-14, m114-04, m114-29, mat1042-20, m301-08, m304-08 |
| follow-up (not measured) | 5: m101-28, m114-27, mat1042-32, m303-10, m304-09 |
| retrieved by a channel, lost before reranking | 1: m114-16 |
