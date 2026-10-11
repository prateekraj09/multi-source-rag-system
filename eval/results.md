## Retrieval evaluation

Evaluated on 30 questions generated from the indexed corpus and manually reviewed. Each question has exactly one known-correct source chunk, so recall@k is the share of questions whose correct chunk appears in the top k results.

| Configuration | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Semantic only (bi-encoder) | 0.400 | 0.467 | 0.533 | 0.600 | 0.460 | 0.494 |
| BM25 only (keyword) | 0.567 | 0.633 | 0.667 | 0.700 | 0.606 | 0.628 |
| Hybrid + RRF | 0.333 | 0.567 | 0.600 | 0.767 | 0.468 | 0.539 |
| Hybrid + RRF + cross-encoder rerank | 0.633 | 0.733 | 0.800 | 0.833 | 0.700 | 0.732 |

### Latency

| Configuration | Median | p95 |
| --- | --- | --- |
| Semantic only (bi-encoder) | 5 ms | 6 ms |
| BM25 only (keyword) | 6 ms | 6 ms |
| Hybrid + RRF | 11 ms | 12 ms |
| Hybrid + RRF + cross-encoder rerank | 138 ms | 278 ms |

### Takeaway

**Hybrid + RRF + cross-encoder rerank** improves recall@5 from **0.533** to **0.800** (+50% relative) over the semantic-only baseline, at a cost of +133 ms median latency.

Caveat: questions were LLM-generated from the source chunks, so they share vocabulary with the text they came from. This favours keyword matching, and the absolute numbers are optimistic relative to real user queries. The comparison between configurations remains valid since all four are scored on the identical question set.