"""Retrieval metrics.

Separated from the evaluation runner so the maths is unit-testable without
loading models or hitting a vector store. Every function takes the rank of the
relevant document (1-based, or None if it was not retrieved).
"""

from __future__ import annotations

import math
from statistics import mean


def recall_at_k(rank: int | None, k: int) -> float:
    """1.0 if the relevant document appears in the top k, else 0.0.

    With exactly one relevant document per query, recall@k reduces to hit rate:
    "how often is the right chunk in the top k results?" This is the metric
    that matters most for RAG, because a chunk the retriever misses is a chunk
    the generator can never use.
    """
    if rank is None:
        return 0.0
    return 1.0 if rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    """1/rank, or 0.0 if not retrieved.

    Rewards ranking the right chunk higher, not merely including it. Rank 1
    scores 1.0, rank 2 scores 0.5, rank 10 scores 0.1. Averaged over queries
    this gives MRR.
    """
    if rank is None:
        return 0.0
    return 1.0 / rank


def ndcg_at_k(rank: int | None, k: int) -> float:
    """Normalised discounted cumulative gain with a single relevant document.

    DCG = 1/log2(rank+1) when the document is in the top k. The ideal DCG is
    1.0 (relevant document at rank 1), so nDCG = DCG here. Included because
    its log discount penalises deep ranks more gently than MRR does.
    """
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def find_rank(retrieved_ids: list[str], target_id: str) -> int | None:
    """1-based position of target_id in retrieved_ids, or None if absent."""
    try:
        return retrieved_ids.index(target_id) + 1
    except ValueError:
        return None


def aggregate(ranks: list[int | None], k_values: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Aggregate per-query ranks into the summary metrics.

    Args:
        ranks: One entry per query -- the rank of that query's relevant
            document, or None if it was not retrieved.
        k_values: Cutoffs to report recall and nDCG at.

    Returns:
        Metric name -> value, plus n_queries and n_found for context.
    """
    if not ranks:
        return {"n_queries": 0}

    results: dict[str, float | int] = {
        "n_queries": len(ranks),
        "n_found": sum(1 for rank in ranks if rank is not None),
    }

    for k in k_values:
        results[f"recall@{k}"] = mean(recall_at_k(rank, k) for rank in ranks)

    results["mrr"] = mean(reciprocal_rank(rank) for rank in ranks)
    results["ndcg@10"] = mean(ndcg_at_k(rank, 10) for rank in ranks)

    found = [rank for rank in ranks if rank is not None]
    results["mean_rank_when_found"] = mean(found) if found else 0.0

    return results


def format_table(rows: list[tuple[str, dict]], k_values: tuple[int, ...] = (1, 3, 5, 10)) -> str:
    """Render results as a markdown table ready to paste into the README."""
    headers = ["Configuration"] + [f"Recall@{k}" for k in k_values] + ["MRR", "nDCG@10"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for label, metrics in rows:
        cells = [label]
        cells += [f"{metrics.get(f'recall@{k}', 0):.3f}" for k in k_values]
        cells.append(f"{metrics.get('mrr', 0):.3f}")
        cells.append(f"{metrics.get('ndcg@10', 0):.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)
