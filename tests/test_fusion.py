"""Tests for RRF fusion and retrieval metrics.

These verify the maths against hand-computed values. The README's numbers are
only as trustworthy as this file.

    pytest tests/ -v
"""

from __future__ import annotations

import math

import pytest
from langchain_core.documents import Document

from src.eval.metrics import (
    aggregate,
    find_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.retrieval.fusion import (
    DEFAULT_K,
    default_doc_key,
    fuse_to_documents,
    reciprocal_rank_fusion,
)


def make_docs(*texts: str) -> list[Document]:
    return [Document(page_content=text, metadata={"source": "test"}) for text in texts]


# ── RRF scoring ───────────────────────────────────────────────────────────────


def test_single_list_scores_are_one_over_k_plus_rank():
    """With one retriever, doc at rank r must score exactly 1/(k+r)."""
    fused = reciprocal_rank_fusion({"a": make_docs("first", "second", "third")}, k=60)

    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)
    assert fused[2].score == pytest.approx(1 / 63)


def test_scores_sum_across_retrievers():
    """A doc found by both retrievers accumulates both contributions."""
    shared = "found by both"
    fused = reciprocal_rank_fusion(
        {
            "semantic": make_docs(shared, "semantic only"),
            "bm25": make_docs("bm25 only", shared),
        },
        k=60,
    )

    top = fused[0]
    assert top.document.page_content == shared
    # rank 1 in semantic, rank 2 in bm25
    assert top.score == pytest.approx(1 / 61 + 1 / 62)
    assert top.ranks == {"semantic": 1, "bm25": 2}


def test_consensus_beats_single_high_rank():
    """The core property of RRF: agreement outranks one retriever's favourite.

    This is exactly what the old concatenate-and-dedupe code failed to do --
    it always put the semantic retriever's rank-1 result first.
    """
    consensus = "both agree on this"
    fused = reciprocal_rank_fusion(
        {
            "semantic": make_docs("semantic favourite", consensus),
            "bm25": make_docs("bm25 favourite", consensus),
        },
        k=60,
    )

    assert fused[0].document.page_content == consensus
    # 2 * 1/62 = 0.03226 > 1/61 = 0.01639
    assert fused[0].score == pytest.approx(2 / 62)
    assert len(fused[0].ranks) == 2


def test_weights_scale_contributions():
    doc = "weighted"
    fused = reciprocal_rank_fusion(
        {"semantic": make_docs(doc), "bm25": make_docs(doc)},
        k=60,
        weights={"semantic": 2.0, "bm25": 0.5},
    )
    assert fused[0].score == pytest.approx(2.0 / 61 + 0.5 / 61)


def test_smaller_k_sharpens_rank_one_advantage():
    """Lower k concentrates score at the top; higher k flattens the curve."""
    lists = {"a": make_docs("first", "second")}

    sharp = reciprocal_rank_fusion(lists, k=1)
    flat = reciprocal_rank_fusion(lists, k=1000)

    sharp_ratio = sharp[0].score / sharp[1].score  # (1/2)/(1/3) = 1.5
    flat_ratio = flat[0].score / flat[1].score  # ~1.001
    assert sharp_ratio > flat_ratio
    assert flat_ratio == pytest.approx(1.0, abs=0.01)


def test_dedup_by_content_when_no_id_metadata():
    """Chroma and BM25 return distinct objects for the same chunk."""
    same_text = "identical chunk text"
    fused = reciprocal_rank_fusion(
        {
            "semantic": [Document(page_content=same_text, metadata={"source": "a"})],
            "bm25": [Document(page_content=same_text, metadata={"source": "b"})],
        }
    )
    assert len(fused) == 1
    assert len(fused[0].ranks) == 2


def test_dedup_prefers_chunk_id_metadata():
    """When chunk_id is present it wins over text comparison."""
    first = Document(page_content="text A", metadata={"chunk_id": "x1"})
    second = Document(page_content="text B differs", metadata={"chunk_id": "x1"})

    assert default_doc_key(first) == default_doc_key(second)
    assert len(reciprocal_rank_fusion({"a": [first], "b": [second]})) == 1


def test_ordering_is_deterministic_under_ties():
    """Equal scores must break ties by best rank, then retriever count."""
    lists = {
        "a": make_docs("tie one", "tie two"),
        "b": make_docs("tie two", "tie one"),
    }
    first = [r.document.page_content for r in reciprocal_rank_fusion(lists)]
    second = [r.document.page_content for r in reciprocal_rank_fusion(lists)]
    assert first == second


def test_empty_and_partial_inputs():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"a": [], "b": []}) == []

    fused = reciprocal_rank_fusion({"a": make_docs("only one"), "b": []})
    assert len(fused) == 1
    assert fused[0].ranks == {"a": 1}


def test_negative_k_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion({"a": make_docs("x")}, k=-1)


def test_fuse_to_documents_attaches_provenance_without_mutating():
    original = Document(page_content="chunk", metadata={"source": "doc.pdf"})
    result = fuse_to_documents({"semantic": [original], "bm25": [original]}, top_n=1)[0]

    assert result.metadata["source"] == "doc.pdf"
    assert result.metadata["retrievers"] == "bm25,semantic"
    assert result.metadata["rrf_score"] > 0
    # The caller's Document must be untouched.
    assert "rrf_score" not in original.metadata


def test_fuse_to_documents_respects_top_n():
    lists = {"a": make_docs(*[f"doc {i}" for i in range(20)])}
    assert len(fuse_to_documents(lists, top_n=5)) == 5


def test_default_k_matches_published_constant():
    assert DEFAULT_K == 60


# ── metrics ───────────────────────────────────────────────────────────────────


def test_recall_at_k_boundary():
    assert recall_at_k(5, 5) == 1.0  # inclusive at the cutoff
    assert recall_at_k(6, 5) == 0.0
    assert recall_at_k(1, 5) == 1.0
    assert recall_at_k(None, 5) == 0.0


def test_reciprocal_rank_values():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(2) == 0.5
    assert reciprocal_rank(10) == pytest.approx(0.1)
    assert reciprocal_rank(None) == 0.0


def test_ndcg_matches_log_discount():
    assert ndcg_at_k(1, 10) == pytest.approx(1.0)  # 1/log2(2)
    assert ndcg_at_k(3, 10) == pytest.approx(1 / math.log2(4))  # 0.5
    assert ndcg_at_k(11, 10) == 0.0  # beyond cutoff
    assert ndcg_at_k(None, 10) == 0.0


def test_ndcg_penalises_less_steeply_than_mrr():
    """At rank 3: MRR gives 0.333, nDCG gives 0.5."""
    assert ndcg_at_k(3, 10) > reciprocal_rank(3)


def test_find_rank_is_one_based():
    assert find_rank(["a", "b", "c"], "a") == 1
    assert find_rank(["a", "b", "c"], "c") == 3
    assert find_rank(["a", "b", "c"], "missing") is None
    assert find_rank([], "a") is None


def test_aggregate_hand_computed():
    """Five queries with known ranks -- verify every aggregate by hand."""
    ranks = [1, 2, 5, None, 3]

    metrics = aggregate(ranks)

    assert metrics["n_queries"] == 5
    assert metrics["n_found"] == 4
    assert metrics["recall@1"] == pytest.approx(1 / 5)  # only rank 1
    assert metrics["recall@3"] == pytest.approx(3 / 5)  # ranks 1, 2, 3
    assert metrics["recall@5"] == pytest.approx(4 / 5)  # ranks 1, 2, 5, 3
    assert metrics["mrr"] == pytest.approx((1 + 0.5 + 0.2 + 0 + 1 / 3) / 5)
    assert metrics["mean_rank_when_found"] == pytest.approx((1 + 2 + 5 + 3) / 4)


def test_aggregate_recall_is_monotonic_in_k():
    metrics = aggregate([1, 4, 7, None, 2, 9])
    assert (
        metrics["recall@1"]
        <= metrics["recall@3"]
        <= metrics["recall@5"]
        <= metrics["recall@10"]
    )


def test_aggregate_all_missed_and_all_perfect():
    missed = aggregate([None, None, None])
    assert missed["recall@10"] == 0.0
    assert missed["mrr"] == 0.0
    assert missed["mean_rank_when_found"] == 0.0

    perfect = aggregate([1, 1, 1])
    assert perfect["recall@1"] == 1.0
    assert perfect["mrr"] == 1.0


def test_aggregate_empty():
    assert aggregate([]) == {"n_queries": 0}
