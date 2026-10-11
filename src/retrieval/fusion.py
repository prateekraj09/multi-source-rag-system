"""Reciprocal Rank Fusion for combining multiple ranked retrieval results.

RRF scores a document by summing 1 / (k + rank) across every ranked list it
appears in, where rank is 1-based. A document ranked highly by several
retrievers therefore outscores one ranked highly by a single retriever, and no
score normalisation between retrievers is required -- only the ordering matters.

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, Sequence

from langchain_core.documents import Document

# k=60 is the constant used in the original paper. It damps the contribution of
# top ranks so that a single retriever cannot dominate the fused ordering.
DEFAULT_K = 60


def default_doc_key(doc: Document) -> Hashable:
    """Identity for deduplication across retrievers.

    Prefers a stable metadata id when the vector store provides one, and falls
    back to the chunk text. Falling back to text is important because BM25 and
    Chroma return distinct Document objects for the same underlying chunk.
    """
    for key in ("chunk_id", "id", "_id"):
        value = doc.metadata.get(key)
        if value is not None:
            return f"{key}:{value}"
    return f"text:{doc.page_content}"


@dataclass
class FusedResult:
    """A document plus its fused score and per-retriever provenance."""

    document: Document
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def retrievers(self) -> list[str]:
        """Names of the retrievers that surfaced this document."""
        return sorted(self.ranks)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Sequence[Document]],
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
    key_fn: Callable[[Document], Hashable] = default_doc_key,
) -> list[FusedResult]:
    """Fuse several ranked document lists into one ordering.

    Args:
        ranked_lists: Mapping of retriever name -> its ranked documents, best
            first. Names are arbitrary labels used for provenance.
        k: RRF damping constant. Larger values flatten the influence of rank.
        weights: Optional per-retriever multipliers. Missing entries default to
            1.0. Use this to bias, for example, semantic search over BM25.
        key_fn: Maps a Document to the identity used for deduplication.

    Returns:
        FusedResults sorted by descending score. Ties are broken by best
        (lowest) rank achieved in any list, then by the number of retrievers
        that found the document, so the ordering is deterministic.

    Raises:
        ValueError: If k is negative.
    """
    if k < 0:
        raise ValueError(f"RRF constant k must be non-negative, got {k}")

    weights = weights or {}
    accumulated: dict[Hashable, FusedResult] = {}

    for retriever_name, documents in ranked_lists.items():
        weight = weights.get(retriever_name, 1.0)

        for zero_based, document in enumerate(documents):
            rank = zero_based + 1  # RRF is defined over 1-based ranks.
            identity = key_fn(document)
            contribution = weight / (k + rank)

            existing = accumulated.get(identity)
            if existing is None:
                accumulated[identity] = FusedResult(
                    document=document,
                    score=contribution,
                    ranks={retriever_name: rank},
                )
            else:
                existing.score += contribution
                # Keep the best rank if a retriever somehow yields duplicates.
                previous = existing.ranks.get(retriever_name)
                if previous is None or rank < previous:
                    existing.ranks[retriever_name] = rank

    return sorted(
        accumulated.values(),
        key=lambda result: (
            -result.score,
            min(result.ranks.values()),
            -len(result.ranks),
        ),
    )


def fuse_to_documents(
    ranked_lists: dict[str, Sequence[Document]],
    top_n: int | None = None,
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
    attach_provenance: bool = True,
) -> list[Document]:
    """Convenience wrapper returning plain Documents instead of FusedResults.

    When attach_provenance is True each returned Document gets `rrf_score` and
    `retrievers` metadata, which the UI can surface to show *why* a chunk was
    retrieved. Metadata is written on a copy so the vector store's cached
    Documents are never mutated.
    """
    fused = reciprocal_rank_fusion(ranked_lists, k=k, weights=weights)
    if top_n is not None:
        fused = fused[:top_n]

    documents: list[Document] = []
    for result in fused:
        if not attach_provenance:
            documents.append(result.document)
            continue

        documents.append(
            Document(
                page_content=result.document.page_content,
                metadata={
                    **result.document.metadata,
                    "rrf_score": round(result.score, 6),
                    "retrievers": ",".join(result.retrievers),
                },
            )
        )
    return documents


def explain_fusion(fused: Iterable[FusedResult], limit: int = 10) -> str:
    """Human-readable breakdown of a fusion, for debugging and demos."""
    lines = [f"{'#':<4}{'score':<12}{'retriever ranks':<40}preview"]
    for position, result in enumerate(list(fused)[:limit], start=1):
        ranks = ", ".join(f"{name}@{rank}" for name, rank in sorted(result.ranks.items()))
        preview = result.document.page_content[:60].replace("\n", " ")
        lines.append(f"{position:<4}{result.score:<12.6f}{ranks:<40}{preview}...")
    return "\n".join(lines)
