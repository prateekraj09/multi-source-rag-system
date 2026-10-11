"""Cross-encoder reranking stage.

Bi-encoders (all-MiniLM-L6-v2, used for the vector index) embed the query and
each chunk independently, so they never see the two together. A cross-encoder
scores the (query, chunk) pair jointly, which is far more accurate but far too
slow to run over a whole corpus.

The standard arrangement -- and the one used here -- is retrieve-then-rerank:
fetch a generous candidate set cheaply (RRF over semantic + BM25), then rerank
only those candidates with the cross-encoder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ~22M params, CPU-friendly (a few hundred ms for 30 candidates on a laptop).
# For higher accuracy at ~3x the latency use "cross-encoder/ms-marco-MiniLM-L-12-v2".
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankedResult:
    document: Document
    score: float
    original_rank: int
    new_rank: int

    @property
    def rank_delta(self) -> int:
        """Positive means the reranker promoted this document."""
        return self.original_rank - self.new_rank


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    """Load and cache the cross-encoder.

    Cached because model load costs seconds and Streamlit re-executes the whole
    script on every interaction.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Cross-encoder reranking requires sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    logger.info("Loading cross-encoder %s (first call only)", model_name)
    return CrossEncoder(model_name, max_length=512)


def rerank(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> list[RerankedResult]:
    """Rerank candidate documents against the query with a cross-encoder.

    Args:
        query: The user's question.
        documents: Candidates, in their pre-rerank order (best first).
        top_n: Keep only this many after reranking. None keeps all.
        model_name: HuggingFace cross-encoder identifier.
        batch_size: Pairs scored per forward pass.

    Returns:
        RerankedResults sorted by descending relevance, each carrying its
        original and new rank so the improvement is measurable.
    """
    if not documents:
        return []

    model = _load_cross_encoder(model_name)
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    scored = sorted(
        zip(documents, scores, range(1, len(documents) + 1)),
        key=lambda triple: float(triple[1]),
        reverse=True,
    )

    results = [
        RerankedResult(
            document=document,
            score=float(score),
            original_rank=original_rank,
            new_rank=new_rank,
        )
        for new_rank, (document, score, original_rank) in enumerate(scored, start=1)
    ]

    return results[:top_n] if top_n is not None else results


def rerank_to_documents(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
    model_name: str = DEFAULT_MODEL,
    attach_provenance: bool = True,
) -> list[Document]:
    """rerank() returning plain Documents, with scores kept in metadata."""
    results = rerank(query, documents, top_n=top_n, model_name=model_name)

    if not attach_provenance:
        return [result.document for result in results]

    return [
        Document(
            page_content=result.document.page_content,
            metadata={
                **result.document.metadata,
                "rerank_score": round(result.score, 4),
                "rank_delta": result.rank_delta,
            },
        )
        for result in results
    ]


def explain_reranking(results: list[RerankedResult], limit: int = 10) -> str:
    """Show how the reranker reordered candidates -- useful in a demo."""
    lines = [f"{'new':<6}{'was':<6}{'delta':<8}{'score':<12}preview"]
    for result in results[:limit]:
        arrow = "^" if result.rank_delta > 0 else ("v" if result.rank_delta < 0 else "=")
        preview = result.document.page_content[:60].replace("\n", " ")
        lines.append(
            f"{result.new_rank:<6}{result.original_rank:<6}"
            f"{arrow}{abs(result.rank_delta):<7}{result.score:<12.4f}{preview}..."
        )
    return "\n".join(lines)
