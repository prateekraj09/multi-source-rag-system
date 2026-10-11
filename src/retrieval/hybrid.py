"""Configurable hybrid retriever.

Every stage is switchable so the same code path serves both the app and the
ablation study. That matters: if the numbers in the README came from a
different implementation than the one the app runs, they would not mean
anything.

    HybridRetriever(db, RetrievalConfig(mode="rrf_rerank")).retrieve(query)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .fusion import DEFAULT_K, fuse_to_documents
from .rerank import DEFAULT_MODEL, rerank_to_documents

logger = logging.getLogger(__name__)

Mode = Literal["semantic", "bm25", "rrf", "rrf_rerank"]

# The four configurations compared in the ablation table.
ALL_MODES: tuple[Mode, ...] = ("semantic", "bm25", "rrf", "rrf_rerank")


@dataclass
class RetrievalConfig:
    """Knobs for the retrieval pipeline.

    Attributes:
        mode: Which pipeline to run. "semantic" and "bm25" are the single-
            retriever baselines; "rrf" fuses them; "rrf_rerank" adds the
            cross-encoder.
        candidate_k: Documents each base retriever contributes before fusion.
            Should exceed final_k -- fusion and reranking can only reorder what
            they are given, so a wider candidate pool raises the recall ceiling.
        final_k: Documents returned to the caller.
        rrf_k: RRF damping constant.
        weights: Optional per-retriever weights for fusion.
        rerank_model: Cross-encoder identifier.
    """

    mode: Mode = "rrf_rerank"
    candidate_k: int = 20
    final_k: int = 6
    rrf_k: int = DEFAULT_K
    weights: dict[str, float] = field(default_factory=dict)
    rerank_model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        if self.mode not in ALL_MODES:
            raise ValueError(f"mode must be one of {ALL_MODES}, got {self.mode!r}")
        if self.final_k < 1:
            raise ValueError(f"final_k must be >= 1, got {self.final_k}")
        if self.candidate_k < self.final_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be >= final_k ({self.final_k}); "
                "fusion and reranking can only reorder the candidates they receive."
            )

    @property
    def label(self) -> str:
        """Readable name for results tables."""
        return {
            "semantic": "Semantic only (bi-encoder)",
            "bm25": "BM25 only (keyword)",
            "rrf": "Hybrid + RRF",
            "rrf_rerank": "Hybrid + RRF + cross-encoder rerank",
        }[self.mode]


@dataclass
class RetrievalTrace:
    """Per-stage timings and counts, for the UI and for latency reporting."""

    mode: str
    n_semantic: int = 0
    n_bm25: int = 0
    n_fused: int = 0
    n_returned: int = 0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.rerank_ms


class HybridRetriever:
    """Semantic + BM25 retrieval with optional RRF fusion and reranking."""

    def __init__(self, db, config: RetrievalConfig | None = None):
        """
        Args:
            db: A Chroma vector store (any LangChain store exposing
                `as_retriever` and `get` works).
            config: Pipeline configuration. Defaults to the full pipeline.
        """
        self.db = db
        self.config = config or RetrievalConfig()
        self._bm25: BM25Retriever | None = None
        self._bm25_doc_count = 0

    # -- base retrievers ---------------------------------------------------

    def _semantic_search(self, query: str, k: int) -> list[Document]:
        retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return retriever.invoke(query)

    def _get_bm25(self, k: int) -> BM25Retriever | None:
        """Build the BM25 index over the corpus, rebuilding when it grows.

        BM25 needs the full document set in memory, unlike the vector store.
        The count check means ingesting new sources invalidates the index
        instead of silently searching a stale corpus.
        """
        stored = self.db.get()
        documents = stored.get("documents") or []
        if not documents:
            return None

        if self._bm25 is not None and len(documents) == self._bm25_doc_count:
            self._bm25.k = k
            return self._bm25

        metadatas = stored.get("metadatas") or [{}] * len(documents)
        corpus = [
            Document(page_content=text, metadata=metadatas[i] or {})
            for i, text in enumerate(documents)
        ]

        logger.info("Building BM25 index over %d chunks", len(corpus))
        self._bm25 = BM25Retriever.from_documents(corpus)
        self._bm25.k = k
        self._bm25_doc_count = len(documents)
        return self._bm25

    def _bm25_search(self, query: str, k: int) -> list[Document]:
        bm25 = self._get_bm25(k)
        return bm25.invoke(query) if bm25 else []

    # -- public API --------------------------------------------------------

    def retrieve(self, query: str, trace: bool = False):
        """Retrieve documents for a query.

        Args:
            query: The user's question.
            trace: If True, return (documents, RetrievalTrace).

        Returns:
            A list of Documents, or (documents, trace) when trace is True.
        """
        config = self.config
        record = RetrievalTrace(mode=config.mode)
        started = time.perf_counter()

        if config.mode == "semantic":
            documents = self._semantic_search(query, config.final_k)
            record.n_semantic = len(documents)

        elif config.mode == "bm25":
            documents = self._bm25_search(query, config.final_k)
            record.n_bm25 = len(documents)

        else:  # "rrf" or "rrf_rerank"
            semantic_docs = self._semantic_search(query, config.candidate_k)
            bm25_docs = self._bm25_search(query, config.candidate_k)
            record.n_semantic = len(semantic_docs)
            record.n_bm25 = len(bm25_docs)

            ranked_lists = {"semantic": semantic_docs, "bm25": bm25_docs}

            if config.mode == "rrf":
                documents = fuse_to_documents(
                    ranked_lists,
                    top_n=config.final_k,
                    k=config.rrf_k,
                    weights=config.weights,
                )
                record.n_fused = len(documents)
            else:
                # Rerank the full fused candidate pool, not just final_k --
                # otherwise the cross-encoder can never rescue a document that
                # RRF ranked poorly, which is much of its value.
                candidates = fuse_to_documents(
                    ranked_lists,
                    top_n=config.candidate_k,
                    k=config.rrf_k,
                    weights=config.weights,
                )
                record.n_fused = len(candidates)
                record.retrieve_ms = (time.perf_counter() - started) * 1000

                rerank_started = time.perf_counter()
                documents = rerank_to_documents(
                    query,
                    candidates,
                    top_n=config.final_k,
                    model_name=config.rerank_model,
                )
                record.rerank_ms = (time.perf_counter() - rerank_started) * 1000
                record.n_returned = len(documents)
                return (documents, record) if trace else documents

        record.retrieve_ms = (time.perf_counter() - started) * 1000
        record.n_returned = len(documents)
        return (documents, record) if trace else documents

    def __call__(self, query: str) -> list[Document]:
        """Callable form, drop-in compatible with the old `ensemble_retrieve`."""
        return self.retrieve(query)


def build_retriever(db, mode: Mode = "rrf_rerank", **overrides) -> HybridRetriever:
    """Shorthand: build_retriever(db, "rrf", final_k=8)."""
    return HybridRetriever(db, RetrievalConfig(mode=mode, **overrides))
