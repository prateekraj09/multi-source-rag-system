"""Tests for batched vector-store insertion.

Regression coverage for the Chroma limit: a single upsert above ~5,461 records
fails with "Batch size of N is greater than max batch size of 5461". A 600-page
PDF exceeds that, so ingestion must chunk its writes.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.ingestion.store import (
    FALLBACK_BATCH_SIZE,
    add_documents,
    count_documents,
    iter_documents,
    max_batch_size,
)

CHROMA_LIMIT = 5461


class FakeChromaClient:
    def __init__(self, limit: int | None = CHROMA_LIMIT):
        self._limit = limit

    def get_max_batch_size(self):
        if self._limit is None:
            raise AttributeError("not supported on this backend")
        return self._limit


class FakeVectorStore:
    """Minimal Chroma stand-in that enforces the batch ceiling."""

    def __init__(self, limit: int | None = CHROMA_LIMIT):
        self._client = FakeChromaClient(limit)
        self._limit = limit
        self.documents: list[Document] = []
        self.batch_sizes: list[int] = []

    def add_documents(self, documents):
        if self._limit is not None and len(documents) > self._limit:
            raise ValueError(
                f"Batch size of {len(documents)} is greater than "
                f"max batch size of {self._limit}"
            )
        self.batch_sizes.append(len(documents))
        self.documents.extend(documents)

    def get(self):
        return {
            "documents": [doc.page_content for doc in self.documents],
            "metadatas": [doc.metadata for doc in self.documents],
        }


class ClientlessStore(FakeVectorStore):
    """A store exposing no _client, to exercise the fallback path."""

    def __init__(self):
        super().__init__(limit=None)
        del self._client


def make_docs(count: int) -> list[Document]:
    return [
        Document(page_content=f"chunk {i}", metadata={"source": "test.pdf", "i": i})
        for i in range(count)
    ]


def test_large_insert_would_fail_unbatched():
    """Confirms the fake reproduces the real error, so the test means something."""
    store = FakeVectorStore()
    with pytest.raises(ValueError, match="greater than max batch size"):
        store.add_documents(make_docs(6120))


def test_add_documents_batches_the_failing_case():
    """6,120 chunks -- the exact size that broke ingestion."""
    store = FakeVectorStore()
    inserted = add_documents(store, make_docs(6120))

    assert inserted == 6120
    assert len(store.documents) == 6120
    assert len(store.batch_sizes) > 1
    assert all(size <= CHROMA_LIMIT for size in store.batch_sizes)


def test_small_insert_stays_single_batch():
    store = FakeVectorStore()
    add_documents(store, make_docs(100))
    assert store.batch_sizes == [100]


def test_batches_respect_explicit_size():
    store = FakeVectorStore()
    add_documents(store, make_docs(250), batch_size=100)
    assert store.batch_sizes == [100, 100, 50]


def test_document_order_is_preserved():
    """Chunk order matters -- metadata carries positional context."""
    store = FakeVectorStore()
    add_documents(store, make_docs(500), batch_size=99)
    assert [doc.metadata["i"] for doc in store.documents] == list(range(500))


def test_empty_input_is_a_noop():
    store = FakeVectorStore()
    assert add_documents(store, []) == 0
    assert store.batch_sizes == []


def test_progress_callback_reports_cumulative_totals():
    store = FakeVectorStore()
    seen: list[tuple[int, int]] = []

    add_documents(store, make_docs(250), batch_size=100, on_progress=lambda d, t: seen.append((d, t)))

    assert seen == [(100, 250), (200, 250), (250, 250)]


def test_max_batch_size_leaves_headroom():
    assert max_batch_size(FakeVectorStore(limit=CHROMA_LIMIT)) < CHROMA_LIMIT


def test_max_batch_size_falls_back_without_client():
    assert max_batch_size(ClientlessStore()) == FALLBACK_BATCH_SIZE


def test_failure_reports_which_batch():
    class FailsOnSecondBatch(FakeVectorStore):
        def add_documents(self, documents):
            if len(self.batch_sizes) == 1:
                raise RuntimeError("disk full")
            super().add_documents(documents)

    with pytest.raises(RuntimeError, match=r"100-200 of 250"):
        add_documents(FailsOnSecondBatch(), make_docs(250), batch_size=100)


def test_count_and_iter_round_trip():
    store = FakeVectorStore()
    add_documents(store, make_docs(10))

    assert count_documents(store) == 10

    recovered = list(iter_documents(store))
    assert len(recovered) == 10
    assert recovered[3].page_content == "chunk 3"
    assert recovered[3].metadata["source"] == "test.pdf"
