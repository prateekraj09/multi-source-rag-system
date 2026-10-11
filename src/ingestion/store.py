"""Vector store construction and batched insertion.

Chroma's Rust backend rejects any single upsert larger than ~5,461 records
(the limit derives from SQLite's maximum variable count per statement). A
600-page PDF easily exceeds that, so all writes go through add_documents here
rather than calling db.add_documents directly.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Conservative default; the real ceiling is queried from the client when
# available, since it varies with Chroma version and backend.
FALLBACK_BATCH_SIZE = 5000


def get_embeddings(model_name: str = EMBEDDING_MODEL):
    """Load the sentence-transformer embedding model.

    normalize_embeddings=True makes cosine similarity equivalent to a dot
    product, which is what Chroma's cosine metric expects.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vector_store(persist_dir: str = "chroma_index", embeddings=None):
    """Open (or create) the Chroma collection."""
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings or get_embeddings(),
    )


def max_batch_size(db) -> int:
    """Largest upsert the backing Chroma client will accept.

    Queried at runtime because the limit is version- and backend-dependent.
    Falls back to a safe constant if the private API isn't present.
    """
    try:
        limit = db._client.get_max_batch_size()
        if isinstance(limit, int) and limit > 0:
            # Leave headroom: the limit counts records, and some versions
            # apply it slightly differently across upsert paths.
            return max(limit - 100, 1)
    except (AttributeError, TypeError, Exception):  # noqa: B014 - defensive
        pass
    return FALLBACK_BATCH_SIZE


def add_documents(
    db,
    documents: list[Document],
    batch_size: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Insert documents into the vector store, batching to respect the limit.

    Args:
        db: Chroma vector store.
        documents: Chunks to insert.
        batch_size: Records per insert. Detected from the client if omitted.
        on_progress: Called as (inserted_so_far, total) after each batch --
            use it to drive a progress bar.

    Returns:
        Number of documents inserted.

    Raises:
        RuntimeError: If a batch fails, with the batch range included so the
            failure can be located in a large ingest.
    """
    if not documents:
        return 0

    size = batch_size or max_batch_size(db)
    total = len(documents)

    if total > size:
        logger.info("Inserting %d chunks in batches of %d", total, size)

    inserted = 0
    for start in range(0, total, size):
        batch = documents[start : start + size]
        try:
            db.add_documents(batch)
        except Exception as error:
            raise RuntimeError(
                f"Failed inserting documents {start}-{start + len(batch)} "
                f"of {total}: {error}"
            ) from error

        inserted += len(batch)
        if on_progress:
            on_progress(inserted, total)

    return inserted


def count_documents(db) -> int:
    """Number of chunks currently indexed."""
    return len(db.get().get("documents") or [])


def iter_documents(db) -> Iterable[Document]:
    """Yield every stored chunk as a Document."""
    stored = db.get()
    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or [{}] * len(documents)

    for index, text in enumerate(documents):
        yield Document(page_content=text, metadata=metadatas[index] or {})
