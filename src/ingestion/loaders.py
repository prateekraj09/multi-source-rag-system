"""Source loaders for PDF, web, and CSV ingestion.

Extracted from the Streamlit app so the same code path serves the UI, the
CLI, and evaluation. Each loader returns split, cleaned Documents ready to be
added to the vector store.
"""

from __future__ import annotations

import csv as csv_module
import io
import os
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .clean import clean_text

# Chunks below this many characters are usually headers, page numbers, or
# extraction noise -- they add index bloat without being retrievable answers.
MIN_CHUNK_CHARS = 100
MIN_CSV_ROW_CHARS = 30

os.environ.setdefault("USER_AGENT", "multi-source-rag-system/1.0")


def split_text(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks.

    700/50 is tuned for the prose in this corpus: large enough to hold a
    complete idea, small enough that a chunk is mostly relevant to any query
    that retrieves it. The separator order prefers paragraph, then line, then
    sentence breaks, so chunks rarely split mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _filter_short(documents: list[Document], minimum: int = MIN_CHUNK_CHARS) -> list[Document]:
    return [doc for doc in documents if len(doc.page_content.strip()) > minimum]


def load_pdf(source, filename: str | None = None) -> list[Document]:
    """Load a PDF from a path or an uploaded file object.

    Args:
        source: Filesystem path, or a file-like object with .read() (as
            Streamlit's uploader provides).
        filename: Display name recorded in metadata. Inferred from a path when
            omitted.

    Returns:
        Cleaned, split Documents tagged with their source name.
    """
    temp_path = None

    if hasattr(source, "read"):
        display_name = filename or getattr(source, "name", "uploaded.pdf")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
            handle.write(source.read())
            temp_path = handle.name
        path = temp_path
    else:
        path = str(source)
        display_name = filename or Path(path).name
        if not Path(path).exists():
            raise FileNotFoundError(f"PDF not found: {path}")

    try:
        documents = PyMuPDFLoader(path).load()
    finally:
        if temp_path:
            os.unlink(temp_path)

    for document in documents:
        document.page_content = clean_text(document.page_content)
        document.metadata["source"] = display_name

    return split_text(_filter_short(documents))


def load_url(url: str) -> list[Document]:
    """Scrape and clean a web page."""
    documents = WebBaseLoader([url]).load()

    for document in documents:
        document.page_content = clean_text(document.page_content)
        document.metadata["source"] = url

    return split_text(_filter_short(documents))


def load_csv(source, filename: str | None = None) -> list[Document]:
    """Convert CSV rows into natural-language sentences.

    Each row becomes "column is value. column is value." because embedding
    models are trained on prose, not comma-delimited fields -- a raw CSV row
    embeds poorly and retrieves badly.
    """
    if hasattr(source, "read"):
        display_name = filename or getattr(source, "name", "uploaded.csv")
        raw = source.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    else:
        path = str(source)
        display_name = filename or Path(path).name
        if not Path(path).exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        text = Path(path).read_text(encoding="utf-8", errors="replace")

    reader = csv_module.DictReader(io.StringIO(text))
    documents = []

    for index, row in enumerate(reader):
        sentence = ". ".join(
            f"{key.strip()} is {value.strip()}"
            for key, value in row.items()
            if key and value and value.strip()
        )
        sentence = clean_text(sentence)
        if len(sentence) > MIN_CSV_ROW_CHARS:
            documents.append(
                Document(
                    page_content=sentence,
                    metadata={"source": display_name, "row": index},
                )
            )

    return split_text(documents)
