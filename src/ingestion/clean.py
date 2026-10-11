"""Text cleaning for ingested documents.

Previously copy-pasted into five separate day*.py files. Hoisted here so the
Streamlit app, the CLI ingester, and the evaluation pipeline all normalise text
identically -- if they didn't, evaluation numbers wouldn't describe the system
the app actually runs.
"""

from __future__ import annotations

import re


def clean_text(text: str, ascii_only: bool = True) -> str:
    """Normalise text extracted from PDFs and web pages.

    PDF extraction produces characteristic damage this repairs: missing spaces
    at letter/digit boundaries, doubled words, and run-together camelCase from
    lost whitespace.

    Args:
        text: Raw extracted text.
        ascii_only: Strip non-ASCII characters. Default True preserves the
            original behaviour, but note this **deletes all non-Latin script** --
            Devanagari, CJK, Cyrillic, Arabic are removed entirely, and accented
            Latin characters are dropped. Pass False for non-English corpora.

    Returns:
        Cleaned, whitespace-normalised text.

    Note:
        The camelCase split is a heuristic and will incorrectly break
        legitimate identifiers ("iPhone" -> "i Phone") and some acronyms. It's
        kept because PDF whitespace loss is the more common problem in this
        corpus, but it is a known tradeoff rather than a correct solution.
    """
    if ascii_only:
        text = re.sub(r"[^\x00-\x7F]+", " ", text)

    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\b(\w{4,})\1\b", r"\1", text)  # collapse doubled words
    text = re.sub(r"(https?://)", r" \1", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
