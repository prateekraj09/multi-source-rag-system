#!/usr/bin/env python3
"""Drop low-quality examples from a generated golden set.

Wikipedia pages end in long reference lists, and chunks drawn from them produce
questions like "what is the title of the 2014 paper by Vinyals et al.?" Those
are bad evaluation items for two reasons: they test bibliographic lookup rather
than comprehension, and reference lists are dense with rare proper nouns, which
inflates BM25's apparent performance relative to semantic search. Leaving them
in would bias the very comparison the eval exists to make.

This applies mechanical heuristics only -- it flags, you confirm.

Usage:
    python scripts/filter_golden_set.py eval/golden_set.json
    python scripts/filter_golden_set.py eval/golden_set.json --apply
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Markers that appear in citation blocks and almost nowhere in prose.
CITATION_MARKERS = (
    "Retrieved ",
    "doi:",
    "ISBN",
    "arXiv:",
    "ISSN",
    "PMID",
    ". pp. ",
    "Archived from the original",
)

# Question phrasings that indicate a bibliographic rather than factual question.
BIBLIOGRAPHIC_PATTERNS = (
    r"\btitle of the\b.*\b(article|paper|book|publication|study)\b",
    r"\bwho (wrote|authored|published)\b",
    r"\bwhat (journal|conference|publisher)\b",
    r"\bwhen was.*\b(retrieved|accessed)\b",
    r"\bpublished (in|on|by)\b.*\b(19|20)\d{2}\b",
    r"\bet al\b",
    r"\baccording to the (reference|citation)\b",
)

# Navigation and UI chrome scraped alongside article text.
BOILERPLATE_MARKERS = (
    "Jump to content",
    "Main menu",
    "Privacy policy",
    "Creative Commons Attribution-ShareAlike",
    "Toggle the table of contents",
    "From Wikipedia, the free encyclopedia",
)


def citation_density(text: str) -> float:
    """Share of lines that look like bibliography entries."""
    marker_hits = sum(text.count(marker) for marker in CITATION_MARKERS)
    # Normalise per 500 characters so long chunks aren't penalised unfairly.
    return marker_hits / max(len(text) / 500, 1)


def diagnose(example: dict) -> list[str]:
    """Return reasons this example looks unusable. Empty means keep."""
    reasons = []
    question = example["question"]
    chunk = example["chunk_text"]

    density = citation_density(chunk)
    if density >= 2.0:
        reasons.append(f"chunk is a reference list (citation density {density:.1f})")

    for pattern in BIBLIOGRAPHIC_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            reasons.append("bibliographic question")
            break

    if any(marker in chunk for marker in BOILERPLATE_MARKERS):
        reasons.append("chunk contains page boilerplate")

    # A question with no content words in common with its chunk usually means
    # the generator drifted onto a topic the chunk only alludes to.
    question_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z]{5,}\b", question)
        if word.lower() not in {"which", "there", "these", "those", "about", "would"}
    }
    if question_words:
        chunk_lower = chunk.lower()
        overlap = sum(1 for word in question_words if word in chunk_lower)
        if overlap / len(question_words) < 0.25:
            reasons.append(
                f"question shares little vocabulary with its chunk "
                f"({overlap}/{len(question_words)} terms)"
            )

    if len(chunk.strip()) < 250:
        reasons.append(f"chunk very short ({len(chunk.strip())} chars)")

    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a golden set.")
    parser.add_argument("path", help="path to golden_set.json")
    parser.add_argument("--apply", action="store_true", help="write the filtered set")
    args = parser.parse_args()

    path = Path(args.path)
    examples = json.loads(path.read_text(encoding="utf-8"))

    kept, dropped = [], []
    for example in examples:
        reasons = diagnose(example)
        (dropped if reasons else kept).append((example, reasons))

    print(f"{len(examples)} examples: {len(kept)} keep, {len(dropped)} flagged\n")

    if dropped:
        print("FLAGGED")
        for example, reasons in dropped:
            print(f"  Q: {example['question'][:88]}")
            for reason in reasons:
                print(f"     - {reason}")
        print()

    if not args.apply:
        print(f"Dry run. Re-run with --apply to keep only the {len(kept)} clean examples.")
        return

    if len(kept) < 20:
        print(
            f"Refusing to write: only {len(kept)} examples would remain, which is too "
            "few to measure anything. Generate a larger set first with:\n"
            "  python -m src.eval.golden_set --n 80 --out eval/golden_set.json"
        )
        return

    backup = path.with_suffix(".raw.json")
    if not backup.exists():
        backup.write_text(json.dumps(examples, indent=2), encoding="utf-8")
        print(f"Backed up original to {backup}")

    path.write_text(
        json.dumps([example for example, _ in kept], indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(kept)} examples to {path}")
    print("\nNext:\n  python -m src.eval.evaluate")


if __name__ == "__main__":
    main()
