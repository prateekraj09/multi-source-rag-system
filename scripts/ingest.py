#!/usr/bin/env python3
"""Populate the vector store from the command line.

Makes ingestion reproducible: anyone cloning the repo can rebuild the exact
index the evaluation numbers were measured on, instead of clicking through the
Streamlit sidebar and hoping they picked the same sources.

Usage:
    python scripts/ingest.py --pdf paper.pdf --url https://example.com
    python scripts/ingest.py --preset nlp --reset
    python scripts/ingest.py --csv data.csv --pdf a.pdf --pdf b.pdf
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow `python scripts/ingest.py` from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loaders import load_csv, load_pdf, load_url  # noqa: E402
from src.ingestion.store import add_documents, count_documents  # noqa: E402

# A ready-made corpus so the eval can be reproduced without hunting for
# documents. These are stable, substantial, and thematically coherent, which
# matters -- a corpus of unrelated documents makes retrieval look artificially
# easy, since any keyword overlap uniquely identifies the right source.
PRESETS = {
    "nlp": [
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
        "https://en.wikipedia.org/wiki/Word_embedding",
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://en.wikipedia.org/wiki/BERT_(language_model)",
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://en.wikipedia.org/wiki/Sentence_embedding",
        "https://en.wikipedia.org/wiki/Okapi_BM25",
        "https://en.wikipedia.org/wiki/Vector_database",
    ],
}


def get_vector_store(persist_dir: str):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Chroma index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Presets: {', '.join(PRESETS)}",
    )
    parser.add_argument("--pdf", action="append", default=[], metavar="PATH")
    parser.add_argument("--url", action="append", default=[], metavar="URL")
    parser.add_argument("--csv", action="append", default=[], metavar="PATH")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="ingest a bundled URL set")
    parser.add_argument("--persist-dir", default="chroma_index")
    parser.add_argument("--reset", action="store_true", help="delete the index first")
    args = parser.parse_args()

    urls = list(args.url)
    if args.preset:
        urls.extend(PRESETS[args.preset])

    if not (args.pdf or urls or args.csv):
        parser.error("nothing to ingest -- pass --pdf, --url, --csv, or --preset")

    if args.reset:
        index_path = Path(args.persist_dir)
        if index_path.exists():
            shutil.rmtree(index_path)
            print(f"Cleared {index_path}")

    print("Loading embedding model...")
    db = get_vector_store(args.persist_dir)

    total = 0
    failures: list[tuple[str, str]] = []

    jobs = (
        [("pdf", path, load_pdf) for path in args.pdf]
        + [("url", url, load_url) for url in urls]
        + [("csv", path, load_csv) for path in args.csv]
    )

    for kind, target, loader in jobs:
        label = target if len(target) <= 70 else target[:67] + "..."
        print(f"  [{kind}] {label}", end=" ... ", flush=True)
        try:
            chunks = loader(target)
            if not chunks:
                print("0 chunks (empty or all filtered)")
                failures.append((target, "produced no chunks"))
                continue
            add_documents(db, chunks)
            total += len(chunks)
            print(f"{len(chunks)} chunks")
        except Exception as error:  # noqa: BLE001 - one bad source shouldn't abort
            print(f"FAILED: {type(error).__name__}: {error}")
            failures.append((target, str(error)))

    stored = count_documents(db)
    print(f"\nIngested {total} chunks. Index now holds {stored} total.")

    if failures:
        print(f"\n{len(failures)} source(s) failed:")
        for target, reason in failures:
            print(f"  - {target}: {reason}")

    if stored < 200:
        print(
            "\nNote: fewer than 200 chunks. Evaluation on a small corpus is noisy "
            "-- retrieval looks easy when there is little to confuse it. "
            "Add more sources for numbers worth quoting."
        )
    else:
        print("\nNext:\n  python -m src.eval.golden_set --n 40 --out eval/golden_set.json")


if __name__ == "__main__":
    main()
