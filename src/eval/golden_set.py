"""Generate a labelled evaluation set from the indexed corpus.

For each sampled chunk we ask an LLM to write a question that the chunk answers.
The chunk is then the known-correct retrieval target for that question, which
gives us ground truth for recall@k and MRR without hand-labelling everything.

This is standard practice, but it has a real limitation worth stating out loud
(and worth stating in an interview): questions generated *from* a chunk tend to
reuse its vocabulary, which flatters keyword retrieval. Treat the absolute
numbers as a benchmark for comparing configurations, not as an estimate of
production accuracy. The --review flag exists so you can edit or delete weak
questions before trusting the set.

Usage:
    python -m src.eval.golden_set --n 40 --out eval/golden_set.json
    python -m src.eval.golden_set --review eval/golden_set.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GENERATION_PROMPT = """You are building an evaluation set for a document retrieval system.

Below is a passage from a document. Write ONE specific question that this passage
answers.

Requirements:
- The question must be answerable using ONLY this passage.
- It must be specific enough that only this passage answers it. Avoid questions
  so generic that any passage from the document would do.
- Do not use phrases like "the passage", "the text", or "this document" -- write
  it the way a real user searching a knowledge base would type it.
- Prefer questions about concrete facts, definitions, numbers, or named entities.
- If the passage is boilerplate (a table of contents, a header, navigation text,
  copyright notices) and contains no substantive content, reply with exactly:
  SKIP

Passage:
\"\"\"{chunk}\"\"\"

Reply with the question only, no preamble."""


@dataclass
class GoldenExample:
    """One evaluation question with its known-correct source chunk."""

    question: str
    chunk_id: str
    chunk_text: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_identity(text: str) -> str:
    """Stable id for a chunk, so labels survive reindexing.

    Content-hashed rather than positional: Chroma ids change when you rebuild
    the index, which would silently invalidate every label.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_corpus(persist_dir: str = "chroma_index") -> list[dict]:
    """Read all chunks out of the Chroma index."""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    stored = db.get()

    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or [{}] * len(documents)

    return [
        {
            "text": text,
            "chunk_id": chunk_identity(text),
            "source": (metadatas[i] or {}).get("source", "unknown"),
        }
        for i, text in enumerate(documents)
    ]


def is_usable(chunk: dict, min_chars: int = 200) -> bool:
    """Filter chunks too short or too boilerplate to yield a fair question."""
    text = chunk["text"].strip()
    if len(text) < min_chars:
        return False
    # Mostly-digits chunks (stray tables, page numbers) make poor questions.
    alpha = sum(character.isalpha() for character in text)
    return alpha / max(len(text), 1) >= 0.5


def stratified_sample(chunks: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample evenly across sources so one large PDF cannot dominate."""
    rng = random.Random(seed)

    by_source: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk["source"], []).append(chunk)

    for bucket in by_source.values():
        rng.shuffle(bucket)

    selected: list[dict] = []
    sources = sorted(by_source)
    position = 0
    # Round-robin across sources until we have n or exhaust the corpus.
    while len(selected) < n and any(len(by_source[s]) > position for s in sources):
        for source in sources:
            if len(selected) >= n:
                break
            bucket = by_source[source]
            if len(bucket) > position:
                selected.append(bucket[position])
        position += 1

    return selected


def generate_question(llm, chunk_text: str) -> str | None:
    """Ask the LLM for one question, or None if the chunk is unusable."""
    response = llm.invoke(GENERATION_PROMPT.format(chunk=chunk_text[:2000]))
    question = getattr(response, "content", str(response)).strip()

    if not question or question.upper().startswith("SKIP"):
        return None

    # Models often wrap the question in quotes or prefix it with "Question:".
    question = re.sub(r"^(question|q)\s*[:\-]\s*", "", question, flags=re.IGNORECASE)
    question = question.strip().strip('"').strip()

    if len(question) < 15 or "?" not in question:
        return None
    return question


def build_golden_set(
    n: int = 40,
    persist_dir: str = "chroma_index",
    model: str = "llama-3.1-8b-instant",
    seed: int = 42,
) -> list[GoldenExample]:
    """Generate the evaluation set end to end."""
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set. Add it to your .env file.")

    print(f"Loading corpus from '{persist_dir}'...")
    corpus = load_corpus(persist_dir)
    if not corpus:
        sys.exit(
            f"No chunks found in '{persist_dir}'. Ingest some documents first "
            "by running the Streamlit app."
        )

    usable = [chunk for chunk in corpus if is_usable(chunk)]
    print(f"  {len(corpus)} chunks total, {len(usable)} usable for question generation")

    # Oversample: some chunks will be SKIPped or rejected.
    candidates = stratified_sample(usable, n=int(n * 1.5), seed=seed)

    # temperature=0 keeps the set reproducible across runs.
    llm = ChatGroq(model=model, api_key=api_key, temperature=0.0)

    examples: list[GoldenExample] = []
    for index, chunk in enumerate(candidates, start=1):
        if len(examples) >= n:
            break
        print(f"  [{len(examples)}/{n}] generating from chunk {index}...", end="\r")

        try:
            question = generate_question(llm, chunk["text"])
        except Exception as error:  # noqa: BLE001 - one bad chunk shouldn't abort
            print(f"\n  warning: generation failed for chunk {index}: {error}")
            continue

        if question is None:
            continue

        examples.append(
            GoldenExample(
                question=question,
                chunk_id=chunk["chunk_id"],
                chunk_text=chunk["text"],
                source=chunk["source"],
            )
        )

    print(f"\nGenerated {len(examples)} questions.")
    if len(examples) < n:
        print(
            f"  note: asked for {n} but the corpus yielded {len(examples)}. "
            "Ingest more documents for a larger set."
        )
    return examples


def save(examples: list[GoldenExample], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([example.to_dict() for example in examples], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(examples)} examples to {output}")


def load(path: str) -> list[GoldenExample]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenExample(**item) for item in raw]


def review(path: str) -> None:
    """Walk the set interactively, keeping or dropping each question.

    Ten minutes here is the difference between numbers you can defend in an
    interview and numbers you can't.
    """
    examples = load(path)
    kept: list[GoldenExample] = []

    print(f"Reviewing {len(examples)} examples. [k]eep (default) / [d]rop / [e]dit / [q]uit\n")
    for index, example in enumerate(examples, start=1):
        print(f"--- {index}/{len(examples)} | source: {example.source}")
        print(f"Q: {example.question}")
        print(f"Chunk: {example.chunk_text[:250]}...\n")

        choice = input("> ").strip().lower()
        if choice == "q":
            kept.extend(examples[index - 1 :])
            break
        if choice == "d":
            continue
        if choice == "e":
            edited = input("New question: ").strip()
            if edited:
                example.question = edited
        kept.append(example)
        print()

    save(kept, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RAG evaluation set.")
    parser.add_argument("--n", type=int, default=40, help="number of questions")
    parser.add_argument("--out", default="eval/golden_set.json", help="output path")
    parser.add_argument("--persist-dir", default="chroma_index", help="Chroma directory")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Groq model")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    parser.add_argument("--review", metavar="PATH", help="review an existing set")
    args = parser.parse_args()

    if args.review:
        review(args.review)
        return

    examples = build_golden_set(
        n=args.n,
        persist_dir=args.persist_dir,
        model=args.model,
        seed=args.seed,
    )
    save(examples, args.out)
    print(f"\nNext: review the questions, then run the evaluation:\n"
          f"  python -m src.eval.golden_set --review {args.out}\n"
          f"  python -m src.eval.evaluate --golden-set {args.out}")


if __name__ == "__main__":
    main()
