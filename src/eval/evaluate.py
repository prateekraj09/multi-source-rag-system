"""Run the retrieval ablation and emit a README-ready results table.

Evaluates every retriever configuration against the same golden set, so the
comparison is apples to apples:

    semantic      -- bi-encoder vector search only (the baseline)
    bm25          -- keyword search only
    rrf           -- both, fused with reciprocal rank fusion
    rrf_rerank    -- fused, then reordered by a cross-encoder

Usage:
    python -m src.eval.evaluate
    python -m src.eval.evaluate --modes semantic rrf_rerank --out eval/results.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from ..retrieval.hybrid import ALL_MODES, HybridRetriever, RetrievalConfig
from .golden_set import chunk_identity, load
from .metrics import aggregate, find_rank, format_table


def get_vector_store(persist_dir: str = "chroma_index"):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def evaluate_config(db, config: RetrievalConfig, examples: list, verbose: bool = True) -> dict:
    """Run one configuration over the whole golden set.

    Returns aggregate metrics plus median and p95 latency. Documents are
    matched by content hash rather than object identity, because BM25 and
    Chroma return different Document instances for the same chunk.
    """
    retriever = HybridRetriever(db, config)
    ranks: list[int | None] = []
    latencies: list[float] = []

    for index, example in enumerate(examples, start=1):
        if verbose:
            print(f"  [{index}/{len(examples)}] {config.mode}...", end="\r")

        started = time.perf_counter()
        documents = retriever.retrieve(example.question)
        latencies.append((time.perf_counter() - started) * 1000)

        retrieved_ids = [chunk_identity(doc.page_content) for doc in documents]
        ranks.append(find_rank(retrieved_ids, example.chunk_id))

    metrics = aggregate(ranks)
    metrics["median_latency_ms"] = round(statistics.median(latencies), 1)
    metrics["p95_latency_ms"] = round(
        sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)], 1
    )
    metrics["mode"] = config.mode
    metrics["label"] = config.label

    if verbose:
        print(
            f"  {config.label:<42} "
            f"recall@5={metrics['recall@5']:.3f}  "
            f"MRR={metrics['mrr']:.3f}  "
            f"{metrics['median_latency_ms']:.0f}ms"
        )
    return metrics


def run(
    golden_set_path: str = "eval/golden_set.json",
    persist_dir: str = "chroma_index",
    modes: tuple[str, ...] = ALL_MODES,
    final_k: int = 10,
    candidate_k: int = 20,
) -> list[dict]:
    """Evaluate each mode and return their metrics.

    final_k defaults to 10 so recall@10 is measurable; the app itself returns
    fewer. Retrieving more than you display costs nothing at eval time and
    makes the metric curve visible.
    """
    path = Path(golden_set_path)
    if not path.exists():
        sys.exit(
            f"No golden set at '{path}'. Create one first:\n"
            f"  python -m src.eval.golden_set --n 40 --out {path}"
        )

    examples = load(str(path))
    print(f"Loaded {len(examples)} evaluation questions from {path}")

    if len(examples) < 20:
        print(
            f"  warning: {len(examples)} questions is a small sample; "
            "differences of a few points will be noise. Aim for 40+."
        )

    db = get_vector_store(persist_dir)
    stored = db.get().get("documents") or []
    print(f"Corpus: {len(stored)} chunks in '{persist_dir}'\n")

    results = []
    for mode in modes:
        config = RetrievalConfig(mode=mode, final_k=final_k, candidate_k=candidate_k)
        results.append(evaluate_config(db, config, examples))

    return results


def write_report(results: list[dict], out_path: str, golden_set_path: str) -> None:
    """Write the markdown report, including the delta over baseline."""
    table = format_table([(r["label"], r) for r in results])

    baseline = next((r for r in results if r["mode"] == "semantic"), results[0])
    best = max(results, key=lambda r: r["recall@5"])

    lines = [
        "## Retrieval evaluation",
        "",
        f"Evaluated on {baseline['n_queries']} questions generated from the indexed "
        "corpus and manually reviewed. Each question has exactly one known-correct "
        "source chunk, so recall@k is the share of questions whose correct chunk "
        "appears in the top k results.",
        "",
        table,
        "",
        "### Latency",
        "",
        "| Configuration | Median | p95 |",
        "| --- | --- | --- |",
    ]

    for result in results:
        lines.append(
            f"| {result['label']} | {result['median_latency_ms']:.0f} ms "
            f"| {result['p95_latency_ms']:.0f} ms |"
        )

    if best["mode"] != baseline["mode"]:
        delta = best["recall@5"] - baseline["recall@5"]
        relative = (delta / baseline["recall@5"] * 100) if baseline["recall@5"] else float("inf")
        slowdown = best["median_latency_ms"] - baseline["median_latency_ms"]
        lines += [
            "",
            "### Takeaway",
            "",
            f"**{best['label']}** improves recall@5 from **{baseline['recall@5']:.3f}** "
            f"to **{best['recall@5']:.3f}** ({relative:+.0f}% relative) over the "
            f"semantic-only baseline, at a cost of {slowdown:+.0f} ms median latency.",
            "",
            "Caveat: questions were LLM-generated from the source chunks, so they "
            "share vocabulary with the text they came from. This favours keyword "
            "matching, and the absolute numbers are optimistic relative to real "
            "user queries. The comparison between configurations remains valid "
            "since all four are scored on the identical question set.",
        ]

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps({"golden_set": golden_set_path, "results": results}, indent=2),
        encoding="utf-8",
    )

    print(f"\nWrote {output} and {json_path}")
    print("\n" + "\n".join(lines[:8]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval configurations.")
    parser.add_argument("--golden-set", default="eval/golden_set.json")
    parser.add_argument("--persist-dir", default="chroma_index")
    parser.add_argument("--out", default="eval/results.md")
    parser.add_argument("--modes", nargs="+", default=list(ALL_MODES), choices=list(ALL_MODES))
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=20)
    args = parser.parse_args()

    results = run(
        golden_set_path=args.golden_set,
        persist_dir=args.persist_dir,
        modes=tuple(args.modes),
        final_k=args.final_k,
        candidate_k=args.candidate_k,
    )
    write_report(results, args.out, args.golden_set)


if __name__ == "__main__":
    main()
