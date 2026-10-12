# 🧠 Multi-Source RAG System

A Retrieval-Augmented Generation system that ingests PDFs, web pages, and CSVs, and answers
questions with source citations — built around a **measured** retrieval pipeline rather than an
assumed one.

**Headline result: recall@5 improved from 0.533 to 0.800 (+50% relative)** over a standard
semantic-search baseline, measured on a 30-question labelled evaluation set.

---

## 📊 Results

Four retrieval configurations, evaluated on identical questions against an 829-chunk corpus.
Each question has exactly one known-correct source chunk, so recall@k is the share of questions
whose correct chunk appears in the top k results.

| Configuration | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Semantic only (bi-encoder) | 0.400 | 0.467 | 0.533 | 0.600 | 0.460 | 0.494 |
| BM25 only (keyword) | 0.567 | 0.633 | 0.667 | 0.700 | 0.606 | 0.628 |
| Hybrid + RRF | 0.333 | 0.567 | 0.600 | 0.767 | 0.468 | 0.539 |
| **Hybrid + RRF + cross-encoder rerank** | **0.633** | **0.733** | **0.800** | **0.833** | **0.700** | **0.732** |

| Configuration | Median latency | p95 |
| --- | --- | --- |
| Semantic only | 5 ms | 6 ms |
| BM25 only | 6 ms | 6 ms |
| Hybrid + RRF | 11 ms | 12 ms |
| Hybrid + RRF + rerank | 139 ms | 278 ms |

Reproduce with `python -m src.eval.evaluate`.

### What the numbers actually say

**Fusion alone made retrieval worse, and that was the most useful thing I learned.**

Reciprocal rank fusion dropped recall@1 to 0.333 — below both individual retrievers. Blending a
weaker ranker (semantic, 0.400) into a stronger one (BM25, 0.567) diluted the top of the list.

But recall@10 went *up*, to 0.767 — the best of any configuration without reranking. So fusion was
doing something valuable: surfacing correct chunks that neither retriever alone found, just placing
them badly. RRF's real job here is **recall, not precision**.

That reframes the pipeline. Fusion widens the candidate pool; the cross-encoder reorders it. Neither
stage is sufficient alone — reranking a narrow pool has little to work with, and fusing without
reranking leaves good chunks buried at rank 8. Together they reach 0.800 recall@5, and recall@1
nearly doubles from RRF's 0.333 to 0.633.

**BM25 beat semantic search on every metric.** Partly a property of the corpus — technical NLP
articles are dense with precise terminology ("BM25", "cross-attention", "cosine similarity") where
exact matching works well, and where a 384-dimension MiniLM embedding blurs distinctions. Partly a
property of the evaluation: see the caveat below.

**Reranking costs 133 ms.** Acceptable here, since LLM generation dominates end-to-end latency
anyway. At higher query volume the honest tradeoff would be reranking a smaller candidate pool.

### Honest caveats

- Questions were LLM-generated from the source chunks and so share vocabulary with them, which
  favours keyword matching. **BM25's lead over semantic search is probably overstated.** The
  comparison between configurations is still valid — all four were scored on identical questions.
- 30 questions is a small sample. Differences under ~5 points are within noise; the 27-point gap
  between baseline and full pipeline is not.
- 5 of 30 questions had their correct chunk missed entirely by every configuration, which caps
  recall@10 at 0.833. Those are chunking failures, not retrieval failures — the answer was split
  across a chunk boundary.
- The generated set was filtered to remove bibliography and boilerplate questions
  (`scripts/filter_golden_set.py`); the unfiltered set is kept at `eval/golden_set.raw.json`.

---

## 🏗️ Architecture

```
User query
    ↓
┌─────────────────────────────────────────┐
│  Semantic search          BM25 search   │   20 candidates each
│  (all-MiniLM-L6-v2)       (rank_bm25)   │
└─────────────────────────────────────────┘
    ↓                            ↓
    └──────────┬─────────────────┘
               ↓
    Reciprocal Rank Fusion            score = Σ 1/(60 + rank)
               ↓                       ~31 unique candidates
    Cross-encoder rerank              ms-marco-MiniLM-L-6-v2
               ↓                       top 6
    Llama 3.1 8B (Groq)
               ↓
    Answer + source citations
```

Ingestion (PDF via PyMuPDF, web via BeautifulSoup, CSV row-to-sentence) → cleaning → 700-char
chunks with 50-char overlap → HuggingFace embeddings → Chroma.

---

## 🚀 Features

- **Multi-source ingestion** — PDFs, web pages, and CSVs in one index, with per-source attribution
- **Hybrid retrieval** — dense vector search and BM25 keyword search, fused with RRF
- **Cross-encoder reranking** — joint (query, chunk) scoring on the fused candidate pool
- **Measured, not assumed** — every configuration benchmarked; see Results
- **Retrieval trace in the UI** — each answer shows candidates → fused → reranked, with timings
- **Source citations** on every answer
- **UMAP embedding visualisation** — interactive view of semantic clustering by source
- **Reproducible corpus** — `scripts/ingest.py --preset nlp` rebuilds the exact evaluated index

---

## 🛠️ Tech stack

| Component | Choice | Why |
| --- | --- | --- |
| Embeddings | all-MiniLM-L6-v2 | 384-dim, runs on CPU, good quality/speed tradeoff |
| Vector store | Chroma | Local persistence, no server to run |
| Keyword search | rank_bm25 | Complements dense retrieval on exact terminology |
| Fusion | Reciprocal Rank Fusion (k=60) | Rank-based, so no score normalisation needed |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder; ~22M params, CPU-viable |
| LLM | Llama 3.1 8B via Groq | Fast inference, generous free tier |
| UI | Streamlit | Fast to build, easy to deploy |

---

## ⚙️ Setup

```bash
git clone https://github.com/ayushii89/multi-source-rag-system
cd multi-source-rag-system

conda create -n rag python=3.11 -y && conda activate rag
pip install -r requirements.txt

echo "GROQ_API_KEY=your_key_here" > .env    # free key: https://console.groq.com
```

Python 3.11 is recommended — chromadb and the ML stack have patchier wheel coverage on 3.13.

**Run the app:**

```bash
streamlit run app.py
```

**Or build the index from the command line:**

```bash
python scripts/ingest.py --preset nlp --reset          # the evaluated corpus
python scripts/ingest.py --pdf paper.pdf --url https://example.com --csv data.csv
```

---

## 🧪 Evaluation

```bash
python scripts/ingest.py --preset nlp --reset
python -m src.eval.golden_set --n 40 --out eval/golden_set.json   # LLM-generated questions
python scripts/filter_golden_set.py eval/golden_set.json --apply  # drop citation/boilerplate items
python -m src.eval.golden_set --review eval/golden_set.json       # optional manual pass
python -m src.eval.evaluate                                       # writes eval/results.md
```

Unit tests cover RRF scoring against hand-computed values, deduplication, tie-break determinism,
and every metric:

```bash
pytest tests/ -v      # 35 tests
```

---

## 📁 Project structure

```
app.py                      # Streamlit UI
src/
├── ingestion/
│   ├── clean.py            # shared text normalisation
│   ├── loaders.py          # PDF / web / CSV → chunked Documents
│   └── store.py            # vector store + batched insertion
├── retrieval/
│   ├── fusion.py           # reciprocal rank fusion
│   ├── rerank.py           # cross-encoder reranking
│   └── hybrid.py           # configurable pipeline (the 4 evaluated modes)
├── generation/
│   └── chain.py            # prompt, LLM, context formatting
├── viz/
│   └── embeddings.py       # UMAP projection
└── eval/
    ├── golden_set.py       # labelled question generation
    ├── metrics.py          # recall@k, MRR, nDCG
    └── evaluate.py         # ablation runner

scripts/
├── ingest.py               # CLI corpus builder
└── filter_golden_set.py    # quality filter for generated questions

experiments/
└── sentence_window.py      # not wired into the pipeline

tests/
├── test_fusion.py          # RRF scoring and retrieval metrics
└── test_store.py           # batched insertion
eval/results.md             # generated results
```

The app, the CLI, and the evaluation all import the same ingestion and retrieval code, so measured
numbers describe the pipeline that actually runs.

---

## 🔭 Known limitations

- `clean_text` strips non-ASCII characters, so **non-English corpora are not supported**. Its
  camelCase-splitting heuristic also mangles some legitimate identifiers ("iPhone" → "i Phone") —
  a deliberate tradeoff, since PDF whitespace loss is the more common problem here.
- Fixed 700-character chunking. The 5 questions no configuration could answer are answers split
  across chunk boundaries; semantic or sentence-window chunking would likely recover them.
- BM25 rebuilds in memory whenever the corpus grows, which won't scale past a few tens of thousands
  of chunks.
- Chroma rejects single upserts above ~5,461 records, so inserts are batched (`src/ingestion/store.py`).
  Large PDFs exceed this easily — a 600-page document produced 6,120 chunks.
- Sentence-window retrieval is implemented in `experiments/sentence_window.py` but is **not**
  wired into the pipeline — it's an experiment, not a feature.

## Next steps

- Chunking ablation (fixed vs. semantic vs. sentence-window) against the same evaluation set
- Answer-level metrics: faithfulness and citation accuracy, not just retrieval
- Larger golden set (100+ questions) to tighten the confidence intervals
- Query rewriting for multi-hop questions

---

Built by Prateek Raj
