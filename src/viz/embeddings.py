"""UMAP projection of the embedding space.

Chunks are embedded in 384 dimensions; UMAP projects them to 2 so the corpus
structure is visible. Chunks from the same source cluster together, and
semantically related content from different sources sits adjacent -- which is
a direct visual check that the embeddings are capturing meaning rather than
surface form.
"""

from __future__ import annotations

from urllib.parse import urlparse

import numpy as np
import plotly.graph_objects as go

# UMAP is stochastic; the fixed seed keeps the plot stable across reruns so
# users don't see the layout jump on every interaction.
RANDOM_STATE = 42

PALETTE = ["#8b5cf6", "#a78bfa", "#6b3fa0", "#c4b5fd", "#7c3aed", "#ddd6fe"]

MIN_POINTS = 5


def clean_source(source: str) -> str:
    """Shorten a source path or URL to a readable label."""
    if not source:
        return "unknown"
    if source.startswith("http"):
        return urlparse(source).netloc
    if source.startswith("/var") or source.startswith("/tmp"):
        return source.split("/")[-1]
    return source


def reduce_dimensions(matrix: np.ndarray, n_neighbors: int | None = None) -> np.ndarray:
    """Project embeddings to 2D with UMAP.

    n_neighbors controls the local/global tradeoff and must stay below the
    point count, so it's clamped for small corpora.
    """
    from umap import UMAP

    neighbors = n_neighbors or min(15, len(matrix) - 1)
    reducer = UMAP(
        n_components=2,
        n_neighbors=max(neighbors, 2),
        min_dist=0.1,
        metric="cosine",  # matches the retrieval similarity metric
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(matrix)


def build_embedding_figure(db) -> go.Figure | None:
    """Build the interactive scatter plot, or None if the corpus is too small."""
    data = db.get(include=["documents", "metadatas", "embeddings"])
    documents = data.get("documents") or []

    if len(documents) < MIN_POINTS:
        return None

    coords = reduce_dimensions(np.array(data["embeddings"]))
    sources = [clean_source((meta or {}).get("source", "unknown")) for meta in data["metadatas"]]
    unique_sources = sorted(set(sources))
    colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(unique_sources)}

    figure = go.Figure()
    for source in unique_sources:
        indices = [i for i, name in enumerate(sources) if name == source]
        figure.add_trace(
            go.Scatter(
                x=[coords[i][0] for i in indices],
                y=[coords[i][1] for i in indices],
                mode="markers",
                name=source,
                marker=dict(
                    size=9,
                    color=colors[source],
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
                text=[documents[i][:200] + "..." for i in indices],
                hovertemplate=(
                    f"<b>Source:</b> {source}<br><b>Chunk:</b><br>%{{text}}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=dict(
            text="Embedding Space — Semantic clustering of your knowledge base",
            font=dict(size=15, color="#2d1f5e", family="DM Serif Display"),
            x=0.5,
        ),
        paper_bgcolor="#f0ebff",
        plot_bgcolor="white",
        font=dict(color="#2d1f5e", family="DM Sans"),
        legend=dict(
            title="Sources",
            bgcolor="white",
            bordercolor="#e0d5ff",
            borderwidth=1,
            font=dict(color="#4a2c8a"),
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#e0d5ff", zeroline=False,
            title="UMAP Dim 1", color="#6b3fa0",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#e0d5ff", zeroline=False,
            title="UMAP Dim 2", color="#6b3fa0",
        ),
        height=550,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return figure
