# sentence_window.py — EXPERIMENT, not wired into the main pipeline.
# Sentence-window retrieval via LlamaIndex, kept for the chunking ablation
# listed in the README's next steps.

import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()


# ── Step 1: Setup embedding model ─────────────────────────────────────────────

def setup_settings():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.llm = None  # we use Groq separately
    print("   Embedding model loaded.")


# ── Step 2: Build sentence window index from PDF ──────────────────────────────

def build_sentence_window_index(pdf_path: str, window_size: int = 3):
    print(f"   Loading PDF: {pdf_path}")

    # Use PyMuPDF to extract text properly
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Convert to LlamaIndex document
    from llama_index.core import Document
    documents = [Document(text=full_text, metadata={"source": pdf_path})]

    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )

    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=node_parser,
        show_progress=True
    )

    print(f"   Index built with window_size={window_size}")
    return index

# ── Step 3: Query with sentence window retrieval ──────────────────────────────

def query_sentence_window(index, query: str, similarity_top_k: int = 3) -> str:
    # MetadataReplacementPostProcessor replaces retrieved sentence
    # with its surrounding window for richer context
    postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )

    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postprocessor],
        response_mode="no_text"  # we just want retrieved nodes, not LlamaIndex LLM
    )

    response = query_engine.query(query)

    # Extract window text from nodes
    windows = []
    for node in response.source_nodes:
        window_text = node.node.metadata.get("window", node.node.text)
        windows.append(window_text)

    return windows


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pdf_path = "sample.pdf"  # change to your PDF

    print("⚙️  Setting up...")
    setup_settings()

    print("📄 Building sentence window index...")
    index = build_sentence_window_index(pdf_path, window_size=3)

    print("\n🔍 Test sentence window retrieval!")
    query = input("Enter your question: ")

    print("\n⏳ Retrieving with sentence window...")
    windows = query_sentence_window(index, query)

    print("\n📌 Retrieved windows:\n")
    for i, w in enumerate(windows):
        print(f"--- Window {i+1} ---")
        print(w[:500])
        print()