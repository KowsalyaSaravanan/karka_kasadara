"""
ingest.py
---------
Loads dataset.txt, splits into chunks, generates embeddings,
and stores them in a FAISS index saved to disk.

Chunking Strategy:
  - Split by double newline (paragraph-level chunks).
  - Each paragraph is a natural semantic unit in this dataset.
  - Skip empty lines.
  - This keeps related sentences together and avoids mid-sentence splits.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATASET_PATH = "dataset.txt"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.json"
EMBED_MODEL = "all-MiniLM-L6-v2"  # free, lightweight, runs locally


def load_and_chunk(path: str) -> list[str]:
    """Load text file and split into paragraph-level chunks."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on blank lines (paragraph boundaries)
    chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
    return chunks


def build_index(chunks: list[str], model: SentenceTransformer):
    """Generate embeddings and build FAISS index."""
    print(f"Encoding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine after normalization
    index.add(embeddings)
    return index


def main():
    chunks = load_and_chunk(DATASET_PATH)
    print(f"Total chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---\n{c[:120]}...")

    model = SentenceTransformer(EMBED_MODEL)
    index = build_index(chunks, model)

    # Save index and chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nIndex saved to {INDEX_PATH}")
    print(f"Chunks saved to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
