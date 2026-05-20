import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# paths
DATASET = "dataset.txt"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.json"

# using all-MiniLM-L6-v2 - lightweight and good enough for this use case
# referred from sentence-transformers docs
MODEL_NAME = "all-MiniLM-L6-v2"


def read_and_chunk(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # splitting by double newline because each paragraph talks about one topic
    # tried sentence-level splitting first but paragraphs gave better retrieval
    raw_chunks = content.split("\n\n")
    chunks = [c.strip() for c in raw_chunks if c.strip()]
    return chunks


def create_faiss_index(chunks, model):
    print("generating embeddings for", len(chunks), "chunks")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    # normalize so we can use inner product as cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


if __name__ == "__main__":
    chunks = read_and_chunk(DATASET)
    print(f"total chunks created: {len(chunks)}")

    for i, ch in enumerate(chunks):
        print(f"\nchunk {i+1}: {ch[:100]}...")

    model = SentenceTransformer(MODEL_NAME)
    index = create_faiss_index(chunks, model)

    # save to disk so we dont have to recompute every time
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("\ndone. index and chunks saved.")
