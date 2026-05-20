"""
rag.py
------
Core RAG logic:
  - Loads FAISS index + chunks from disk
  - Retrieves top-3 relevant chunks for a query
  - Generates answer using only retrieved chunks via Gemini Flash API
  - No hallucination: prompt strictly limits to provided context
"""

import json
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_API_KEY = "AIzaSyBLMWb472_mmbvFcAjl1gfB755cMGDgyBE"
TOP_K = 3


class RAGSystem:
    def __init__(self):
        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)

        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)

        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        print("Connecting to Gemini Flash...")
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini = genai.GenerativeModel(GEMINI_MODEL)

        print("RAG system ready.\n")

    def retrieve(self, query: str) -> list[tuple[int, float, str]]:
        """Return top-K (index, score, chunk) for the query."""
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, TOP_K)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append((int(idx), float(score), self.chunks[idx]))
        return results

    def generate_answer(self, query: str, retrieved_chunks: list[str]) -> str:
        """Generate answer strictly from retrieved chunks only using Gemini."""
        context = "\n\n".join(retrieved_chunks)
        prompt = (
            f"You are a helpful assistant. Answer the question using ONLY the context below.\n"
            f"Do NOT use any outside knowledge. If the answer is not in the context, say 'I don't know based on the provided information.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        response = self.gemini.generate_content(prompt)
        return response.text.strip()

    def query(self, user_query: str) -> dict:
        """Full RAG pipeline: retrieve + generate."""
        retrieved = self.retrieve(user_query)
        chunks_text = [r[2] for r in retrieved]
        answer = self.generate_answer(user_query, chunks_text)
        return {
            "query": user_query,
            "retrieved_chunks": [
                {"rank": i + 1, "score": round(r[1], 4), "chunk": r[2]}
                for i, r in enumerate(retrieved)
            ],
            "answer": answer,
        }
