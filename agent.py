"""
agent.py
--------
LangGraph-based RAG agent.

Graph nodes:
  1. retrieve  - embed query, search FAISS, return top-3 chunks
  2. generate  - build prompt from chunks, call Gemini Flash, return answer

State flows: START -> retrieve -> generate -> END
"""

import os
import json
from typing import TypedDict

import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ── constants ────────────────────────────────────────────────────────────────
INDEX_PATH  = "faiss_index.bin"
CHUNKS_PATH = "chunks.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 3

# ── shared resources (loaded once at import) ─────────────────────────────────
_embed_model = SentenceTransformer(EMBED_MODEL)
_index       = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "r", encoding="utf-8") as _f:
    _chunks: list[str] = json.load(_f)

_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
)


# ── graph state ──────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    query:            str
    retrieved_chunks: list[dict]   # [{rank, score, chunk}]
    answer:           str


# ── node 1: retrieve ─────────────────────────────────────────────────────────
def retrieve(state: RAGState) -> RAGState:
    """Embed the query and fetch top-K chunks from FAISS."""
    q_emb = _embed_model.encode([state["query"]], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = _index.search(q_emb, TOP_K)

    chunks = [
        {"rank": int(i + 1), "score": round(float(scores[0][i]), 4), "chunk": _chunks[int(indices[0][i])]}
        for i in range(TOP_K)
    ]
    return {**state, "retrieved_chunks": chunks}


# ── node 2: generate ─────────────────────────────────────────────────────────
def generate(state: RAGState) -> RAGState:
    """Build a grounded prompt and call Gemini Flash."""
    context = "\n\n".join(c["chunk"] for c in state["retrieved_chunks"])
    prompt = (
        "You are a helpful assistant for Velammal School.\n"
        "Answer the question using ONLY the context provided below.\n"
        "Do NOT use any outside knowledge.\n"
        "If the answer is not in the context, say: "
        "'I don't have enough information to answer that.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\n"
        "Answer:"
    )
    response = _llm.invoke([HumanMessage(content=prompt)])
    return {**state, "answer": response.content.strip()}


# ── build graph ──────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


# single compiled graph instance
rag_graph = build_graph()


def run_query(query: str) -> dict:
    """Run the full RAG graph for a given query."""
    initial_state: RAGState = {
        "query": query,
        "retrieved_chunks": [],
        "answer": "",
    }
    result = rag_graph.invoke(initial_state)
    return result
