"""
app.py
------
FastAPI backend exposing the LangGraph RAG agent via REST endpoints.

Endpoints:
  POST /query   - ask a question, get answer + retrieved chunks
  GET  /health  - health check
  GET  /chunks  - list all indexed chunks (useful for debugging)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agent import run_query

app = FastAPI(
    title="Velammal RAG API",
    description="RAG system powered by LangGraph + Gemini Flash + FAISS",
    version="1.0.0",
)


# ── request / response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, example="What is the admission process?")


class ChunkResult(BaseModel):
    rank:  int
    score: float
    chunk: str


class QueryResponse(BaseModel):
    query:            str
    retrieved_chunks: list[ChunkResult]
    answer:           str


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Check if the API is running."""
    return {"status": "ok", "message": "Velammal RAG API is running"}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(request: QueryRequest):
    """
    Submit a question and get an answer grounded in the Velammal dataset.

    - Retrieves top-3 relevant chunks from FAISS
    - Generates answer using Gemini Flash (no hallucination, no external knowledge)
    """
    try:
        result = run_query(request.query)
        return QueryResponse(
            query=result["query"],
            retrieved_chunks=[ChunkResult(**c) for c in result["retrieved_chunks"]],
            answer=result["answer"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks", tags=["Debug"])
def list_chunks():
    """Return all indexed chunks with their index numbers."""
    import json
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {"total": len(chunks), "chunks": [{"index": i, "text": c} for i, c in enumerate(chunks)]}
