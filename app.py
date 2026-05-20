import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import run_query

app = FastAPI(
    title="Velammal RAG API",
    description="RAG system for Velammal school dataset using LangGraph + FAISS + Gemini",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    query: str


class ChunkInfo(BaseModel):
    rank: int
    score: float
    chunk: str


class QueryResponse(BaseModel):
    query: str
    retrieved_chunks: list[ChunkInfo]
    answer: str


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}


@app.post("/query", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    try:
        result = run_query(request.query)
        return QueryResponse(
            query=result["query"],
            retrieved_chunks=[ChunkInfo(**c) for c in result["chunks"]],
            answer=result["answer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chunks")
def get_all_chunks():
    # useful to verify what got indexed
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {
        "total": len(chunks),
        "chunks": [{"index": i, "text": c} for i, c in enumerate(chunks)]
    }
