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

INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.json"
TOP_K = 3

# load everything once when module is imported
# dont want to reload model on every request - too slow
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

faiss_index = faiss.read_index(INDEX_FILE)

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

# using gemini-2.5-flash - checked available models via list_models()
# temperature low so answers stay grounded
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2
)


# langgraph needs a typed state dict to pass between nodes
class AgentState(TypedDict):
    query: str
    chunks: list[dict]
    answer: str


def retrieve_node(state: AgentState):
    query = state["query"]

    # embed the query the same way we embedded the chunks
    q_vec = embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)

    scores, indices = faiss_index.search(q_vec, TOP_K)

    results = []
    for i in range(TOP_K):
        results.append({
            "rank": i + 1,
            "score": round(float(scores[0][i]), 4),
            "chunk": all_chunks[int(indices[0][i])]
        })

    return {**state, "chunks": results}


def generate_node(state: AgentState):
    # join top chunks as context
    context = "\n\n".join(item["chunk"] for item in state["chunks"])

    # strict prompt - only answer from context, no outside knowledge
    prompt = f"""You are a helpful assistant for Velammal School.
Answer the question using ONLY the context below.
Do not use any outside knowledge.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {state['query']}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {**state, "answer": response.content.strip()}


def build_rag_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# compile once at startup
rag_graph = build_rag_graph()


def run_query(user_query: str):
    state = {
        "query": user_query,
        "chunks": [],
        "answer": ""
    }
    result = rag_graph.invoke(state)
    return result
