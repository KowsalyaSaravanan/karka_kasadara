# Velammal RAG System

A Retrieval-Augmented Generation (RAG) system that answers questions **strictly from the provided Velammal School dataset** — no hallucination, no external knowledge, no paid APIs beyond Gemini.

Built as part of the AI Platform Lead assignment.

---

## Architecture

```
User Query
    │
    ▼
[FastAPI /query endpoint]
    │
    ▼
[LangGraph Agent]
    │
    ├── Node 1: retrieve
    │     └── Embed query → FAISS search → Top-3 chunks
    │
    └── Node 2: generate
          └── Build grounded prompt → Gemini 2.5 Flash → Answer
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (free, local) |
| Vector Store | `FAISS` (cosine similarity, saved to disk) |
| Agent Framework | `LangGraph` (StateGraph with retrieve → generate nodes) |
| LLM | `Gemini 2.5 Flash` via `langchain-google-genai` |
| Backend API | `FastAPI` + `Uvicorn` |
| Package Manager | `uv` (Python 3.12) |

---

## Project Structure

```
rag_project/
├── dataset.txt          # Velammal school content (source of truth)
├── ingest.py            # Chunk + embed + build FAISS index
├── agent.py             # LangGraph RAG agent (retrieve + generate nodes)
├── app.py               # FastAPI backend (REST endpoints)
├── main.py              # CLI runner with 3 sample queries
├── rag.py               # Standalone RAG class (used before LangGraph)
├── requirements.txt     # All dependencies
├── pyproject.toml       # uv project config
└── README.md
```

---

## Chunking Strategy

- Split dataset by **double newline** (paragraph boundaries)
- Each paragraph is a natural semantic unit — related sentences stay together
- Avoids mid-sentence splits that break context
- Result: **12 clean chunks** from the dataset

---

## Setup & Run

### 1. Prerequisites
- Python 3.12
- `uv` installed → `pip install uv`

### 2. Clone & setup

```bash
git clone https://github.com/KowsalyaSaravanan/karka_kasadara.git
cd karka_kasadara
```

### 3. Create virtual environment

```bash
uv venv --python 3.12
```

### 4. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 5. Set API key

Create a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 6. Build FAISS index (run once)

```bash
.venv\Scripts\python.exe ingest.py        # Windows
# or
.venv/bin/python ingest.py                # Mac/Linux
```

### 7. Start FastAPI server

```bash
.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 8. Or run CLI with sample queries

```bash
.venv\Scripts\python.exe main.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/query` | Ask a question, get answer + retrieved chunks |
| GET | `/chunks` | List all indexed chunks |
| GET | `/docs` | Swagger UI (auto-generated) |

### POST /query — Request

```json
{
  "query": "What is the admission process at Velammal?"
}
```

### POST /query — Response

```json
{
  "query": "What is the admission process at Velammal?",
  "retrieved_chunks": [
    {
      "rank": 1,
      "score": 0.6188,
      "chunk": "The admission process typically involves an application submission, followed by an interaction or assessment depending on the grade level. Parents are encouraged to understand the school's vision and teaching methodology before enrolling."
    },
    {
      "rank": 2,
      "score": 0.5033,
      "chunk": "One of the common concerns raised by parents is regarding the fee structure. While the fees may be higher compared to some schools, Velammal offers modern infrastructure, experienced faculty, and a strong academic track record."
    },
    {
      "rank": 3,
      "score": 0.4252,
      "chunk": "Velammal Group of Schools focuses on holistic education combining academic excellence with extracurricular development. The schools emphasize discipline, values, and leadership skills among students."
    }
  ],
  "answer": "The admission process at Velammal typically involves an application submission, followed by an interaction or assessment depending on the grade level."
}
```

---

## Sample Outputs (3 Queries)

### Query 1: Admission Process

**Request:**
```json
{ "query": "What is the admission process at Velammal?" }
```

**Retrieved Chunks:**
- Rank 1 (Score: 0.6188): *The admission process typically involves an application submission, followed by an interaction or assessment depending on the grade level...*
- Rank 2 (Score: 0.5033): *One of the common concerns raised by parents is regarding the fee structure...*
- Rank 3 (Score: 0.4252): *Velammal Group of Schools focuses on holistic education...*

**Answer:**
> The admission process at Velammal typically involves an application submission, followed by an interaction or assessment depending on the grade level.

---

### Query 2: Fee Payments

**Request:**
```json
{ "query": "How are fee payments structured and what happens if fees are delayed?" }
```

**Retrieved Chunks:**
- Rank 1 (Score: 0.8411): *Fee payments are structured in installments, and timely payment is encouraged. Delays in fee payment may attract penalties or restrictions on certain services.*
- Rank 2 (Score: 0.2829): *One of the common concerns raised by parents is regarding the fee structure...*
- Rank 3 (Score: 0.1218): *The admission process typically involves an application submission...*

**Answer:**
> Fee payments are structured in installments. Timely payment is encouraged. Delays may attract penalties or restrictions on certain services.

---

### Query 3: Extracurricular Activities

**Request:**
```json
{ "query": "What extracurricular activities does Velammal offer?" }
```

**Retrieved Chunks:**
- Rank 1 (Score: 0.6566): *Velammal Group of Schools focuses on holistic education combining academic excellence with extracurricular development...*
- Rank 2 (Score: 0.6295): *Extracurricular activities such as sports, arts, and cultural programs are given equal importance to academics to ensure all-round development.*
- Rank 3 (Score: 0.4752): *One of the common concerns raised by parents is regarding the fee structure...*

**Answer:**
> Velammal offers extracurricular activities such as sports, arts, and cultural programs, which are given equal importance to academics to ensure all-round development.

---

## Design Decisions

- **No hallucination** — prompt explicitly instructs the model to answer only from provided context
- **No external knowledge** — FAISS retrieves only from the ingested dataset
- **LangGraph** — clean node-based agent graph (retrieve → generate), easy to extend with more nodes
- **FAISS cosine similarity** — embeddings are L2-normalized before indexing so inner product = cosine
- **FastAPI** — production-ready REST API with Pydantic validation and auto Swagger docs
- **Paragraph chunking** — preserves semantic coherence better than fixed-size character splits
