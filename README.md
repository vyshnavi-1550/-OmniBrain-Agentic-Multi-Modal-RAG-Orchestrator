# OmniBrain — Agentic Multi-Modal RAG

OmniBrain is an agentic Retrieval-Augmented Generation (RAG) system that lets you upload a PDF and ask natural-language questions about it. It combines document search, historical SQL lookups, self-correction, and safety guardrails behind a single conversational interface.

## Features

- **Multi-modal document ingestion** — parses PDF text, tables, and images, extracting page-level chunks and embedding them for semantic search.
- **Agentic orchestration (LangGraph)** — a supervisor node routes each question to the right specialist agent: a Search Agent for document Q&A, or a SQL Agent for historical/numeric data lookups.
- **Self-RAG self-correction** — when a retrieved result's relevance score falls below a threshold, the system automatically rewrites the query and retries (up to 2 additional attempts) before falling back to its best available result.
- **Guardrails** — every answer is checked for topic relevance and grounding in the ingested document before being returned, with graceful refusals for off-topic or ungrounded questions.
- **Real NeMo Guardrails integration** — uses the actual NeMo Guardrails library (Colang flow + custom action + RailsConfig/LLMRails) where available, with an automatic, dependency-free fallback to equivalent logic on environments where the native library can't install (e.g. Python 3.14 compatibility gaps).

## Architecture

```
User question
     │
     ▼
 Supervisor Agent  ──► routes to "search" or "sql"
     │                          │
     ▼                          ▼
 Search Agent            SQL Agent (Text-to-SQL)
 (with Self-RAG                │
  retry loop)                  │
     │                          │
     └────────────┬─────────────┘
                   ▼
             Synthesize Answer
                   │
                   ▼
             Guardrail Check
        (scope check → topic
         relevance → grounding)
                   │
                   ▼
              Final Response
```

- **Supervisor node** — classifies the question as needing document search or a SQL lookup, either via keyword matching or an LLM call.
- **Search Agent** — retrieves relevant document chunks; wrapped in a self-correction loop (`search_with_self_correction`) that rewrites the query and retries if the best match is below a relevance threshold.
- **SQL Agent** — handles historical/numeric queries (price, revenue, volume, etc.) via Text-to-SQL.
- **Synthesize node** — drafts an answer from the retrieved context or SQL result.
- **Guardrail node** — runs a two-stage check:
  1. **Scope check** — confirms there's any usable context at all.
  2. **Topic relevance check** (via NeMo Guardrails or fallback) — compares the question's key terms against retrieved context to catch off-topic questions, while preserving short uppercase acronyms (e.g. "DDL", "DML") so they aren't wrongly filtered out.
  3. **Grounding check** — verifies the final answer is actually supported by the retrieved context, returning a grounding score.

## Setup

### Prerequisites
- Python 3.x (developed and tested on Python 3.14; note the NeMo Guardrails fallback below)
- A virtual environment (`.venv`) with dependencies installed

### Running the app

You need **two terminals running simultaneously** — the backend and frontend are separate processes.

**Terminal 1 — Backend (FastAPI)**
```powershell
cd backend
python -m uvicorn app:app --reload --port 8000
```
Runs at `http://localhost:8000`. Health check: `GET /health`.

**Terminal 2 — Frontend (Streamlit)**
```powershell
cd frontend
streamlit run streamlit_app.py
```
Runs at `http://localhost:8501`.

Open `http://localhost:8501` in your browser, upload a PDF, click **Ingest document**, and start asking questions.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Simple healthcheck |
| `POST` | `/upload` | Upload a PDF; kicks off ingestion as a background job |
| `GET` | `/status/{job_id}` | Poll ingestion progress/result |
| `GET` | `/images/{doc_id}` | List extracted images for a document (direct disk lookup) |
| `POST` | `/query` | Ask a question; routed through the LangGraph orchestrator |

### `/query` response shape

```json
{
  "question": "string",
  "route": "search" | "sql",
  "answer": "string",
  "grounded": true,
  "grounding_score": 0.87,
  "refused": false,
  "images": ["path/to/image.png"],
  "retrieved": [ ... ],
  "sql": null,
  "trace": ["supervisor -> routed to 'search'", "..."]
}
```

## Known Limitations / Notes

- **Python 3.14 + NeMo Guardrails**: the native `nemoguardrails` package has dependencies that don't currently install cleanly on Python 3.14. The integration is built with an automatic fallback — the guardrail behaves identically whether the real library is available or not — but you may see an "Import could not be resolved" warning in your IDE if the package isn't installed. This is expected and does not affect runtime behavior.
- **Image retrieval without `OPENAI_API_KEY`**: without an OpenAI key configured, extracted images use generic placeholder captions rather than real GPT-4o-generated descriptions, which limits semantic image search precision (an image request may return the nearest available images rather than a page-exact match).
- **Very short/single-word queries**: extremely short queries (e.g. a single acronym in isolation) can occasionally be flagged as off-topic by the relevance check, since there's less text to compare against the retrieved context. Multi-word questions are more reliable.

## Development Progress

### Week 1 — Core RAG Pipeline
- PDF ingestion (text, tables, images)
- Embedding + vector storage
- Basic search and Q&A flow

### Week 2 — Agentic Orchestration
- LangGraph-based supervisor routing (Search Agent vs. SQL Agent)
- Guardrails tested; 2 real bugs found and fixed:
  - Off-topic questions could return a false "grounded" result if retrieved text was self-consistent but irrelevant — fixed by adding a query-vs-context relevance check.
  - Short technical acronyms (DDL, DML) were wrongly filtered by a word-length rule — fixed by preserving uppercase acronyms regardless of length.

### Week 3 — Self-Correction & Guardrails
- **Day 1**: Self-RAG self-correction loop — automatic query rewriting and retry (up to 2 attempts) when retrieval relevance is too low.
- **Day 2**: Guardrail bug fixes (see above), verified via live testing.
- **Day 3**: Real NeMo Guardrails library integration (Colang flow, custom action, RailsConfig/LLMRails), with dependency-free fallback for Python 3.14 environments. Verified end-to-end: off-topic refusal, on-topic grounded answers, inline image responses.
- **Day 4**: Wired guardrail results into the `/query` API response (flat `grounded`/`refused`/`grounding_score`/`images` fields) and the Streamlit UI (⚠️ refusal banner, ✅ grounded badge, inline image rendering). Ran an edge-case sweep (empty query, long query, acronym-only query, no-match image request, rapid repeated queries) — confirmed no crashes or 500 errors across all cases.
- **Day 5**: Final regression pass, README and architecture documentation, close-out commit.

## Key Terms

Self-RAG · self-correction loop · query rewriting · guardrails · NeMo Guardrails · Colang flow · custom actions · RailsConfig/LLMRails · scope check · grounding check · query relevance check · hallucination prevention
