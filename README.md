# OmniBrain — Agentic Multi-Modal RAG

OmniBrain is an agentic Retrieval-Augmented Generation (RAG) system that lets you upload a PDF and ask natural-language questions about it. It combines document search, historical SQL lookups, self-correction, safety guardrails, observability, and clickable source citations behind a single conversational interface.

## Features

- **Multi-modal document ingestion** — parses PDF text, tables, and images, extracting page-level chunks and embedding them for semantic search.
- **Agentic orchestration (LangGraph)** — a supervisor node routes each question to the right specialist agent: a Search Agent for document Q&A, or a SQL Agent for historical/numeric data lookups.
- **Self-RAG self-correction** — when a retrieved result's relevance score falls below a threshold, the system automatically rewrites the query and retries (up to 2 additional attempts) before falling back to its best available result.
- **Guardrails** — every answer is checked for topic relevance and grounding in the ingested document before being returned, with graceful refusals for off-topic or ungrounded questions.
- **Real NeMo Guardrails integration** — uses the actual NeMo Guardrails library (Colang flow + custom action + RailsConfig/LLMRails) where available, with an automatic, dependency-free fallback to equivalent logic on environments where the native library can't install (e.g. Python 3.14 compatibility gaps).
- **Observability (Langfuse)** — every query is traced end-to-end, with nested spans for each agent step (supervisor, search agent, SQL agent, synthesize, guardrail), showing per-step latency and outcomes in the Langfuse dashboard.
- **Clickable citations** — every search-route answer shows buttons for each cited page (e.g. `p.16`); clicking one renders and displays the exact source PDF page inline, so claims can be instantly verified against the original document.

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

Every node above runs inside a Langfuse-traced span, nested under a top-level `omnibrain-query` trace, giving a full latency and outcome breakdown per query.

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
- (Optional but recommended) A free Langfuse account for observability — see below

### Langfuse setup (for tracing/observability)

1. Sign up for a free account at [cloud.langfuse.com](https://cloud.langfuse.com).
2. Create a project (e.g. "OmniBrain").
3. Go to **Project Settings → API Keys → Create new API key**.
4. Copy the generated `.env` snippet, which will look like:
   ```
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```
5. Create a file named `.env` in the project root (`omnibrain/.env`) and paste those three lines in.
6. Confirm `.env` is listed in `.gitignore` (it should already be) so your keys are never committed.

If these variables aren't set, tracing is skipped gracefully — the app runs normally either way, since `USE_LANGFUSE` is derived from whether both keys are present.

**Note:** double-check the `LANGFUSE_HOST` matches the region your project's keys belong to (e.g. `cloud.langfuse.com` vs. a region-specific subdomain) — using the wrong host will cause traces to silently land in the wrong project or fail to send.

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
| `GET` | `/page/{doc_id}/{page_number}` | Render a specific PDF page as a PNG image on demand (used for clickable citations) |
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
  "retrieved": [ { "doc_id": "...", "page": 6, "text": "...", "score": 0.4 } ],
  "sql": null,
  "trace": ["supervisor -> routed to 'search'", "..."]
}
```

Each item in `retrieved` carries a `doc_id` and `page` field, which the frontend uses to render clickable citation buttons that call `/page/{doc_id}/{page_number}`.

## Citation Links Feature

For search-route answers, the Streamlit UI extracts the unique set of `(doc_id, page)` pairs referenced in `retrieved`, and renders one button per page (e.g. `p.6`, `p.16`). Clicking a button fetches that page from `/page/{doc_id}/{page_number}` and displays the rendered PDF page inline — letting anyone instantly verify an AI-generated claim against its exact source page, rather than trusting the citation blindly.

## Known Limitations / Notes

- **Python 3.14 + NeMo Guardrails**: the native `nemoguardrails` package has dependencies that don't currently install cleanly on Python 3.14. The integration is built with an automatic fallback — the guardrail behaves identically whether the real library is available or not — but you may see an "Import could not be resolved" warning in your IDE if the package isn't installed. This is expected and does not affect runtime behavior.
- **Image retrieval without `OPENAI_API_KEY`**: without an OpenAI key configured, extracted images use generic placeholder captions rather than real GPT-4o-generated descriptions, which limits semantic image search precision (an image request may return the nearest available images rather than a page-exact match). Since GPT-4o and LLaVA are both viable per the original project spec, swapping in a local/open-source VLM (e.g. LLaVA via Ollama) is a documented path to closing this gap without API cost.
- **Very short/single-word queries**: extremely short queries (e.g. a single acronym in isolation) can occasionally be flagged as off-topic by the relevance check, since there's less text to compare against the retrieved context. Multi-word questions are more reliable.
- **Citation page numbers across multiple ingested documents**: if the vector index contains chunks from more than one ingested PDF that happen to share the same page number, the citation button logic prefers the currently active document (`last_doc_id` in the Streamlit session) to avoid duplicate/ambiguous buttons.

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

### Week 4 — Observability & Citation Links
- **Day 1**: Integrated Langfuse tracing on the top-level `orchestrator.ask()` call via the `@observe` decorator, capturing input, output, latency, and route/grounding metadata for every query.
- **Day 2**: Extended tracing to node-level granularity — each LangGraph node (supervisor, search_agent, sql_agent, synthesize, guardrail) now runs inside its own nested Langfuse span, giving a full per-step latency breakdown (e.g. surfaced that search retrieval, not synthesis or guardrails, was the dominant cost in a slow query).
- **Day 3**: Added a new `GET /page/{doc_id}/{page_number}` backend endpoint that renders any page of an ingested PDF as a PNG on demand, using the originally uploaded file (no pre-rendering or extra storage required).
- **Day 4**: Wired the new endpoint into the Streamlit UI — search-route answers now show clickable citation buttons (e.g. `p.6`, `p.16`) that fetch and display the exact source page inline, closing the "Refine & Polish: citation links" requirement from the original project spec.
- **Day 5**: Full regression pass across all 4 weeks, README and documentation update, close-out commit.

## Key Terms

Self-RAG · self-correction loop · query rewriting · guardrails · NeMo Guardrails · Colang flow · custom actions · RailsConfig/LLMRails · scope check · grounding check · query relevance check · hallucination prevention · Langfuse · observability · nested spans · citation links
