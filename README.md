# OmniBrain — Agentic Multi-Modal RAG Orchestrator

A working implementation of the OmniBrain project (Week 1 + Week 2 + Week 3
self-correction, plus guardrails). This is real, runnable code — not pseudocode.

## What's implemented

| Spec module | Implementation | File |
|---|---|---|
| Multi-Modal Ingestion | PyMuPDF text/image extraction, chunking, embedding | `backend/ingestion.py` |
| Multi-Modal Retrieval (Qdrant/FAISS) | FAISS index with text + image-caption vectors | `backend/vector_store.py` |
| Vision-Language Model | GPT-4o (real) or offline placeholder captioner | `backend/vision_agent.py` |
| Agentic Orchestrator (LangGraph) | Supervisor routes to Search Agent / SQL Agent | `backend/orchestrator.py` |
| Text-to-SQL Agent | NL→SQL over a sample stock-price SQLite DB | `backend/sql_agent.py` |
| Self-RAG self-correction (Week 3) | Query rewrite + retry loop on low-relevance hits | `backend/orchestrator.py` (`search_with_self_correction`) |
| Guardrails (Week 3) | Scope check + grounding/hallucination heuristic | `backend/guardrails.py` |
| API Scaffolding (Week 1) | FastAPI `/upload` and `/query` endpoints | `backend/app.py` |
| Chat UI (Week 2/4) | Streamlit chat with thought-process + image display | `frontend/streamlit_app.py` |

**Important — this runs with zero API keys out of the box.** Every LLM-dependent
piece (embeddings, vision captions, NL→SQL, query routing) has a free local
fallback so you can develop and demo the *architecture* immediately. Set
`OPENAI_API_KEY` as an environment variable any time to switch every piece
over to the real GPT-4o / embedding calls automatically — no code changes
needed.

## Setup

```bash
cd omnibrain
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional — enables real GPT-4o vision + LLM routing + LLM embeddings
export OPENAI_API_KEY=sk-...
```

## Run it

Terminal 1 — backend:
```bash
cd backend
uvicorn app:app --reload --port 8000
```

Terminal 2 — frontend:
```bash
cd frontend
streamlit run streamlit_app.py
```

Then open the Streamlit URL, upload a PDF in the sidebar, and start asking
questions in the chat box. Try one question about the document's text/charts
(routes to the Search Agent) and one about "stock price" / "closing price"
(routes to the Text-to-SQL Agent against the sample `stock_data.db`).

## Testing without the UI

```bash
cd backend
python3 -c "
import ingestion, orchestrator
print(ingestion.ingest_pdf('/path/to/your.pdf'))
print(orchestrator.ask('What does the document say about X?'))
print(orchestrator.ask('What is the latest closing price for ACME?'))
"
```

## Architecture notes / what to build next (Week 4)

- **Langfuse observability**: not wired in yet. Add it by wrapping each node
  in `orchestrator.py` with `langfuse.trace()` spans — the `state["trace"]`
  list already gives you the exact points to instrument.
- **NeMo Guardrails**: `guardrails.py` is a lightweight stand-in. To use the
  real NeMo Guardrails, define a Colang flow that calls `scope_check` /
  `grounding_check` as custom actions.
- **Citation click-through UI**: `retrieved` results already carry `page` and
  `image_path` metadata — wire a click handler in Streamlit to open the
  source PDF at that page (e.g. via `pymupdf` render-to-image + `st.image`).
- **Real CLIP embeddings**: images are currently indexed via VLM-generated
  captions (text embeddings), not true CLIP image embeddings. Swap in
  `open_clip` in `embeddings.py` if you need genuine cross-modal
  image↔text similarity instead of caption-based search.
- **TF-IDF → real embedding model**: the free fallback in `embeddings.py`
  uses TF-IDF, which is weak semantically. For a stronger free option,
  install `sentence-transformers` and swap the fallback branch.

## Known limitations (by design, for the free-tier fallback path)

- TF-IDF embeddings are refit per search call for demo simplicity — fine for
  a few hundred chunks, not a production embedding strategy.
- Image "understanding" without `OPENAI_API_KEY` is a placeholder string, not
  real chart/table extraction.
- SQL NL→SQL fallback is keyword-based, not a full parser — ask questions
  containing a ticker name (ACME / GLOBEX / INITECH) and words like "latest",
  "highest", "average" for best results.
