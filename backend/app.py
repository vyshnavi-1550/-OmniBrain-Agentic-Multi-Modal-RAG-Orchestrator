"""
FastAPI backend for OmniBrain.

Endpoints:
  POST /upload            - upload a PDF, kicks off ingestion as a background job
  GET  /status/{job_id}   - poll ingestion progress/result for a job
  POST /query              - ask a question, routed through the LangGraph orchestrator
  GET  /health              - simple healthcheck

Run:
  uvicorn app:app --reload --port 8000
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
import shutil
import uuid
import threading
import traceback

from config import UPLOAD_DIR
import ingestion
import orchestrator

app = FastAPI(title="OmniBrain API")

_jobs: dict = {}
_jobs_lock = threading.Lock()


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_ingestion_job(job_id: str, pdf_path: str, doc_id: str):
    try:
        summary = ingestion.ingest_pdf(pdf_path, doc_id=doc_id)
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "result": summary}
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id] = {
                "status": "error",
                "error": str(e),
                "trace": traceback.format_exc(),
            }


@app.post("/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    doc_id = str(uuid.uuid4())[:8]
    dest_dir = UPLOAD_DIR / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing"}

    background_tasks.add_task(_run_ingestion_job, job_id, str(dest_path), doc_id)

    return {"job_id": job_id, "doc_id": doc_id, "status": "processing"}


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id.")
    return {"job_id": job_id, **job}


@app.get("/images/{doc_id}")
async def list_images(doc_id: str):
    image_dir = UPLOAD_DIR / doc_id / "images"
    if not image_dir.exists():
        raise HTTPException(404, "No images found for this doc_id.")
    images = sorted(str(p) for p in image_dir.glob("*") if p.is_file())
    return {"doc_id": doc_id, "count": len(images), "image_paths": images}


@app.post("/query")
async def query(req: QueryRequest):
    result = orchestrator.ask(req.question)

    grounding = result.get("grounding") or {}
    is_refused = "reason" in grounding
    is_grounded = grounding.get("grounded", False)
    grounding_score = grounding.get("score", 0.0)

    images = []
    for item in result.get("retrieved", []):
        if isinstance(item, dict) and item.get("modality") == "image":
            path = item.get("image_path") or item.get("path") or item.get("file_path")
            if path:
                images.append(path)

    return {
        "question": req.question,
        "route": result["route"],
        "answer": result["answer"],
        "grounded": is_grounded,
        "grounding_score": grounding_score,
        "refused": is_refused,
        "images": images,
        "retrieved": result.get("retrieved", []),
        "sql": result.get("sql_result"),
        "trace": result["trace"],
    }