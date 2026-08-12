"""
Week 1 deliverable: Multi-Modal Ingestion pipeline.

- Parses a PDF page by page with PyMuPDF (fitz)
- Chunks text with overlap
- Extracts embedded images, saves them to disk, and captions them via the
  vision agent (real GPT-4o or fallback placeholder)
- Embeds both text chunks and image captions and writes them into the
  VectorStore, tagged with `modality` so the retriever can filter/route
"""
import fitz  # PyMuPDF
import uuid
from pathlib import Path
from typing import List, Dict, Any

from config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOAD_DIR
from vector_store import VectorStore
from vision_agent import describe_image


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_pdf(pdf_path: str, doc_id: str = None) -> Dict[str, Any]:
    """Parse a PDF, extract text + images, embed both, and persist into the
    VectorStore. Returns a summary of what was ingested."""
    doc_id = doc_id or str(uuid.uuid4())[:8]
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    store = VectorStore()
    store.load()  # append to any existing index

    text_texts, text_payloads = [], []
    image_texts, image_payloads = [], []

    image_out_dir = UPLOAD_DIR / doc_id / "images"
    image_out_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        # --- text ---
        page_text = page.get_text()
        for chunk in chunk_text(page_text):
            text_texts.append(chunk)
            text_payloads.append({
                "modality": "text",
                "doc_id": doc_id,
                "page": page_number,
                "source_file": pdf_path.name,
            })

        # --- images ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            img_filename = f"p{page_number}_{img_index}.{ext}"
            img_path = image_out_dir / img_filename
            img_path.write_bytes(image_bytes)

            caption = describe_image(
                image_bytes, page_number,
                context_hint=f"(from {pdf_path.name})"
            )
            image_texts.append(caption)
            image_payloads.append({
                "modality": "image",
                "doc_id": doc_id,
                "page": page_number,
                "source_file": pdf_path.name,
                "image_path": str(img_path),
            })

    store.add(text_texts, text_payloads)
    store.add(image_texts, image_payloads)
    store.save()

    return {
        "doc_id": doc_id,
        "pages": len(doc),
        "text_chunks_indexed": len(text_texts),
        "images_indexed": len(image_texts),
    }