"""
Streamlit chat UI for OmniBrain.

Run:
  streamlit run streamlit_app.py

Assumes the FastAPI backend is running at http://localhost:8000
(start it separately: `uvicorn app:app --reload --port 8000` from backend/).
"""
import streamlit as st
import requests
import time
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(page_title="OmniBrain", layout="wide")
st.title("🧠 OmniBrain — Agentic Multi-Modal RAG")

with st.sidebar:
    st.header("📄 Upload a document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and st.button("Ingest document"):
        resp = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
        )
        if not resp.ok:
            st.error(resp.text)
        else:
            job_id = resp.json()["job_id"]
            status_box = st.empty()
            progress = st.progress(0, text="Parsing, chunking, extracting images, embedding...")
            tick = 0
            while True:
                status_resp = requests.get(f"{API_URL}/status/{job_id}")
                status = status_resp.json()
                if status["status"] == "processing":
                    tick = (tick + 15) % 100
                    progress.progress(tick, text="Parsing, chunking, extracting images, embedding...")
                    time.sleep(1)
                    continue
                progress.empty()
                if status["status"] == "done":
                    st.success(f"Ingested: {status['result']}")
                    st.session_state["last_doc_id"] = status["result"]["doc_id"]
                else:
                    st.error(f"Ingestion failed: {status.get('error')}")
                break

    if st.session_state.get("last_doc_id"):
        with st.expander("🖼️ View extracted images (raw, bypasses search)"):
            doc_id = st.session_state["last_doc_id"]
            img_resp = requests.get(f"{API_URL}/images/{doc_id}")
            if img_resp.ok:
                paths = img_resp.json()["image_paths"]
                if not paths:
                    st.caption("No images were found in this PDF.")
                for p in paths:
                    if Path(p).exists():
                        st.image(p, caption=Path(p).name)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("trace"):
            with st.expander("🔎 Agent thought process"):
                for step in msg["trace"]:
                    st.text(step)
        if msg.get("grounding"):
            g = msg["grounding"]
            if g.get("grounded") is False:
                st.warning(f"⚠️ Guardrail: {g.get('reason')}")
            else:
                st.caption(f"✅ Grounded (overlap={g.get('query_overlap', g.get('score', 'N/A'))})")
        for img_ref in msg.get("images", []):
            p = Path(img_ref)
            if p.exists():
                st.image(str(p), caption=f"Referenced image: {p.name}")

if question := st.chat_input("Ask OmniBrain about your document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing through supervisor agent..."):
            resp = requests.post(f"{API_URL}/query", json={"question": question})
        if resp.ok:
            data = resp.json()
            st.write(f"**Route:** `{data['route']}`")
            st.write(data["answer"])
            image_paths = [r["image_path"] for r in data.get("retrieved", []) if "image_path" in r]
            for ip in image_paths:
                p = Path(ip)
                if p.exists():
                    st.image(str(p), caption=f"Referenced image: {p.name}")
            with st.expander("🔎 Agent thought process"):
                for step in data["trace"]:
                    st.text(step)
            g = data.get("grounding") or {}
            if g.get("grounded") is False:
                st.warning(f"⚠️ Guardrail: {g.get('reason')}")
            else:
                st.caption(f"✅ Grounded (overlap={g.get('query_overlap', g.get('score', 'N/A'))})")

            st.session_state.messages.append({
                "role": "assistant", "content": data["answer"],
                "trace": data["trace"], "grounding": g, "images": image_paths,
            })
        else:
            st.error(resp.text)
