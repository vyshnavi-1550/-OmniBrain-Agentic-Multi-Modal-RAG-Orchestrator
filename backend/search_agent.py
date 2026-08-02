"""Search Agent: semantic retrieval over indexed text + image captions."""
from config import TOP_K
from vector_store import VectorStore


def run(query: str, top_k: int = TOP_K) -> dict:
    store = VectorStore()
    found = store.load()
    if not found:
        return {"agent": "search", "results": [], "note": "No documents ingested yet."}
    results = store.search(query, top_k=top_k)
    return {"agent": "search", "results": results}
