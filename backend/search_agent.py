"""Search Agent: semantic retrieval over indexed text + image captions."""
from config import TOP_K
from vector_store import VectorStore

IMAGE_INTENT_KEYWORDS = [
    "image", "images", "picture", "diagram", "chart", "figure",
    "graph", "screenshot", "illustration", "photo", "visual",
]


def _wants_image(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in IMAGE_INTENT_KEYWORDS)


def run(query: str, top_k: int = TOP_K) -> dict:
    store = VectorStore()
    found = store.load()
    if not found:
        return {"agent": "search", "results": [], "note": "No documents ingested yet."}

    if _wants_image(query):
        image_results = store.search(query, top_k=top_k, modality_filter="image")
        if image_results:
            return {"agent": "search", "results": image_results, "intent": "image"}

    results = store.search(query, top_k=top_k, modality_filter="text")
    return {"agent": "search", "results": results}