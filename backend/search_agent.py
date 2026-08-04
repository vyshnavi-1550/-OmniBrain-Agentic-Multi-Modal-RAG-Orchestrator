"""Search Agent: semantic retrieval over indexed text + image captions."""
from config import TOP_K
from vector_store import VectorStore

# Keywords that signal the user explicitly wants to see an extracted
# image/chart/diagram/table, not just read about the topic in prose.
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
        # Deliberately pull from the image collection. Without a real VLM
        # (OPENAI_API_KEY), captions are near-identical placeholders, so
        # semantic ranking can't meaningfully distinguish between images --
        # explicit intent detection guarantees the feature is exercised
        # instead of silently failing to surface any image at all.
        image_results = store.search(query, top_k=top_k, modality_filter="image")
        if image_results:
            return {"agent": "search", "results": image_results, "intent": "image"}
        # fall through to normal search if no images were indexed at all

    results = store.search(query, top_k=top_k)
    return {"agent": "search", "results": results}