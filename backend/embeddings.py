"""
Pluggable embedding backend.

- If OPENAI_API_KEY is set -> uses OpenAI's text-embedding-3-small.
- Otherwise -> uses a local TF-IDF vectorizer (scikit-learn) as a drop-in
  stand-in so the retrieval pipeline is fully runnable offline/for free.

Swap TF-IDF for a real model (e.g. sentence-transformers or CLIP for images)
by implementing the same `embed_texts(list[str]) -> np.ndarray` interface.
"""
import numpy as np
from typing import List
from config import USE_OPENAI, OPENAI_API_KEY

_tfidf_vectorizer = None
_tfidf_fitted = False


def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return an (N, D) float32 embedding matrix for a list of strings."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    if USE_OPENAI:
        client = _get_openai_client()
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return vecs

    # --- Local fallback: TF-IDF (fit fresh each call over the given corpus) ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    global _tfidf_vectorizer
    _tfidf_vectorizer = TfidfVectorizer(max_features=384)
    mat = _tfidf_vectorizer.fit_transform(texts).toarray().astype(np.float32)
    # Pad to fixed width so vectors are comparable across calls
    if mat.shape[1] < 384:
        pad = np.zeros((mat.shape[0], 384 - mat.shape[1]), dtype=np.float32)
        mat = np.hstack([mat, pad])
    return mat


def embed_query(query: str, reference_corpus: List[str] = None) -> np.ndarray:
    """Embed a single query. For the TF-IDF fallback we need the corpus the
    index was built on so the vector space matches; pass it in."""
    if USE_OPENAI:
        return embed_texts([query])[0]
    corpus = (reference_corpus or []) + [query]
    mat = embed_texts(corpus)
    return mat[-1]
