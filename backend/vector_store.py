"""
Multi-modal vector store backed by FAISS (stand-in for Qdrant -- same
concept: approximate nearest-neighbor search over embeddings with attached
metadata payloads). Swap `faiss.IndexFlatL2` for a `qdrant_client.QdrantClient`
collection later without changing the calling code's shape.

Two logical collections are kept in one store, distinguished by `modality`:
  - "text"  : chunks of parsed document text
  - "image" : captions/descriptions of extracted images & charts
"""
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any

from config import INDEX_DIR
from embeddings import embed_texts, embed_query


class VectorStore:
    def __init__(self, name: str = "omnibrain"):
        self.name = name
        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)
        self.payloads: List[Dict[str, Any]] = []   # metadata per vector, same order as index
        self.corpus: List[str] = []                # raw text per vector (needed for TF-IDF fallback)

    # ---------- persistence ----------
    def path_prefix(self) -> Path:
        return INDEX_DIR / self.name

    def save(self):
        faiss.write_index(self.index, str(self.path_prefix()) + ".faiss")
        with open(str(self.path_prefix()) + ".meta.pkl", "wb") as f:
            pickle.dump({"payloads": self.payloads, "corpus": self.corpus}, f)

    def load(self):
        p = self.path_prefix()
        if (Path(str(p) + ".faiss")).exists():
            self.index = faiss.read_index(str(p) + ".faiss")
            with open(str(p) + ".meta.pkl", "rb") as f:
                meta = pickle.load(f)
            self.payloads = meta["payloads"]
            self.corpus = meta["corpus"]
            return True
        return False

    # ---------- writes ----------
    def add(self, texts: List[str], payloads: List[Dict[str, Any]]):
        assert len(texts) == len(payloads)
        if not texts:
            return
        vecs = embed_texts(texts)
        if vecs.shape[1] != self.dim:
            self.dim = vecs.shape[1]
        self.index.add(vecs)
        self.payloads.extend(payloads)
        self.corpus.extend(texts)

    # ---------- reads ----------
    def search(self, query: str, top_k: int = 5, modality_filter: str = None) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        if modality_filter:
            # Filter to the matching subset BEFORE ranking -- filtering only
            # the global top-k *after* ranking can silently return nothing
            # if no matching-modality vector happens to land in that top-k.
            matching_idx = [i for i, p in enumerate(self.payloads) if p.get("modality") == modality_filter]
            if not matching_idx:
                return []
            sub_corpus = [self.corpus[i] for i in matching_idx]
            sub_vecs = embed_texts(sub_corpus)
            qvec = embed_query(query, reference_corpus=sub_corpus)
            sub_index = faiss.IndexFlatL2(sub_vecs.shape[1])
            sub_index.add(sub_vecs)
            D, I = sub_index.search(np.array([qvec]), min(top_k, sub_index.ntotal))
            results = []
            for dist, sub_i in zip(D[0], I[0]):
                if sub_i < 0:
                    continue
                orig_idx = matching_idx[sub_i]
                results.append({
                    "score": float(dist),
                    "text": self.corpus[orig_idx],
                    **self.payloads[orig_idx],
                })
            return results

        qvec = embed_query(query, reference_corpus=self.corpus)
        # rebuild a temporary index if using TF-IDF fallback so dims match
        # (embed_query returns a vector sized to current corpus+query fit)
        if qvec.shape[0] != self.index.d:
            # Rebuild index fresh from stored corpus (TF-IDF fallback path)
            vecs = embed_texts(self.corpus)
            tmp_index = faiss.IndexFlatL2(vecs.shape[1])
            tmp_index.add(vecs)
            qvec = embed_query(query, reference_corpus=self.corpus)
            D, I = tmp_index.search(np.array([qvec]), min(top_k, tmp_index.ntotal))
        else:
            D, I = self.index.search(np.array([qvec]), min(top_k, self.index.ntotal))

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.payloads):
                continue
            payload = self.payloads[idx]
            results.append({
                "score": float(dist),
                "text": self.corpus[idx],
                **payload
            })
        return results