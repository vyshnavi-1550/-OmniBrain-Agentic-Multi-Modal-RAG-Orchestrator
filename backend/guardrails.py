"""
Lightweight guardrails layer (stand-in for NeMo Guardrails config).

- scope_check: blocks answering questions with no relevant retrieved context
  at all (keeps the assistant "strictly grounded in the provided documents").
- grounding_check: rough heuristic that flags an answer as possibly
  unsupported if it shares very little vocabulary with the retrieved
  context. Not a substitute for a real hallucination classifier, but
  demonstrates the guardrail *hook point* -- swap in a NeMo Guardrails
  colang flow or an LLM-judge call here later.
"""
from typing import List, Dict, Any

MIN_CONTEXT_OVERLAP = 0.08

STOPWORDS = {
    "what", "which", "does", "this", "that", "with", "from", "have",
    "about", "tell", "explain", "when", "where", "there", "their",
    "your", "some", "into", "than", "then", "them", "these", "those",
}


def scope_check(retrieved: List[Dict[str, Any]]) -> bool:
    """Return True if there is enough retrieved context to answer at all."""
    return len(retrieved) > 0


def query_relevance_check(query: str, context_chunks: List[str], threshold: float = 0.2) -> Dict[str, Any]:
    """Check whether the QUESTION's own vocabulary actually shows up in the
    retrieved context."""
    def tokenize(t):
        tokens = set()
        for w in t.split():
            clean = w.strip(".,?!:;()")
            if not clean:
                continue
            if clean.isupper() and len(clean) >= 2:
                tokens.add(clean.lower())
            elif len(clean) > 3:
                tokens.add(clean.lower())
        return tokens - STOPWORDS

    query_tokens = tokenize(query)
    if not query_tokens:
        return {"relevant": True, "overlap": None, "reason": None}

    context_tokens = set()
    for c in context_chunks:
        context_tokens |= tokenize(c)

    overlap = len(query_tokens & context_tokens) / len(query_tokens)
    relevant = overlap >= threshold
    return {
        "relevant": relevant,
        "overlap": round(overlap, 3),
        "reason": None if relevant else "Question's own key terms do not appear in the retrieved document context -- likely off-topic for this document.",
    }


def grounding_check(answer: str, context_chunks: List[str]) -> Dict[str, Any]:
    if not context_chunks:
        return {"grounded": False, "overlap": 0.0, "reason": "No context retrieved."}

    def tokenize(t):
        return set(w.lower() for w in t.split() if len(w) > 3)

    answer_tokens = tokenize(answer)
    context_tokens = set()
    for c in context_chunks:
        context_tokens |= tokenize(c)

    if not answer_tokens:
        return {"grounded": False, "overlap": 0.0, "reason": "Empty answer."}

    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    grounded = overlap >= MIN_CONTEXT_OVERLAP
    return {
        "grounded": grounded,
        "overlap": round(overlap, 3),
        "reason": None if grounded else "Low vocabulary overlap with retrieved context; possible hallucination.",
    }