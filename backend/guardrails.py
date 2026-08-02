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


def scope_check(retrieved: List[Dict[str, Any]]) -> bool:
    """Return True if there is enough retrieved context to answer at all."""
    return len(retrieved) > 0


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
