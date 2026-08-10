"""
Custom action registered with NeMo Guardrails.

`check_topic_relevance` is called from the Colang flow in
rails/topic_control.co. It receives the current retrieved document context
(set by the orchestrator before invoking the rails engine) and decides
whether the user's question is actually answerable from it.

Two modes:
  - No OPENAI_API_KEY: runs the same free lexical overlap check used
    elsewhere in the project (query's own key terms vs. retrieved context).
  - OPENAI_API_KEY set: could be swapped for a real LLM-judge call here
    without changing the Colang flow or the calling code.
"""
import sys
from pathlib import Path
from nemoguardrails.actions import action

# Reuse the existing free relevance-check logic rather than duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import guardrails as local_guardrails
from config import USE_OPENAI


@action(name="check_topic_relevance")
async def check_topic_relevance(context: dict) -> dict:
    query = context.get("user_message", "")
    retrieved_context = context.get("retrieved_context_texts", [])

    if USE_OPENAI:
        # Placeholder hook: an LLM-based topical judge could go here,
        # reusing the same `context` dict. Falls back to the local check
        # for now so behavior stays consistent either way.
        pass

    result = local_guardrails.query_relevance_check(query, retrieved_context)
    return {"relevant": result["relevant"], "overlap": result["overlap"], "reason": result["reason"]}