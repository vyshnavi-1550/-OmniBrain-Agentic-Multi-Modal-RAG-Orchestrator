"""
Week 3 deliverable: real NeMo Guardrails integration (replaces the earlier
hand-rolled guardrails.py-only approach with the actual NeMo Guardrails
framework -- Colang flows + custom actions + rails engine).

Usage from orchestrator.py:

    from nemo_integration import check_topic_relevance_nemo
    result = check_topic_relevance_nemo(query, context_texts)
    # -> {"relevant": bool, "overlap": float, "reason": str|None}

The rails engine is initialized once (lazily) and reused across requests.
If NeMo Guardrails fails to initialize or isn't installed at all (e.g. on
platforms where its native dependencies aren't yet supported), this module
falls back to calling the local check function directly -- with NO
dependency on the nemoguardrails package -- so the app never breaks because
of the guardrails layer.
"""
from pathlib import Path
from typing import List, Dict, Any

CONFIG_PATH = Path(__file__).resolve().parent / "guardrails_config"

_rails = None
_init_failed = False


def _get_rails():
    global _rails, _init_failed
    if _rails is not None or _init_failed:
        return _rails
    try:
        from nemoguardrails import RailsConfig, LLMRails
        config = RailsConfig.from_path(str(CONFIG_PATH))
        _rails = LLMRails(config)
    except Exception as e:
        print(f"[nemo_integration] NeMo Guardrails unavailable, using direct fallback check: {e}")
        _init_failed = True
        _rails = None
    return _rails


def check_topic_relevance_nemo(query: str, context_texts: List[str]) -> Dict[str, Any]:
    """Run the topic-relevance guardrail through NeMo Guardrails' action
    system. Falls back to calling the check function directly -- with no
    dependency on the nemoguardrails package at all -- if the library isn't
    installed or the rails engine can't initialize for any reason. This
    keeps the app fully functional even on platforms where nemoguardrails
    can't currently be installed (e.g. very new Python versions not yet
    supported by the library or its native dependencies)."""
    rails = _get_rails()

    if rails is None:
        import guardrails as local_guardrails
        return local_guardrails.query_relevance_check(query, context_texts)

    import asyncio
    action_fn = rails.runtime.action_dispatcher.get_action(name="check_topic_relevance")
    result = asyncio.run(action_fn(context={
        "user_message": query,
        "retrieved_context_texts": context_texts,
    }))
    return result