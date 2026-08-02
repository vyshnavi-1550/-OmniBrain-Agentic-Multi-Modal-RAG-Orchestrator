"""
Week 2 deliverable: Agentic Orchestrator built with LangGraph.

Graph shape:

    supervisor -> (search_agent | sql_agent) -> synthesize -> guardrail -> END

The supervisor node decides whether the question needs semantic document
search or a historical-data SQL lookup (Text-to-SQL agent). The Vision Agent
is invoked directly by `answer_with_image()` when a result references a
specific chart/image the user wants explained (used by the API layer for
follow-up "explain that chart" queries).

Self-RAG style self-correction (Week 3) lives in `search_with_self_correction`.
"""
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

import search_agent
import sql_agent
import guardrails
from config import USE_OPENAI, OPENAI_API_KEY


class GraphState(TypedDict):
    query: str
    route: Optional[str]
    retrieved: List[Dict[str, Any]]
    sql_result: Optional[Dict[str, Any]]
    answer: Optional[str]
    grounding: Optional[Dict[str, Any]]
    trace: List[str]


SQL_KEYWORDS = [
    "price", "stock", "close", "open", "volume", "high", "low",
    "average", "historical", "quarter", "revenue", "ticker",
]


def supervisor_node(state: GraphState) -> GraphState:
    q = state["query"].lower()
    if USE_OPENAI:
        route = _llm_route(state["query"])
    else:
        route = "sql" if any(k in q for k in SQL_KEYWORDS) else "search"
    state["route"] = route
    state["trace"].append(f"supervisor -> routed to '{route}'")
    return state


def _llm_route(question: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                "Classify this question as exactly one word, 'sql' or 'search'. "
                "'sql' = requires looking up historical numeric stock data. "
                "'search' = requires reading document text/charts/tables. "
                f"Question: {question}\nAnswer with one word:"
            )
        }],
        max_tokens=5,
        temperature=0,
    )
    word = resp.choices[0].message.content.strip().lower()
    return "sql" if "sql" in word else "search"


def search_node(state: GraphState) -> GraphState:
    result = search_with_self_correction(state["query"])
    state["retrieved"] = result["results"]
    state["trace"].extend(result["trace"])
    return state


def sql_node(state: GraphState) -> GraphState:
    result = sql_agent.run(state["query"])
    state["sql_result"] = result
    state["trace"].append(f"sql_agent -> {result.get('sql', result.get('error'))}")
    return state


def synthesize_node(state: GraphState) -> GraphState:
    if state["route"] == "sql":
        r = state["sql_result"]
        if r.get("error"):
            state["answer"] = f"SQL lookup failed: {r['error']}"
        else:
            state["answer"] = (
                f"Ran query: `{r['sql']}`\nResult rows: {r['rows']}"
            )
    else:
        chunks = state["retrieved"]
        if not chunks:
            state["answer"] = "No relevant information found in the ingested documents."
        else:
            citations = ", ".join(sorted({f"p.{c['page']}" for c in chunks if 'page' in c}))
            preview = " ".join(c["text"][:200] for c in chunks[:3])
            state["answer"] = f"Based on retrieved context [{citations}]: {preview}"
    state["trace"].append("synthesize -> answer drafted")
    return state


def guardrail_node(state: GraphState) -> GraphState:
    if state["route"] == "sql":
        sql_res = state.get("sql_result") or {}
        has_context = bool(sql_res.get("rows")) or bool(sql_res.get("error"))
        context_texts = [str(sql_res.get("rows", [])), str(sql_res.get("sql", ""))]
    else:
        has_context = guardrails.scope_check(state.get("retrieved", []))
        context_texts = [c["text"] for c in state.get("retrieved", [])]

    if not has_context:
        state["answer"] = "I don't have enough grounded context to answer that question from the ingested documents."
        state["grounding"] = {"grounded": False, "reason": "scope_check failed"}
    else:
        state["grounding"] = guardrails.grounding_check(state["answer"] or "", context_texts)
    state["trace"].append(f"guardrail -> {state['grounding']}")
    return state


def route_decision(state: GraphState) -> str:
    return state["route"]


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search_agent", search_node)
    graph.add_node("sql_agent", sql_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("guardrail", guardrail_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", route_decision, {
        "search": "search_agent",
        "sql": "sql_agent",
    })
    graph.add_edge("search_agent", "synthesize")
    graph.add_edge("sql_agent", "synthesize")
    graph.add_edge("synthesize", "guardrail")
    graph.add_edge("guardrail", END)
    return graph.compile()


_compiled_graph = None


def ask(query: str) -> Dict[str, Any]:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    initial: GraphState = {
        "query": query, "route": None, "retrieved": [], "sql_result": None,
        "answer": None, "grounding": None, "trace": [],
    }
    final_state = _compiled_graph.invoke(initial)
    return final_state


# ---------- Week 3: Self-RAG self-correction loop ----------
def search_with_self_correction(query: str, max_retries: int = 2) -> Dict[str, Any]:
    trace = []
    q = query
    for attempt in range(max_retries + 1):
        result = search_agent.run(q)
        results = result["results"]
        trace.append(f"search_agent attempt {attempt+1} for '{q}' -> {len(results)} hits")

        if not results:
            trace.append("no results at all, stopping retries")
            break

        # crude relevance signal: FAISS L2 distance, lower = better
        best_score = min(r["score"] for r in results)
        RELEVANCE_THRESHOLD = 1.5  # tune per embedding space
        if best_score <= RELEVANCE_THRESHOLD or attempt == max_retries:
            trace.append(f"accepted results (best_score={best_score:.3f})")
            return {"results": results, "trace": trace}

        # Self-RAG: rewrite the query and retry
        q = _rewrite_query(query, attempt)
        trace.append(f"low relevance (best_score={best_score:.3f}), rewriting query -> '{q}'")

    return {"results": [], "trace": trace}


def _rewrite_query(original_query: str, attempt: int) -> str:
    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite this search query to be more specific and likely "
                    f"to retrieve relevant document chunks: '{original_query}'. "
                    "Output only the rewritten query."
                )
            }],
            max_tokens=50, temperature=0.5,
        )
        return resp.choices[0].message.content.strip()
    # Fallback heuristic rewrite: broaden by dropping stop-ish qualifier words
    words = [w for w in original_query.split() if len(w) > 3]
    return " ".join(words[: max(2, len(words) - attempt)])
