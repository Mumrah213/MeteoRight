"""High-level entry point: question in, structured answer out.

Probes the backend ONCE and threads the cached result through the graph (so the
tools never re-probe), then invokes the constrained graph and returns its
``result`` block plus a compact trace.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from agent.graph import MAX_TOOL_STEPS, build_graph
from agent.llm import make_llm


def answer(
    question: str,
    *,
    backend_base_url: str = "http://127.0.0.1:8080/v1/forecast",
    llm: Any | None = None,
    backend_desc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer a forecast-accuracy question through the constrained agent graph.

    Args:
        question: Natural-language question.
        backend_base_url: MeteoRight backend forecast endpoint.
        llm: Override the chat model (tests inject a fake); default builds the
            configured ChatOpenAI.
        backend_desc: Override/inject a cached describe_backend() result (tests
            and the orchestrator pass this to avoid a live probe).

    Returns:
        ``{"question", "answer", "in_scope", "tool_calls": [...], "steps",
           "error": None | {...}}``
    """
    if llm is None:
        llm = make_llm()

    # Probe the backend once unless a cached description was provided.
    if backend_desc is None:
        from agent_tools import describe_backend

        backend_desc = describe_backend(backend_base_url)

    graph = build_graph(llm, backend_desc)
    state_in = {
        "messages": [HumanMessage(content=question)],
        "backend_desc": backend_desc,
        "steps": 0,
        "in_scope": False,
        "result": None,
    }
    try:
        final = graph.invoke(state_in, {"recursion_limit": 2 * MAX_TOOL_STEPS + 6})
    except Exception as exc:  # never raise across the entry boundary
        return {
            "question": question,
            "answer": None,
            "in_scope": None,
            "tool_calls": [],
            "steps": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    result = final.get("result") or {}
    return {
        "question": question,
        "answer": result.get("answer"),
        "in_scope": result.get("in_scope"),
        "tool_calls": result.get("tool_calls", []),
        "steps": result.get("steps", 0),
        "error": None,
    }


def mermaid(backend_base_url: str = "http://127.0.0.1:8080/v1/forecast") -> str:
    """Return the graph's mermaid diagram (no LLM/backend calls needed)."""

    class _Stub:  # minimal stand-in; draw_mermaid never invokes the model
        def bind_tools(self, tools):
            return self

    return build_graph(_Stub(), backend_desc=None).get_graph().draw_mermaid()
