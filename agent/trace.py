"""Render the agent's flow through the graph, node by node, live.

``iter_flow`` drives ``graph.stream(stream_mode="updates")`` and yields one
``FlowEvent`` per node as it fires — a structured, testable description of what
that node did (scope verdict, the tool calls the agent proposed, a validation
pass/fix, each tool's result summary, the final synthesized answer).

``render_flow`` prints those events with rich, drawing the node sequence as a
flow. Both the one-shot ``--trace`` mode and the ``chat`` REPL use these, so the
flow looks identical in either.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console

from agent.graph import build_graph
from agent.llm import make_llm

# Short glyphs per node for the flow line.
_NODE_LABEL = {
    "scope_gate": "scope_gate",
    "agent": "agent",
    "validate_args": "validate",
    "tools": "tools",
    "synthesize": "synthesize",
    "refuse": "refuse",
}


@dataclass
class FlowEvent:
    """One node firing, with a human-readable summary and any detail lines."""

    node: str
    summary: str
    details: list[str] = field(default_factory=list)


@dataclass
class FlowResult:
    """The terminal outcome of a run, after all events."""

    answer: str | None
    in_scope: bool | None
    steps: int
    tool_calls: list[dict[str, Any]]


def _summarize_tool_result(raw: str) -> str:
    """Pull a compact headline out of a tool's JSON result string."""
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return (raw or "")[:80]
    if not isinstance(d, dict):
        return str(d)[:80]
    if d.get("error"):
        return f"error: {d['error'].get('type', 'error')} — {d['error'].get('message', '')[:60]}"
    ans = d.get("answer")
    if isinstance(ans, dict) and "metrics" in ans:
        metrics = ", ".join(f"{k}={v}" for k, v in ans["metrics"].items())
        return f"{ans.get('variable', '')} {metrics} {ans.get('units') or ''}".strip()
    if "ranking" in d:  # compare_models
        top = d["ranking"][0] if d["ranking"] else None
        if top:
            return f"best: {top['model_backend_name']} {d.get('metric')}={top['value']} ({len(d['ranking'])} models)"
        return "no models ranked"
    if "all" in d:  # list_variables
        return f"{len(d['all'])} variables"
    if "models" in d:  # list_models
        return f"{len(d['models'])} models"
    return "ok"


def _event_for(node: str, delta: dict[str, Any]) -> FlowEvent:
    """Translate a node's state delta into a FlowEvent."""
    if node == "scope_gate":
        verdict = "in-scope" if delta.get("in_scope") else "off-domain"
        return FlowEvent(node, f"classified question: {verdict}")

    if node == "refuse":
        return FlowEvent(node, "refused (off-domain)")

    if node == "agent":
        msgs = delta.get("messages") or []
        last = msgs[-1] if msgs else None
        if isinstance(last, AIMessage) and last.tool_calls:
            details = [
                f"{c['name']}({', '.join(f'{k}={v!r}' for k, v in (c.get('args') or {}).items())})"
                for c in last.tool_calls
            ]
            plural = "s" if len(details) != 1 else ""
            return FlowEvent(node, f"proposed {len(details)} tool call{plural}", details)
        return FlowEvent(node, "produced final answer")

    if node == "validate_args":
        msgs = delta.get("messages") or []
        corrections = [m for m in msgs if isinstance(m, ToolMessage)]
        if corrections:
            return FlowEvent(
                node, "rejected call(s) — sending corrections back to agent",
                [m.content[:90] for m in corrections],
            )
        return FlowEvent(node, "arguments valid")

    if node == "tools":
        msgs = delta.get("messages") or []
        details = [
            f"{m.name}: {_summarize_tool_result(m.content)}"
            for m in msgs
            if isinstance(m, ToolMessage)
        ]
        return FlowEvent(node, f"executed {len(details)} tool(s)", details)

    if node == "synthesize":
        return FlowEvent(node, "synthesized final answer")

    return FlowEvent(node, "")


def iter_flow(
    question: str,
    *,
    backend_base_url: str = "http://127.0.0.1:8080/v1/forecast",
    llm: Any | None = None,
    backend_desc: dict[str, Any] | None = None,
) -> Iterator[FlowEvent | FlowResult]:
    """Yield a FlowEvent per node as the graph runs, then a final FlowResult.

    Mirrors agent.run.answer's setup (probe once, thread backend_desc) but
    streams node updates instead of returning only the end state.
    """
    if llm is None:
        llm = make_llm()
    if backend_desc is None:
        from agent_tools import describe_backend

        backend_desc = describe_backend(backend_base_url)

    from langchain_core.messages import HumanMessage

    from agent.graph import MAX_TOOL_STEPS

    graph = build_graph(llm, backend_desc)
    state_in = {
        "messages": [HumanMessage(content=question)],
        "backend_desc": backend_desc,
        "steps": 0,
        "in_scope": False,
        "result": None,
    }
    final_result: dict[str, Any] | None = None
    for chunk in graph.stream(
        state_in, {"recursion_limit": 2 * MAX_TOOL_STEPS + 6}, stream_mode="updates"
    ):
        for node, delta in chunk.items():
            if not isinstance(delta, dict):
                continue
            yield _event_for(node, delta)
            if node in ("synthesize", "refuse") and delta.get("result"):
                final_result = delta["result"]

    result = final_result or {}
    yield FlowResult(
        answer=result.get("answer"),
        in_scope=result.get("in_scope"),
        steps=result.get("steps", 0),
        tool_calls=result.get("tool_calls", []),
    )


def render_flow(
    question: str,
    *,
    console: Console | None = None,
    **kwargs: Any,
) -> FlowResult:
    """Run a question and print its flow through the nodes; return the result."""
    console = console or Console()
    console.print(f"[bold cyan]?[/bold cyan] {question}")
    result: FlowResult | None = None
    first = True
    for item in iter_flow(question, **kwargs):
        if isinstance(item, FlowResult):
            result = item
            continue
        if not first:
            console.print("  [dim]│[/dim]")
            console.print("  [dim]▼[/dim]")
        first = False
        console.print(f"  [bold]{_NODE_LABEL.get(item.node, item.node)}[/bold]"
                      f" [dim]—[/dim] {item.summary}")
        for d in item.details:
            console.print(f"      [dim]· {d}[/dim]")

    console.print()
    if result and result.answer:
        console.print(f"[bold green]✓[/bold green] {result.answer}")
    elif result:
        console.print("[yellow]No answer produced.[/yellow]")
    return result or FlowResult(None, None, 0, [])
