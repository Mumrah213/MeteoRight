"""The constrained agent graph.

Topology (see plan mermaid):

    scope_gate ──in──> agent ──tool_calls──> validate_args ──valid──> tools ──┐
        │ out                  │ done/cap                  │ invalid           │
        v                      v                           v                   │
     refuse                synthesize <───────────────── agent <──────────────┘

The agent chooses which whitelisted tools to call and with what arguments, but
cannot leave the graph: every tool call is validated against the catalog, the
loop is capped, and off-domain questions are refused before any tool runs.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.prompts import REFUSAL_MESSAGE, SCOPE_GATE_PROMPT, SYSTEM_PROMPT
from agent.tools import build_tools
from agent.validation import validate_tool_call

# Max agent<->tools round-trips before we force a synthesis.
MAX_TOOL_STEPS = 6


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    backend_desc: dict[str, Any] | None
    steps: int
    in_scope: bool
    result: dict[str, Any] | None


def build_graph(llm: BaseChatModel, backend_desc: dict[str, Any] | None):
    """Compile the agent graph for a given LLM and cached backend probe."""
    tools = build_tools(backend_desc)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # --- nodes ----------------------------------------------------------------

    def scope_gate(state: AgentState) -> dict[str, Any]:
        question = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        verdict = llm.invoke([HumanMessage(content=SCOPE_GATE_PROMPT.format(question=question))])
        text = (verdict.content or "").strip().upper()
        return {"in_scope": text.startswith("IN")}

    def refuse(state: AgentState) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content=REFUSAL_MESSAGE)],
            "result": {"answer": REFUSAL_MESSAGE, "in_scope": False, "tool_calls": []},
        }

    def agent(state: AgentState) -> dict[str, Any]:
        msgs = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs = [SystemMessage(content=SYSTEM_PROMPT), *msgs]
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response]}

    def validate_args(state: AgentState) -> dict[str, Any]:
        """Intercept the agent's tool calls; validate all-or-nothing.

        If every call is valid we append nothing and let the ToolNode run them.
        If ANY call is invalid we answer EVERY call with a ToolMessage (a
        correction for the bad ones, a "skipped" note for the good ones) so no
        tool actually executes and the agent gets a clean turn to retry.
        """
        last = state["messages"][-1]
        problems = {
            call["id"]: validate_tool_call(
                call["name"], call.get("args", {}), state["backend_desc"]
            )
            for call in last.tool_calls
        }
        any_invalid = any(p for p in problems.values())
        messages: list[ToolMessage] = []
        if any_invalid:
            for call in last.tool_calls:
                problem = problems[call["id"]]
                content = problem or "Skipped: another tool call in this turn was invalid; retry all."
                messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
        # Count one step per agent turn that proposed tools.
        return {"messages": messages, "steps": state["steps"] + 1}

    def synthesize(state: AgentState) -> dict[str, Any]:
        """Collect the final answer text + the structured tool results used."""
        final_text = ""
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and not m.tool_calls:
                final_text = m.content or ""
                break
        tool_results = [
            {"tool": m.name, "result": m.content}
            for m in state["messages"]
            if isinstance(m, ToolMessage)
        ]
        return {
            "result": {
                "answer": final_text,
                "in_scope": True,
                "tool_calls": tool_results,
                "steps": state["steps"],
            }
        }

    # --- edges ----------------------------------------------------------------

    def after_scope(state: AgentState) -> Literal["agent", "refuse"]:
        return "agent" if state["in_scope"] else "refuse"

    def after_agent(state: AgentState) -> Literal["validate_args", "synthesize"]:
        last = state["messages"][-1]
        has_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
        if has_calls and state["steps"] < MAX_TOOL_STEPS:
            return "validate_args"
        return "synthesize"

    # validate_args appended a correction ToolMessage for each INVALID call. We
    # must not run ToolNode when any call was invalid (partial execution would
    # be confusing), so route to the agent if every call was corrected.
    def route_after_validate(state: AgentState) -> Literal["tools", "agent"]:
        # The AIMessage with tool_calls is the most recent NON-ToolMessage.
        # Find it and compare its call ids to any correction ToolMessages that
        # follow it.
        ai_idx = max(
            i for i, m in enumerate(state["messages"]) if isinstance(m, AIMessage) and m.tool_calls
        )
        ai_msg = state["messages"][ai_idx]
        corrected_ids = {
            m.tool_call_id
            for m in state["messages"][ai_idx + 1 :]
            if isinstance(m, ToolMessage)
        }
        all_ids = {c["id"] for c in ai_msg.tool_calls}
        # If every call already has a (correction) ToolMessage, none need running.
        return "agent" if corrected_ids == all_ids and all_ids else "tools"

    # --- assembly -------------------------------------------------------------

    g = StateGraph(AgentState)
    g.add_node("scope_gate", scope_gate)
    g.add_node("refuse", refuse)
    g.add_node("agent", agent)
    g.add_node("validate_args", validate_args)
    g.add_node("tools", tool_node)
    g.add_node("synthesize", synthesize)

    g.set_entry_point("scope_gate")
    g.add_conditional_edges("scope_gate", after_scope, {"agent": "agent", "refuse": "refuse"})
    g.add_edge("refuse", END)
    g.add_conditional_edges(
        "agent", after_agent, {"validate_args": "validate_args", "synthesize": "synthesize"}
    )
    g.add_conditional_edges(
        "validate_args", route_after_validate, {"tools": "tools", "agent": "agent"}
    )
    g.add_edge("tools", "agent")
    g.add_edge("synthesize", END)

    return g.compile()
