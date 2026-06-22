"""Offline tests for the constrained agent graph (no network, no real LLM).

A scripted stub chat model drives the graph deterministically so we can assert
each control-flow path: in-scope -> tool -> synthesize, off-scope -> refuse,
invalid-arg -> correction -> retry, and the step cap.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import MAX_TOOL_STEPS, build_graph

# A backend description with two models, injected so no probe runs.
BACKEND_DESC = {
    "base_url": "http://test/v1/forecast",
    "reachable": True,
    "models": [
        {"backend_name": "ecmwf_ifs025", "has_data": True,
         "coverage": {"start": "2026-05-07", "end": "2026-05-18"}},
        {"backend_name": "dwd_icon", "has_data": True,
         "coverage": {"start": "2026-05-07", "end": "2026-05-18"}},
    ],
    "observations": {"source": "archive", "url": "http://test/v1/archive",
                     "coverage": {"start": "2026-05-07", "end": "2026-05-18"}},
    "error": None,
}


class ScriptedLLM:
    """Stub chat model.

    - For the scope gate (a single HumanMessage containing 'Answer with a single
      word') it returns `scope_verdict`.
    - For agent turns it pops the next AIMessage from `agent_script`.
    """

    def __init__(self, scope_verdict: str, agent_script: list[AIMessage]):
        self.scope_verdict = scope_verdict
        self._script = list(agent_script)
        self.tools = None

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        text = " ".join(
            m.content for m in messages if isinstance(m, HumanMessage) and m.content
        )
        if "single\nword" in text or "single word" in text:
            return AIMessage(content=self.scope_verdict)
        if self._script:
            return self._script.pop(0)
        return AIMessage(content="done")


def _tool_call(name, args, cid="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}])


def test_off_scope_refuses_without_tools(monkeypatch):
    llm = ScriptedLLM("OUT", [])
    graph = build_graph(llm, BACKEND_DESC)
    out = graph.invoke(
        {"messages": [HumanMessage(content="write me a poem")],
         "backend_desc": BACKEND_DESC, "steps": 0, "in_scope": False, "result": None},
        {"recursion_limit": 20},
    )
    assert out["result"]["in_scope"] is False
    assert out["result"]["tool_calls"] == []
    assert "forecast accuracy" in out["result"]["answer"].lower()


def test_in_scope_tool_then_synthesize(monkeypatch):
    # Patch the real tool so no network: forecast_accuracy returns a canned dict.
    import agent.tools as toolsmod

    def fake_fa(variable, area, **kw):
        return {"answer": {"variable": variable, "metrics": {"mae": 1.23}, "units": "°C"},
                "error": None}

    monkeypatch.setattr(toolsmod, "forecast_accuracy", fake_fa)

    script = [
        _tool_call("forecast_accuracy", {"variable": "temperature_2m", "area": "Copenhagen"}),
        AIMessage(content="Temperature MAE is 1.23 °C."),
    ]
    llm = ScriptedLLM("IN", script)
    graph = build_graph(llm, BACKEND_DESC)
    out = graph.invoke(
        {"messages": [HumanMessage(content="how accurate is temperature for Copenhagen?")],
         "backend_desc": BACKEND_DESC, "steps": 0, "in_scope": False, "result": None},
        {"recursion_limit": 20},
    )
    res = out["result"]
    assert res["in_scope"] is True
    assert res["answer"] == "Temperature MAE is 1.23 °C."
    assert len(res["tool_calls"]) == 1
    assert res["tool_calls"][0]["tool"] == "forecast_accuracy"
    assert "1.23" in res["tool_calls"][0]["result"]


def test_invalid_arg_corrected_then_retry(monkeypatch):
    import agent.tools as toolsmod

    monkeypatch.setattr(
        toolsmod, "forecast_accuracy",
        lambda variable, area, **kw: {"answer": {"metrics": {"mae": 2.0}}, "error": None},
    )
    # First the model asks for a bogus variable -> validator corrects -> retry valid.
    script = [
        _tool_call("forecast_accuracy", {"variable": "unicorns", "area": "Copenhagen"}, "c1"),
        _tool_call("forecast_accuracy", {"variable": "temperature_2m", "area": "Copenhagen"}, "c2"),
        AIMessage(content="Corrected and answered: MAE 2.0."),
    ]
    llm = ScriptedLLM("IN", script)
    graph = build_graph(llm, BACKEND_DESC)
    out = graph.invoke(
        {"messages": [HumanMessage(content="accuracy of unicorns in Copenhagen?")],
         "backend_desc": BACKEND_DESC, "steps": 0, "in_scope": False, "result": None},
        {"recursion_limit": 30},
    )
    # The bogus call must NOT have executed (no successful tool result for it);
    # the retry must have produced the answer.
    assert out["result"]["answer"] == "Corrected and answered: MAE 2.0."
    contents = [tc["result"] for tc in out["result"]["tool_calls"]]
    assert any("not valid" in c for c in contents)  # the correction was surfaced
    assert any("2.0" in c for c in contents)  # the valid retry ran


def test_step_cap_forces_synthesize(monkeypatch):
    import agent.tools as toolsmod

    monkeypatch.setattr(
        toolsmod, "forecast_accuracy",
        lambda variable, area, **kw: {"answer": {"metrics": {"mae": 1.0}}, "error": None},
    )
    # Model loops forever asking for valid tools; the cap must stop it.
    looping = [
        _tool_call("forecast_accuracy", {"variable": "temperature_2m", "area": "Copenhagen"}, f"c{i}")
        for i in range(MAX_TOOL_STEPS + 5)
    ]
    llm = ScriptedLLM("IN", looping)
    graph = build_graph(llm, BACKEND_DESC)
    out = graph.invoke(
        {"messages": [HumanMessage(content="temperature Copenhagen")],
         "backend_desc": BACKEND_DESC, "steps": 0, "in_scope": False, "result": None},
        {"recursion_limit": 100},
    )
    # It terminated (didn't hit recursion error) and capped at MAX_TOOL_STEPS.
    assert out["result"]["steps"] <= MAX_TOOL_STEPS


def test_mermaid_renders():
    from agent.run import mermaid

    diagram = mermaid()
    for node in ("scope_gate", "agent", "validate_args", "tools", "synthesize", "refuse"):
        assert node in diagram


def test_iter_flow_emits_node_sequence(monkeypatch):
    """The trace renderer's event stream follows the graph node order."""
    import agent.tools as toolsmod
    from agent.trace import FlowEvent, FlowResult, iter_flow

    monkeypatch.setattr(
        toolsmod, "forecast_accuracy",
        lambda variable, area, **kw: {
            "answer": {"variable": variable, "metrics": {"mae": 1.24}, "units": "°C"},
            "error": None,
        },
    )
    script = [
        _tool_call("forecast_accuracy", {"variable": "temperature_2m", "area": "Copenhagen"}),
        AIMessage(content="Temperature MAE is 1.24 °C."),
    ]
    llm = ScriptedLLM("IN", script)
    items = list(iter_flow("how accurate is temperature for Copenhagen?",
                           llm=llm, backend_desc=BACKEND_DESC))
    events = [i for i in items if isinstance(i, FlowEvent)]
    result = next(i for i in items if isinstance(i, FlowResult))

    assert [e.node for e in events] == [
        "scope_gate", "agent", "validate_args", "tools", "agent", "synthesize"
    ]
    # The tool node's detail line carries the result summary.
    tool_event = next(e for e in events if e.node == "tools")
    assert "mae=1.24" in " ".join(tool_event.details)
    assert result.answer == "Temperature MAE is 1.24 °C."
    assert result.in_scope is True


def test_iter_flow_off_scope_short_circuits(monkeypatch):
    from agent.trace import FlowEvent, FlowResult, iter_flow

    items = list(iter_flow("write me a poem", llm=ScriptedLLM("OUT", []),
                           backend_desc=BACKEND_DESC))
    events = [i for i in items if isinstance(i, FlowEvent)]
    result = next(i for i in items if isinstance(i, FlowResult))
    assert [e.node for e in events] == ["scope_gate", "refuse"]
    assert result.in_scope is False
