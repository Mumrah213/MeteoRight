"""Live end-to-end test of the agent against bazzite LLM + local backend.

Marked ``network``; skips when either the LLM or the MeteoRight backend is
unreachable, so it never breaks the default suite. Run with:

    pytest -m network tests/test_agent_live.py
"""

import pytest

pytestmark = pytest.mark.network

BACKEND = "http://127.0.0.1:8080/v1/forecast"


@pytest.fixture(scope="module")
def backend_desc():
    from agent_tools import describe_backend

    desc = describe_backend(BACKEND)
    if not desc.get("reachable"):
        pytest.skip(f"MeteoRight backend not reachable at {BACKEND}")
    return desc


@pytest.fixture(scope="module")
def llm():
    from agent.llm import make_llm

    model = make_llm()
    try:
        from langchain_core.messages import HumanMessage

        model.invoke([HumanMessage(content="ping")])
    except Exception as exc:  # endpoint down / unreachable
        pytest.skip(f"LLM endpoint not reachable: {exc}")
    return model


def test_agent_answers_temperature(backend_desc, llm):
    from agent.run import answer

    r = answer(
        "How accurate is the temperature forecast for the Malmö-Copenhagen area?",
        backend_base_url=BACKEND, llm=llm, backend_desc=backend_desc,
    )
    assert r["error"] is None, r["error"]
    assert r["in_scope"] is True
    # At least one tool ran and the answer references a number.
    assert r["tool_calls"], "expected at least one tool call"
    assert any(ch.isdigit() for ch in (r["answer"] or ""))


def test_agent_refuses_off_domain(backend_desc, llm):
    from agent.run import answer

    r = answer("Write me a haiku about cats.", backend_base_url=BACKEND, llm=llm,
               backend_desc=backend_desc)
    assert r["in_scope"] is False
    assert r["tool_calls"] == []
