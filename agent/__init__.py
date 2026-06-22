"""Constrained LangGraph agent that orchestrates the MeteoRight tool layer.

A natural-language question ("how accurate is temperature for Malmö-Copenhagen?")
is turned into the right agent_tools calls and a synthesized answer, inside an
explicit graph that bounds what the agent may do: a whitelisted tool set,
argument validation against the data catalog, a scope gate, and a step cap.
"""

__all__ = ["answer"]


def answer(*args, **kwargs):
    """Lazy entry point — see agent.run.answer."""
    from agent.run import answer as _answer

    return _answer(*args, **kwargs)
