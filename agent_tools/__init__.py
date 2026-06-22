"""Agent-facing tool layer for MeteoRight.

A small set of explicit, composable callables that an LLM agent (or a human via
the CLI) can use to answer forecast-accuracy questions. Every tool returns a
plain JSON-serializable dict with an ``"error"`` key (``None`` on success, or
``{"type", "message"}`` on failure) so results stay well-formed for tool use and
no exception crosses the tool boundary.

The tools express intent in friendly terms (place names, friendly model names,
variable names); backend specifics (which models exist, what date range) are
discovered at runtime via :func:`agent_tools.backend.describe_backend` rather
than hardcoded, so the tools keep working as the backend changes.
"""

from agent_tools.backend import describe_backend

__all__ = ["describe_backend"]
