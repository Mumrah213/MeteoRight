"""Validate a proposed tool call against the data catalog before executing it.

This is the guardrail that keeps the agent honest: if it asks for a variable,
model, or metric that doesn't exist, we don't execute and surface a confusing
backend error — we hand the model the list of allowed values so it can correct
itself. Returns ``None`` when the call is fine, or a correction string to feed
back as a tool message.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agent_tools import list_models, list_variables, resolve_model_alias

_VALID_METRICS = {"mae", "rmse", "bias"}


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def validate_tool_call(
    name: str, args: dict[str, Any], backend_desc: dict[str, Any] | None
) -> str | None:
    """Check one tool call. Return None if valid, else a correction message.

    Only the catalog-bound tools (forecast_accuracy, compare_models) are
    checked; list_* and resolve_area take no constrained args.
    """
    if name not in ("forecast_accuracy", "compare_models"):
        return None

    problems: list[str] = []

    # Variable must be a known one.
    variable = args.get("variable")
    valid_vars = list_variables()["all"]
    if variable not in valid_vars:
        problems.append(
            f"variable '{variable}' is not valid. Choose one of: {', '.join(valid_vars)}."
        )

    # Model(s) must resolve to a backend domain that actually has data.
    available = {m["backend_name"] for m in list_models(backend_desc=backend_desc)["models"]}
    if available:
        requested = [args["model"]] if name == "forecast_accuracy" and args.get("model") else []
        for m in requested:
            if resolve_model_alias(m) not in available:
                problems.append(
                    f"model '{m}' is not available. Available models: {', '.join(sorted(available))}."
                )

    # Metric(s) constrained to mae/rmse/bias.
    metric = args.get("metric")
    if metric is not None and metric not in _VALID_METRICS:
        problems.append(f"metric '{metric}' is not valid. Choose one of: mae, rmse, bias.")

    # Dates, if supplied, must be ISO.
    for key in ("start", "end"):
        val = args.get(key)
        if val is not None and not _valid_date(val):
            problems.append(f"{key} '{val}' is not a valid YYYY-MM-DD date.")

    if not problems:
        return None
    return "Cannot run that tool call: " + " ".join(problems) + " Please retry with valid values."
