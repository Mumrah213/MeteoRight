"""LLM-facing tools: thin wrappers over agent_tools with infra args injected.

The LLM only ever sees agent-decision parameters (variable, area, model, dates,
metric). Infrastructure — backend URL, the cached backend probe, the
offline-geocoding policy — is bound here via closures so the model cannot set
it. Each tool returns the same JSON dict the underlying agent_tools callable
returns (with its ``{"error": ...}`` envelope), serialized to a string for the
tool message.

``build_tools(backend_desc)`` returns the list bound to LangGraph's ToolNode;
pass the cached ``describe_backend()`` result so the backend is probed once.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agent_tools import (
    compare_models,
    forecast_accuracy,
    list_models,
    list_variables,
    resolve_area,
)

# Names the agent is allowed to call. The ToolNode is built from exactly these.
EXPOSED_TOOL_NAMES = (
    "forecast_accuracy",
    "compare_models",
    "list_variables",
    "list_models",
    "resolve_area",
)


def _dumps(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, default=str)


# --- Argument schemas (only agent-decision params; infra is injected) --------


class ForecastAccuracyArgs(BaseModel):
    variable: str = Field(description="Weather variable, e.g. temperature_2m, precipitation")
    area: str = Field(description='Place or area, e.g. "Copenhagen" or "Malmö-Copenhagen"')
    model: str = Field(default="ecmwf", description="Friendly model name, e.g. ecmwf, icon")
    start: str | None = Field(default=None, description="ISO start date YYYY-MM-DD (optional)")
    end: str | None = Field(default=None, description="ISO end date YYYY-MM-DD (optional)")


class CompareModelsArgs(BaseModel):
    variable: str = Field(description="Weather variable, e.g. temperature_2m")
    area: str = Field(description='Place or area, e.g. "Malmö-Copenhagen"')
    metric: str = Field(default="mae", description="Ranking metric: mae, rmse, or bias")
    start: str | None = Field(default=None, description="ISO start date (optional)")
    end: str | None = Field(default=None, description="ISO end date (optional)")


class ResolveAreaArgs(BaseModel):
    area: str = Field(description='Place or area to resolve, e.g. "Malmö-Copenhagen"')


class NoArgs(BaseModel):
    pass


def build_tools(backend_desc: dict[str, Any] | None) -> list[StructuredTool]:
    """Build the whitelisted LLM tools with infra bound from ``backend_desc``."""

    def _forecast_accuracy(variable, area, model="ecmwf", start=None, end=None) -> str:
        return _dumps(
            forecast_accuracy(
                variable, area, model=model, start=start, end=end,
                backend_desc=backend_desc, use_network_geocode=False,
            )
        )

    def _compare_models(variable, area, metric="mae", start=None, end=None) -> str:
        return _dumps(
            compare_models(
                variable, area, metric=metric, start=start, end=end,
                backend_desc=backend_desc, use_network_geocode=False,
            )
        )

    def _list_variables() -> str:
        return _dumps(list_variables())

    def _list_models() -> str:
        return _dumps(list_models(backend_desc=backend_desc))

    def _resolve_area(area) -> str:
        return _dumps(resolve_area(area, use_network=False))

    return [
        StructuredTool.from_function(
            _forecast_accuracy,
            name="forecast_accuracy",
            description=(
                "Compute forecast accuracy (MAE/RMSE/bias) for one weather variable "
                "from one model over an area. Use for 'how accurate is X for Y'."
            ),
            args_schema=ForecastAccuracyArgs,
        ),
        StructuredTool.from_function(
            _compare_models,
            name="compare_models",
            description=(
                "Rank the available models by accuracy for one variable over an area. "
                "Use for 'which model is best' / 'compare models'."
            ),
            args_schema=CompareModelsArgs,
        ),
        StructuredTool.from_function(
            _list_variables,
            name="list_variables",
            description="List valid weather variables, which are circular, and their units.",
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            _list_models,
            name="list_models",
            description="List the forecast models available on the backend, with coverage.",
            args_schema=NoArgs,
        ),
        StructuredTool.from_function(
            _resolve_area,
            name="resolve_area",
            description="Inspect how a place/area resolves to a grid point (no accuracy).",
            args_schema=ResolveAreaArgs,
        ),
    ]
