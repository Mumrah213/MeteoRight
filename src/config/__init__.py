"""Unified configuration for the weather pipeline.

Combines v2's data architecture config (locations, variables, events)
with docker's ingestion config (HTTP transport, providers).

Usage
-----
>>> from src.config import PipelineConfig
>>> config = PipelineConfig()
>>> print(config.get_location("copenhagen"))
>>> print(config.get_variables("minimal"))
"""

from .models import ForecastConfig, HTTPConfig, PipelineConfig, StorageConfig
from .settings import (
    BUILT_IN_EVENTS,
    DEFAULT_METRIC_GROUPS,
    DEFAULT_MODELS,
    EVENT_DEFINITIONS,
    LOCATION_PRESETS,
    VARIABLE_SETS,
)

__all__ = [
    "BUILT_IN_EVENTS",
    "DEFAULT_METRIC_GROUPS",
    "DEFAULT_MODELS",
    "EVENT_DEFINITIONS",
    "LOCATION_PRESETS",
    "VARIABLE_SETS",
    "ForecastConfig",
    "HTTPConfig",
    "PipelineConfig",
    "StorageConfig",
]
