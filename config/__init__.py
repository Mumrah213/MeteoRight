"""Configuration for the weather pipeline.

Location presets, variable sets, default models, and event definitions.

Usage
-----
>>> from config import LOCATION_PRESETS, VARIABLE_SETS
>>> loc = LOCATION_PRESETS["copenhagen"]
>>> vars = VARIABLE_SETS["minimal"]
"""

from .settings import (
    BUILT_IN_EVENTS,
    DEFAULT_METRIC_GROUPS,
    DEFAULT_MODELS,
    EVENT_DEFINITIONS,
    LOCATION_PRESETS,
    VARIABLE_SETS,
    generate_dataset_name,
)

__all__ = [
    "BUILT_IN_EVENTS",
    "DEFAULT_METRIC_GROUPS",
    "DEFAULT_MODELS",
    "EVENT_DEFINITIONS",
    "LOCATION_PRESETS",
    "VARIABLE_SETS",
    "generate_dataset_name",
]
