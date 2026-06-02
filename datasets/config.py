"""Configuration for curated research datasets.

Defines:
- Location presets (Copenhagen, Stockholm, etc.)
- Variable sets (minimal, standard, all)
- Default models and event definitions
- Pipeline configuration

Usage
-----
>>> from datasets.config import LOCATION_PRESETS, VARIABLE_SETS

>>> loc = LOCATION_PRESETS["copenhagen"]
>>> print(loc["name"], loc["lat"], loc["lon"])
Copenhagen 55.605 12.574

>>> vars = VARIABLE_SETS["all"]
>>> print(vars)
['temperature_2m', 'dew_point_2m', ...]
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Location presets
# ---------------------------------------------------------------------------

LOCATION_PRESETS: dict[str, dict[str, Any]] = {
    "copenhagen": {
        "name": "Copenhagen",
        "lat": 55.605,
        "lon": 12.574,
        "timezone": "Europe/Copenhagen",
    },
    "stockholm": {
        "name": "Stockholm",
        "lat": 59.329,
        "lon": 18.069,
        "timezone": "Europe/Stockholm",
    },
    "berlin": {
        "name": "Berlin",
        "lat": 52.520,
        "lon": 13.405,
        "timezone": "Europe/Berlin",
    },
}

# ---------------------------------------------------------------------------
# Variable sets
# ---------------------------------------------------------------------------

VARIABLE_SETS: dict[str, tuple[str, ...]] = {
    "minimal": (
        "temperature_2m",
        "precipitation",
    ),
    "standard": (
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "pressure_msl",
    ),
    "all": (
        "temperature_2m",
        "dew_point_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "pressure_msl",
        "relative_humidity_2m",
        "cloud_cover",
    ),
}

# ---------------------------------------------------------------------------
# Default models for common locations
# ---------------------------------------------------------------------------

DEFAULT_MODELS: dict[str, tuple[str, ...]] = {
    "copenhagen": ("icon_eu", "ecmwf_ifs_hres"),
    "stockholm": ("icon_eu", "ecmwf_ifs_hres"),
    "berlin": ("icon_eu", "ecmwf_ifs_hres"),
}

# ---------------------------------------------------------------------------
# Built-in event names (mirrors advanced_verification)
# ---------------------------------------------------------------------------

BUILT_IN_EVENTS = (
    "frost",
    "heavy_precip",
    "hot_day",
    "no_rain",
)

# Event definitions (mirrors advanced_verification.schema)
EVENT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "frost": {
        "variable": "temperature_2m",
        "threshold": 0.0,
        "operator": "le",
        "description": "Temperature at or below freezing (frost event)",
    },
    "heavy_precip": {
        "variable": "precipitation",
        "threshold": 5.0,
        "operator": "ge",
        "description": "Heavy precipitation event (>= 5 mm)",
    },
    "hot_day": {
        "variable": "temperature_2m",
        "threshold": 30.0,
        "operator": "ge",
        "description": "Hot day event (>= 30 °C)",
    },
    "no_rain": {
        "variable": "precipitation",
        "threshold": 0.1,
        "operator": "lt",
        "description": "Dry condition — precipitation effectively zero",
    },
}

# ---------------------------------------------------------------------------
# Metric groups to precompute
# ---------------------------------------------------------------------------

DEFAULT_METRIC_GROUPS = (
    "lead_hours",
    "season",
    "month",
    "model",
)

# ---------------------------------------------------------------------------
# Dataset name generation
# ---------------------------------------------------------------------------

def generate_dataset_name(
    location_name: str,
    start_date: str,
    end_date: str,
) -> str:
    """Generate a dataset directory name from location and date range.

    Examples:
        >>> generate_dataset_name("Copenhagen", "2024-01-01", "2025-12-31")
        'copenhagen_2y'
        >>> generate_dataset_name("Stockholm", "2023-06-01", "2024-05-31")
        'stockholm_1y'
    """
    import re

    # Clean location name to lowercase slug
    slug = re.sub(r"[^a-z0-9]+", "_", location_name.lower())
    slug = slug.strip("_")

    # Parse dates
    start_parts = start_date.split("-")
    end_parts = end_date.split("-")
    start_year = int(start_parts[0])
    end_year = int(end_parts[0])

    if start_year == end_year:
        year_suffix = f"{start_year}"
    else:
        # Inclusive range: 2024-01-01 to 2025-12-31 is 2 years
        year_span = (end_year - start_year) + 1
        year_suffix = f"{year_span}y"

    return f"{slug}_{year_suffix}"
