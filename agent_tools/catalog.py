"""List the weather variables MeteoRight understands.

Thin wrapper over the curated VARIABLE_SETS plus the circular-variable registry,
so an agent can discover valid variable names (and which ones are angular).
"""

from __future__ import annotations

from typing import Any

from config.settings import VARIABLE_SETS
from util.circular import CIRCULAR_VARIABLES

# Human-friendly units per variable, for labelling answers.
VARIABLE_UNITS: dict[str, str] = {
    "temperature_2m": "°C",
    "dew_point_2m": "°C",
    "precipitation": "mm",
    "wind_speed_10m": "km/h",
    "wind_direction_10m": "degrees",
    "wind_gusts_10m": "km/h",
    "pressure_msl": "hPa",
    "relative_humidity_2m": "%",
    "cloud_cover": "%",
}


def variable_units(variable: str) -> str | None:
    """Return the display units for a variable, or None if unknown."""
    return VARIABLE_UNITS.get(variable)


def list_variables() -> dict[str, Any]:
    """List known variables, grouped sets, units, and which are circular.

    Returns:
        ``{"all": [...], "sets": {"minimal": [...], ...},
           "circular": [...], "units": {var: unit}, "error": None}``
    """
    all_vars = sorted({v for group in VARIABLE_SETS.values() for v in group})
    return {
        "all": all_vars,
        "sets": {name: list(group) for name, group in VARIABLE_SETS.items()},
        "circular": sorted(CIRCULAR_VARIABLES),
        "units": {v: VARIABLE_UNITS.get(v) for v in all_vars},
        "error": None,
    }
