"""Circular (angular) error for direction-like variables.

Some weather variables are angles in degrees (e.g. wind direction) where the
raw difference forecast - observation is wrong near the 0/360 wraparound: a
forecast of 10 against an observation of 350 is a 20 error, not 340.

This module is the single source of truth for which variables are circular and
how their error is computed. Both the live DataFrame/parquet verification path
and the record-based path import from here so the convention can never drift.

The error is the shortest signed angle wrapped to [-180, +180): positive means
the forecast is clockwise of the observation, negative counter-clockwise.
"""

from __future__ import annotations

# Variables treated as circular (0-360 degree phase angles).
CIRCULAR_VARIABLES = frozenset({"wind_direction_10m"})


def is_circular(variable: str) -> bool:
    """Return True if the variable should use shortest-angle error."""
    return variable in CIRCULAR_VARIABLES


def circular_error(forecast: float, observation: float) -> float:
    """Shortest signed angular error for a single pair, in [-180, +180)."""
    return ((forecast - observation + 180.0) % 360.0) - 180.0


def circular_error_series(forecast, observation):
    """Vectorized shortest signed angular error for pandas Series / arrays.

    Accepts anything supporting elementwise arithmetic and modulo (pandas
    Series, numpy arrays). NaN propagates as with plain subtraction.
    """
    return ((forecast - observation + 180.0) % 360.0) - 180.0
