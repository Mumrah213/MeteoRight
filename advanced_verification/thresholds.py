"""Threshold comparison semantics for event-based verification.

Defines how continuous forecast and observation values are converted into
binary event flags using configurable thresholds and comparison operators.

Comparison Semantics
--------------------

Each operator explicitly specifies inclusive/exclusive behavior:

    Operator   Symbol   Inclusive   Example (threshold=0)
    ---------- -------- ----------- ---------------------
    le         <=       Yes         temp <= 0  →  -0.1 is event, 0.0 is event, 0.1 is not
    lt         <        No          temp < 0   →  -0.1 is event, 0.0 is NOT event
    ge         >=       Yes         temp >= 0  →  0.0 is event, 0.1 is event, -0.1 is not
    gt         >        No          temp > 0   →  0.0 is NOT event, 0.1 is event
    eq         ==       Yes         temp == 0  →  only exactly 0.0 is event
    ne         !=       No          temp != 0  →  everything except 0.0 is event

NaN Handling
------------

If the input value is NaN:
    - The comparison result is always False (not an event).
    - NaN values are never treated as equal to any value, including NaN.
    - This is consistent with IEEE 754 floating-point semantics.

Unit Consistency
----------------

The threshold value must be in the same units as the variable:
    - temperature_2m: threshold in °C
    - precipitation:  threshold in mm

The caller is responsible for ensuring unit consistency. No unit conversion
is performed — this module is agnostic to physical units.

Design Principle
----------------

This module is intentionally simple and explicit. There are no hidden
assumptions about which operator to use or what the threshold should be.
Every comparison is a direct function call with documented behavior.

Usage
-----
>>> from advanced_verification.thresholds import compare_threshold
>>> compare_threshold(-1.0, 0.0, "le")   # True  (−1 ≤ 0)
>>> compare_threshold(0.0, 0.0, "le")    # True  (0 ≤ 0, inclusive)
>>> compare_threshold(0.1, 0.0, "le")    # False (0.1 > 0)
>>> compare_threshold(float("nan"), 0.0, "le")  # False (NaN → not event)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Valid operator names
ValidOperator = Literal["le", "lt", "ge", "gt", "eq", "ne"]


# Operator registry — maps operator name to comparison function
# Each function takes (value, threshold) and returns bool
def _op_le(value: float, threshold: float) -> bool:
    """Less than or equal (inclusive)."""
    return value <= threshold


def _op_lt(value: float, threshold: float) -> bool:
    """Strictly less than (exclusive)."""
    return value < threshold


def _op_ge(value: float, threshold: float) -> bool:
    """Greater than or equal (inclusive)."""
    return value >= threshold


def _op_gt(value: float, threshold: float) -> bool:
    """Strictly greater than (exclusive)."""
    return value > threshold


def _op_eq(value: float, threshold: float) -> bool:
    """Equal to."""
    return value == threshold


def _op_ne(value: float, threshold: float) -> bool:
    """Not equal to."""
    return value != threshold


OPERATOR_FUNCTIONS: dict[str, callable] = {
    "le": _op_le,
    "lt": _op_lt,
    "ge": _op_ge,
    "gt": _op_gt,
    "eq": _op_eq,
    "ne": _op_ne,
}


def compare_threshold(value: float, threshold: float, operator: str) -> bool:
    """Compare a single value against a threshold using the specified operator.

    Args:
        value: The value to compare (forecast or observation).
        threshold: The threshold value to compare against.
        operator: Comparison operator — one of "le", "lt", "ge", "gt", "eq", "ne".
            - "le": value <= threshold (inclusive)
            - "lt": value < threshold (exclusive)
            - "ge": value >= threshold (inclusive)
            - "gt": value > threshold (exclusive)
            - "eq": value == threshold
            - "ne": value != threshold

    Returns:
        True if the comparison is satisfied, False otherwise.

        Special cases:
            - If value is NaN → returns False (not an event)
            - NaN is never equal to any value, including itself

    Raises:
        ValueError: If operator is not recognized.

    Examples:
        >>> compare_threshold(-1.0, 0.0, "le")
        True
        >>> compare_threshold(0.0, 0.0, "le")
        True
        >>> compare_threshold(0.0, 0.0, "lt")
        False
        >>> compare_threshold(float("nan"), 0.0, "le")
        False
    """
    if operator not in OPERATOR_FUNCTIONS:
        raise ValueError(
            f"Unknown operator '{operator}'. Valid operators: {', '.join(sorted(OPERATOR_FUNCTIONS.keys()))}"
        )

    # NaN handling: NaN comparisons always return False
    if not np.isfinite(value):
        return False

    return OPERATOR_FUNCTIONS[operator](value, threshold)


def apply_threshold_series(
    series: pd.Series,
    threshold: float,
    operator: str,
) -> pd.Series:
    """Apply a threshold comparison to a pandas Series.

    Converts a numeric series into a boolean series indicating whether
    each value meets the threshold condition.

    Args:
        series: Pandas Series of numeric values.
        threshold: Threshold value to compare against.
        operator: Comparison operator — one of "le", "lt", "ge", "gt", "eq", "ne".

    Returns:
        Boolean Series — True where the condition is met, False otherwise.
            NaN values in the input produce False in the output.

    Raises:
        ValueError: If operator is not recognized.

    Examples:
        >>> import pandas as pd
        >>> temps = pd.Series([-1.0, 0.0, 1.0, float("nan")])
        >>> apply_threshold_series(temps, 0.0, "le")
        0     True
        1     True
        2    False
        3    False
        dtype: bool
    """
    if operator not in OPERATOR_FUNCTIONS:
        raise ValueError(
            f"Unknown operator '{operator}'. Valid operators: {', '.join(sorted(OPERATOR_FUNCTIONS.keys()))}"
        )

    # Use numpy for vectorized comparison — np.isfinite handles NaN correctly
    values = series.values.astype(float)
    finite_mask = np.isfinite(values)

    result = np.zeros(len(series), dtype=bool)

    if operator == "le":
        result[finite_mask] = values[finite_mask] <= threshold
    elif operator == "lt":
        result[finite_mask] = values[finite_mask] < threshold
    elif operator == "ge":
        result[finite_mask] = values[finite_mask] >= threshold
    elif operator == "gt":
        result[finite_mask] = values[finite_mask] > threshold
    elif operator == "eq":
        result[finite_mask] = values[finite_mask] == threshold
    elif operator == "ne":
        result[finite_mask] = values[finite_mask] != threshold

    return pd.Series(result, index=series.index, dtype=bool)
