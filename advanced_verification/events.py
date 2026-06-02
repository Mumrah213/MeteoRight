"""Event generation utilities for forecast verification.

Converts continuous forecast and observation values into binary event flags
using configurable thresholds and comparison operators.

Event Definition
----------------
An event is defined by three parameters:
    1. variable   : column name in the verification DataFrame (e.g., "temperature_2m")
    2. threshold  : numeric threshold value (e.g., 0.0 for frost)
    3. operator   : comparison operator ("le", "lt", "ge", "gt", "eq", "ne")

For each row in the verification DataFrame:
    event_forecast = (forecast_variable OP threshold)
    event_observed = (observed_variable OP threshold)

NaN Handling
------------
    - NaN forecast value → event_forecast = False (not an event)
    - NaN observed value → event_observed = False (not an event)
    - Both NaN → row excluded from all subsequent computations

Built-in Events
---------------
The module provides several built-in events with meteorologically meaningful
thresholds:

    Event          | Variable        | Threshold | Operator | Description
    ---------------|-----------------|-----------|----------|-------------------------
    frost          | temperature_2m  | 0.0       | le       | Temperature ≤ 0°C
    heavy_precip   | precipitation   | 5.0       | ge       | Precipitation ≥ 5 mm
    hot_day        | temperature_2m  | 30.0      | ge       | Temperature ≥ 30°C
    no_rain        | precipitation   | 0.1       | lt       | Precipitation < 0.1 mm

Usage
-----
>>> from advanced_verification.events import generate_events
>>> df = read_verification("data/verification/verification.parquet")
>>> events_df = generate_events(df, event_name="frost")
>>> # Returns DataFrame with event_forecast_frost and event_observed_frost columns
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from .schema import (
    BUILTIN_EVENT_DEFS,
    BUILTIN_EVENTS,
    VALID_OPERATORS,
    forecast_event_col,
    observed_event_col,
)
from .thresholds import apply_threshold_series

logger = logging.getLogger(__name__)

# Type alias for event operator
EventOperator = Literal["le", "lt", "ge", "gt", "eq", "ne"]


def get_event_definition(event_name: str) -> dict:
    """Get the definition for a built-in event.

    Args:
        event_name: Event name (e.g., "frost", "heavy_precip").

    Returns:
        Dict with keys: variable, threshold, operator, description.

    Raises:
        ValueError: If event_name is not a recognized built-in event.
    """
    if event_name not in BUILTIN_EVENT_DEFS:
        raise ValueError(
            f"Unknown event '{event_name}'. Built-in events: {', '.join(sorted(BUILTIN_EVENTS))}"
        )
    return BUILTIN_EVENT_DEFS[event_name].copy()


def generate_events(
    df: pd.DataFrame,
    event_name: str | None = None,
    variable: str | None = None,
    threshold: float | None = None,
    operator: str | None = None,
) -> pd.DataFrame:
    """Generate binary event columns from a verification DataFrame.

    Adds two boolean columns to the DataFrame:
        event_forecast_{name} : did the forecast predict the event?
        event_observed_{name} : did the event actually occur?

    Either use a built-in event name OR specify variable/threshold/operator manually.

    Args:
        df: Verification DataFrame from read_verification() or cmd_verify().
            Must contain forecast and observed variable columns.
        event_name: Name of a built-in event (e.g., "frost", "heavy_precip").
            If provided, variable, threshold, and operator are derived from
            the built-in definition.
        variable: Variable name to check (e.g., "temperature_2m").
            Required if event_name is not provided.
        threshold: Threshold value (e.g., 0.0 for frost).
            Required if event_name is not provided.
        operator: Comparison operator ("le", "lt", "ge", "gt", "eq", "ne").
            Required if event_name is not provided.

    Returns:
        DataFrame with original columns plus two new boolean columns:
            event_forecast_{event_name} and event_observed_{event_name}.

    Raises:
        ValueError: If required parameters are missing or invalid.

    Example:
        >>> df = read_verification("data/verification/verification.parquet")
        >>> events_df = generate_events(df, event_name="frost")
        >>> print(events_df.columns)
        # ... event_forecast_frost, event_observed_frost added
    """
    # Resolve event definition
    if event_name is not None:
        if variable is not None or threshold is not None or operator is not None:
            raise ValueError(
                "Cannot specify both event_name and (variable, threshold, operator). "
                "Use one or the other."
            )
        definition = get_event_definition(event_name)
        variable = definition["variable"]
        threshold = definition["threshold"]
        operator = definition["operator"]
    else:
        if variable is None or threshold is None or operator is None:
            raise ValueError("Must specify either event_name OR (variable, threshold, operator).")
        if operator not in VALID_OPERATORS:
            raise ValueError(
                f"Unknown operator '{operator}'. Valid operators: {', '.join(sorted(VALID_OPERATORS))}"
            )

    forecast_col = f"forecast_{variable}"
    observed_col = f"observed_{variable}"

    # Check columns exist
    missing = []
    if forecast_col not in df.columns:
        missing.append(f"forecast_{variable}")
    if observed_col not in df.columns:
        missing.append(f"observed_{variable}")
    if missing:
        raise ValueError(
            f"Missing columns in DataFrame: {', '.join(missing)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    # Generate event flags using vectorized threshold comparison
    fcst_event_col = forecast_event_col(event_name)
    obs_event_col = observed_event_col(event_name)

    df = df.copy()
    df[fcst_event_col] = apply_threshold_series(df[forecast_col], threshold, operator)
    df[obs_event_col] = apply_threshold_series(df[observed_col], threshold, operator)

    fcst_count = df[fcst_event_col].sum()
    obs_count = df[obs_event_col].sum()

    logger.info(
        "Event '%s' (%s %s %.2f): forecast=%d/%d, observed=%d/%d",
        event_name,
        variable,
        operator,
        threshold,
        fcst_count,
        len(df),
        obs_count,
        len(df),
    )

    return df


def generate_events_from_path(
    verification_path: str | Path,
    event_name: str | None = None,
    variable: str | None = None,
    threshold: float | None = None,
    operator: str | None = None,
) -> pd.DataFrame:
    """Generate event columns from a verification parquet file.

    Convenience function that reads the verification data, generates events,
    and returns the enriched DataFrame.

    Args:
        verification_path: Path to verification parquet file or glob pattern.
        event_name: Name of built-in event.
        variable: Variable name (if using custom event).
        threshold: Threshold value (if using custom event).
        operator: Comparison operator (if using custom event).

    Returns:
        DataFrame with event columns added.
    """
    verification_path = Path(verification_path)

    if verification_path.is_dir():
        parquet_files = list(verification_path.glob("*.parquet"))
    else:
        parquet_files = list(verification_path.parent.glob(verification_path.name))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at: {verification_path}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    return generate_events(
        df, event_name=event_name, variable=variable, threshold=threshold, operator=operator
    )
