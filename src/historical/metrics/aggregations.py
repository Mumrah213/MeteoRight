"""Aggregation engine for forecast verification metrics.

Computes grouped metrics from a canonical verification DataFrame.

Supports:
- Multi-column groupby (e.g. ["lead_hours", "model"])
- Multiple metrics (e.g. ["mae", "rmse", "bias"])
- Multiple variables (e.g. ["temperature_2m", "precipitation"])
- Automatic sample_size, total_rows, missing_count per bucket

Output format: LONG format (one row per metric per group).

Usage
-----
>>> from weather_metrics.aggregations import aggregate_metrics
>>> df = read_verification("data/verification/*.parquet")
>>> result = aggregate_metrics(
...     df,
...     groupby=["lead_hours"],
...     metrics=["mae", "rmse", "bias"],
...     variables=["temperature_2m"],
... )
>>> print(result)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from .metrics import get_metric_function
from .schemas import (
    METRIC_NAME_COL,
    METRIC_VALUE_COL,
    MISSING_COUNT_COL,
    SAMPLE_SIZE_COL,
    TOTAL_ROWS_COL,
    VARIABLE_COL,
    long_format_columns,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error column helpers — uses same naming convention as P2 schema
# ---------------------------------------------------------------------------


def _error_col(variable: str) -> str:
    """Return the error column name for a variable.

    Args:
        variable: Variable name (e.g. "temperature_2m").

    Returns:
        Error column name (e.g. "temperature_2m_error").
    """
    return f"{variable}_error"


def _forecast_col(variable: str) -> str:
    """Return the forecast column name for a variable.

    Args:
        variable: Variable name (e.g. "temperature_2m").

    Returns:
        Forecast column name (e.g. "forecast_temperature_2m").
    """
    return f"forecast_{variable}"


def _observed_col(variable: str) -> str:
    """Return the observed column name for a variable.

    Args:
        variable: Variable name (e.g. "temperature_2m").

    Returns:
        Observed column name (e.g. "observed_temperature_2m").
    """
    return f"observed_{variable}"


# ---------------------------------------------------------------------------
# Season helper
# ---------------------------------------------------------------------------


def _add_season_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'season' column based on the month of forecast_target_time.

    Uses meteorological seasons:
        DJF: December, January, February
        MAM: March, April, May
        JJA: June, July, August
        SON: September, October, November

    Args:
        df: DataFrame with a 'forecast_target_time' column (datetime).

    Returns:
        DataFrame with an added 'season' column.
    """
    from .schemas import SEASON_MAP

    df = df.copy()
    # Extract month from forecast_target_time
    months = df["forecast_target_time"].dt.month
    df["season"] = months.map(SEASON_MAP).fillna("UNKNOWN")
    return df


# ---------------------------------------------------------------------------
# Core aggregation function
# ---------------------------------------------------------------------------


def aggregate_metrics(
    df: pd.DataFrame,
    groupby: list[str],
    metrics: list[str],
    variables: list[str],
) -> pd.DataFrame:
    """Compute grouped verification metrics.

    This is the core aggregation engine. It:

    1. Validates inputs (column existence, metric names, groupby dimensions)
    2. Adds derived columns (season) if needed
    3. For each variable, for each metric:
       a. Groups by the specified dimensions
       b. Computes the metric on error columns
       c. Records sample_size, total_rows, missing_count
    4. Returns a long-format DataFrame

    Args:
        df: Canonical verification DataFrame from read_verification().
            Must contain error columns for all specified variables.
        groupby: List of dimension columns to group by.
            Valid values: "lead_hours", "model", "month", "season".
            Can be multi-column (e.g. ["model", "lead_hours"]).
        metrics: List of metric names to compute.
            Valid values: "mae", "rmse", "bias", "std".
        variables: List of variable names to compute metrics for.
            Must match variables in the verification DataFrame.

    Returns:
        Long-format DataFrame with columns:
            [dimension_cols..., variable, metric_name, metric_value,
             sample_size, total_rows, missing_count]

    Raises:
        ValueError: If any input is invalid (unknown metric, missing column, etc.)

    Example:
        >>> df = read_verification("data/verification/*.parquet")
        >>> result = aggregate_metrics(
        ...     df,
        ...     groupby=["lead_hours"],
        ...     metrics=["mae", "rmse", "bias"],
        ...     variables=["temperature_2m"],
        ... )
        >>> print(result.columns.tolist())
        ['lead_hours', 'variable', 'metric_name', 'metric_value',
         'sample_size', 'total_rows', 'missing_count']
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    _validate_inputs(df, groupby, metrics, variables)

    # ------------------------------------------------------------------
    # Add derived columns if needed
    # ------------------------------------------------------------------
    df = df.copy()
    if "season" in groupby and "season" not in df.columns:
        logger.info("Adding 'season' column for groupby")
        df = _add_season_column(df)

    # ------------------------------------------------------------------
    # Build the output
    # ------------------------------------------------------------------
    output_rows: list[dict] = []

    for variable in variables:
        error_col = _error_col(variable)

        for metric_name in metrics:
            metric_func = get_metric_function(metric_name)

            # Group by the specified dimensions
            grouped = df.groupby(groupby, sort=True)

            for group_key, group_df in grouped:
                # Handle single-group vs multi-group keys
                if isinstance(group_key, tuple):
                    key_dict = dict(zip(groupby, group_key, strict=False))
                else:
                    key_dict = {groupby[0]: group_key}

                errors = group_df[error_col]
                total = len(errors)
                clean = errors.dropna()
                n = len(clean)
                missing = total - n

                # Compute the metric
                value = metric_func(clean) if n > 0 else float("nan")

                row = {
                    **key_dict,
                    VARIABLE_COL: variable,
                    METRIC_NAME_COL: metric_name,
                    METRIC_VALUE_COL: value,
                    SAMPLE_SIZE_COL: n,
                    TOTAL_ROWS_COL: total,
                    MISSING_COUNT_COL: missing,
                }
                output_rows.append(row)

            logger.debug(
                "Computed '%s' for '%s': %d groups, %d rows",
                metric_name,
                variable,
                len(grouped),
                len(output_rows),
            )

    # ------------------------------------------------------------------
    # Build output DataFrame
    # ------------------------------------------------------------------
    if not output_rows:
        logger.warning("aggregate_metrics: no output rows produced")
        cols = long_format_columns(groupby)
        return pd.DataFrame(columns=cols)

    result = pd.DataFrame(output_rows)

    # Ensure column order
    cols = long_format_columns(groupby)
    # Add any missing columns (e.g. if a dimension wasn't present in data)
    for col in cols:
        if col not in result.columns:
            result[col] = pd.NA
    result = result[cols]

    logger.info(
        "aggregate_metrics: produced %d rows (%d variables × %d metrics × %d groups)",
        len(result),
        len(variables),
        len(metrics),
        len(grouped),
    )

    return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_METRICS = {"mae", "rmse", "bias", "std"}
_VALID_DIMENSIONS = {"lead_hours", "model", "month", "season"}


def _validate_inputs(
    df: pd.DataFrame,
    groupby: list[str],
    metrics: list[str],
    variables: list[str],
) -> None:
    """Validate aggregation inputs.

    Args:
        df: Verification DataFrame.
        groupby: Groupby dimensions.
        metrics: Metric names.
        variables: Variable names.

    Raises:
        ValueError: If any input is invalid.
    """
    # Check groupby dimensions
    for dim in groupby:
        if dim not in _VALID_DIMENSIONS:
            raise ValueError(
                f"Invalid groupby dimension '{dim}'. "
                f"Valid dimensions: {', '.join(sorted(_VALID_DIMENSIONS))}"
            )

    # Check metrics
    for m in metrics:
        if m not in _VALID_METRICS:
            raise ValueError(
                f"Unknown metric '{m}'. Valid metrics: {', '.join(sorted(_VALID_METRICS))}"
            )

    # Check variables — each must have an error column
    for var in variables:
        error_col = f"{var}_error"
        if error_col not in df.columns:
            raise ValueError(
                f"Missing error column '{error_col}' in verification DataFrame. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )

    # Check groupby columns exist
    for dim in groupby:
        if dim not in df.columns:
            raise ValueError(
                f"Groupby dimension '{dim}' not found in DataFrame. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )

    # Check for duplicates in groupby
    if len(groupby) != len(set(groupby)):
        raise ValueError(f"Duplicate groupby dimensions: {groupby}")

    # Check for duplicates in metrics
    if len(metrics) != len(set(metrics)):
        raise ValueError(f"Duplicate metrics: {metrics}")

    # Check for duplicates in variables
    if len(variables) != len(set(variables)):
        raise ValueError(f"Duplicate variables: {variables}")
