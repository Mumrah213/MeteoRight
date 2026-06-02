"""Confusion matrix computation for event-based forecast verification.

Computes the four fundamental counts of binary classification from event flags:
    hits, misses, false_alarms, correct_negatives.

Confusion Matrix Definitions
----------------------------

    |                       | Event Observed (True)  | Event Observed (False) |
    |-----------------------|------------------------|------------------------|
    | Event Forecast (True) | Hit                    | False Alarm            |
    | Event Forecast (False)| Miss                   | Correct Negative       |

    Hit         = forecast=event AND observed=event   → Correctly predicted occurrence
    Miss        = forecast=no-event AND observed=event → Missed an occurring event
    False Alarm = forecast=event AND observed=no-event → Predicted event that did not occur
    Correct Neg = forecast=no-event AND observed=no-event → Correctly predicted non-event

NaN Handling
------------
Rows where forecast OR observed values are NaN are excluded from all counts.
These rows contribute to an `excluded_nan` counter but do not affect hits,
misses, false_alarms, or correct_negatives.

The invariant is preserved:
    hits + misses + false_alarms + correct_negatives + excluded_nan == total_rows

Grouping Semantics
------------------
Like P3 metrics, confusion counts are computed per group defined by dimension
columns (lead_hours, model, month, season). Each group produces one row of
confusion statistics.

Usage
-----
>>> from advanced_verification.confusion import compute_confusion_matrix
>>> df = read_verification("data/verification/verification.parquet")
>>> df = generate_events(df, event_name="frost")
>>> result = compute_confusion_matrix(df, event_name="frost", groupby=["lead_hours"])
>>> print(result)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from .schema import (
    CORRECT_NEGATIVES_COL,
    EVENT_NAME_COL,
    EXCLUDED_NAN_COL,
    FALSE_ALARMS_COL,
    HITS_COL,
    LONG_FORMAT_DIMENSION_COLS,
    MISSES_COL,
    OPERATOR_COL,
    SAMPLE_SIZE_COL,
    THRESHOLD_COL,
    TOTAL_ROWS_COL,
    VARIABLE_COL,
    forecast_event_col,
    observed_event_col,
)

logger = logging.getLogger(__name__)

# Type alias for valid operators
ValidOperator = Literal["le", "lt", "ge", "gt", "eq", "ne"]


def compute_confusion_counts(
    forecast_event: pd.Series,
    observed_event: pd.Series,
) -> dict[str, int]:
    """Compute confusion matrix counts from binary event series.

    Args:
        forecast_event: Boolean Series — did the forecast predict the event?
        observed_event: Boolean Series — did the event actually occur?

    Returns:
        Dict with keys: hits, misses, false_alarms, correct_negatives, excluded_nan.

    Examples:
        >>> fc = pd.Series([True, True, False, False])
        >>> obs = pd.Series([True, False, True, False])
        >>> compute_confusion_counts(fc, obs)
        {'hits': 1, 'misses': 1, 'false_alarms': 1, 'correct_negatives': 1, 'excluded_nan': 0}
    """
    # Count each outcome
    hits = int((forecast_event & observed_event).sum())
    misses = int((~forecast_event & observed_event).sum())
    false_alarms = int((forecast_event & ~observed_event).sum())
    correct_negatives = int((~forecast_event & ~observed_event).sum())

    total = len(forecast_event)
    excluded_nan = total - hits - misses - false_alarms - correct_negatives

    return {
        HITS_COL: hits,
        MISSES_COL: misses,
        FALSE_ALARMS_COL: false_alarms,
        CORRECT_NEGATIVES_COL: correct_negatives,
        EXCLUDED_NAN_COL: excluded_nan,
    }


def _count_non_nan_pairs(forecast_col: pd.Series, observed_col: pd.Series) -> int:
    """Count rows where BOTH forecast and observed values are non-NaN."""
    return int((forecast_col.notna() & observed_col.notna()).sum())


def compute_confusion_matrix(
    df: pd.DataFrame,
    event_name: str,
    groupby: list[str] | None = None,
) -> pd.DataFrame:
    """Compute confusion matrix counts from a DataFrame with event columns.

    This is the core function that converts event flags into operational
    forecast verification counts.

    Args:
        df: DataFrame with event columns (event_forecast_{name}, event_observed_{name}).
            Must also have the original forecast and observed variable columns
            for NaN detection.
        event_name: Name of the event to analyze.
        groupby: List of dimension columns to group by.
            Valid values: "lead_hours", "model", "month", "season".
            If None, all rows are treated as a single group.

    Returns:
        DataFrame with one row per group containing:
            event_name, variable, threshold, operator,
            [groupby_dims...],
            hits, misses, false_alarms, correct_negatives, excluded_nan,
            sample_size, total_rows

    Raises:
        ValueError: If event columns or required variables are missing.

    Example:
        >>> df = read_verification("data/verification/verification.parquet")
        >>> from advanced_verification.events import generate_events
        >>> df = generate_events(df, event_name="frost")
        >>> result = compute_confusion_matrix(df, event_name="frost", groupby=["lead_hours"])
        >>> print(result.columns)
    """
    from .schema import get_event_definition

    if groupby is None:
        groupby = []

    # Validate groupby dimensions
    for dim in groupby:
        if dim not in LONG_FORMAT_DIMENSION_COLS:
            raise ValueError(
                f"Invalid groupby dimension '{dim}'. "
                f"Valid dimensions: {', '.join(sorted(LONG_FORMAT_DIMENSION_COLS))}"
            )

    # Get event definition to find the variable, threshold, and operator
    definition = get_event_definition(event_name)
    variable = definition["variable"]
    threshold = definition["threshold"]
    operator = definition["operator"]

    fcst_event_col = forecast_event_col(event_name)
    obs_event_col = observed_event_col(event_name)
    forecast_var_col = f"forecast_{variable}"
    observed_var_col = f"observed_{variable}"

    # Check columns exist
    required = [fcst_event_col, obs_event_col, forecast_var_col, observed_var_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for event '{event_name}': {', '.join(missing)}. "
            f"Run generate_events() first."
        )

    # Build output rows
    output_rows: list[dict] = []

    if groupby:
        grouped = df.groupby(groupby, sort=True)
    else:
        # Single group: all data
        grouped = [("all", df)]

    for group_key, group_df in grouped:
        # Skip empty groups
        if len(group_df) == 0:
            continue

        if isinstance(group_key, tuple):
            key_dict = dict(zip(groupby, group_key, strict=False))
        else:
            key_dict = {groupby[0]: group_key} if groupby else {}

        fcst_evt = group_df[fcst_event_col]
        obs_evt = group_df[obs_event_col]

        # Determine valid rows (both forecast and observed values are non-NaN)
        valid_mask = group_df[forecast_var_col].notna() & group_df[observed_var_col].notna()
        n_valid = valid_mask.sum()
        n_nan_excluded = len(group_df) - n_valid

        valid_fcst = fcst_evt[valid_mask]
        valid_obs = obs_evt[valid_mask]

        # Compute confusion counts on valid rows
        if n_valid > 0:
            counts = compute_confusion_counts(valid_fcst, valid_obs)
        else:
            counts = {
                HITS_COL: 0,
                MISSES_COL: 0,
                FALSE_ALARMS_COL: 0,
                CORRECT_NEGATIVES_COL: 0,
                EXCLUDED_NAN_COL: 0,
            }

        total_rows = len(group_df)

        row = {
            EVENT_NAME_COL: event_name,
            VARIABLE_COL: variable,
            THRESHOLD_COL: threshold,
            OPERATOR_COL: operator,
            **key_dict,
            HITS_COL: counts[HITS_COL],
            MISSES_COL: counts[MISSES_COL],
            FALSE_ALARMS_COL: counts[FALSE_ALARMS_COL],
            CORRECT_NEGATIVES_COL: counts[CORRECT_NEGATIVES_COL],
            EXCLUDED_NAN_COL: n_nan_excluded,
            SAMPLE_SIZE_COL: counts[HITS_COL]
            + counts[MISSES_COL]
            + counts[FALSE_ALARMS_COL]
            + counts[CORRECT_NEGATIVES_COL],
            TOTAL_ROWS_COL: total_rows,
        }
        output_rows.append(row)

    if not output_rows:
        cols = schema_confusion_columns(groupby)
        return pd.DataFrame(columns=cols)

    result = pd.DataFrame(output_rows)

    # Ensure column order
    cols = schema_confusion_columns(groupby)
    for col in cols:
        if col not in result.columns:
            result[col] = pd.NA
    result = result[cols]

    logger.info(
        "Confusion matrix for '%s': %d groups, %d total rows",
        event_name,
        len(output_rows),
        sum(r[TOTAL_ROWS_COL] for r in output_rows),
    )

    return result


# Re-export schema function with correct import path
from .schema import (
    CONFUSION_COLUMNS as _CONFUSION_COLUMNS,
)
from .schema import (
    LONG_FORMAT_DIMENSION_COLS as _LONG_FORMAT_DIMENSION_COLS,
)


def schema_confusion_columns(dimensions: list[str] | None = None) -> list[str]:
    """Return the full column order for a confusion statistics output DataFrame.

    Args:
        dimensions: List of dimension columns to include
            (e.g. ["lead_hours", "model"]). If None, no dimension columns
            are included.

    Returns:
        Ordered list of column names.
    """
    if dimensions is None:
        dimensions = []
    valid_dims = [d for d in dimensions if d in _LONG_FORMAT_DIMENSION_COLS]
    return valid_dims + list(_CONFUSION_COLUMNS)


def compute_confusion_from_path(
    verification_path: str | Path,
    event_name: str,
    groupby: list[str] | None = None,
) -> pd.DataFrame:
    """Compute confusion matrix from a verification parquet file.

    Convenience function that reads verification data, generates events,
    computes confusion counts, and returns the result.

    Args:
        verification_path: Path to verification parquet file or glob pattern.
        event_name: Name of built-in event.
        groupby: List of dimension columns to group by.

    Returns:
        DataFrame with confusion matrix counts.
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

    from .events import generate_events

    df = generate_events(df, event_name=event_name)

    return compute_confusion_matrix(df, event_name=event_name, groupby=groupby)
