"""Extreme error analysis.

Identifies and analyzes forecast failure modes:
- Top N largest errors (catastrophic misses)
- Error quantiles by lead time
- Extreme event frequency
- Lead-time breakdown of extremes

Usage
-----
>>> from weather_analysis.analysis.extremes import (
...     find_top_errors,
...     compute_error_quantiles,
... )
>>> top_errors = find_top_errors(df, "temperature_2m", n=50)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Top errors
# ---------------------------------------------------------------------------


def find_top_errors(
    df: pd.DataFrame,
    variable: str,
    n: int = 50,
    model: str | None = None,
    lead_hours: int | None = None,
) -> pd.DataFrame:
    """Find the N largest absolute errors.

    Identifies forecast failures — cases where the forecast was
    significantly wrong. Useful for:
    - Understanding catastrophic miss patterns
    - Identifying systematic failure modes
    - Case study selection

    Args:
        df: Verification DataFrame.
        variable: Variable name.
        n: Number of top errors to return.
        model: Optional model filter.
        lead_hours: Optional lead time filter.

    Returns:
        DataFrame with the N largest absolute errors, sorted descending.
    """
    error_col = f"{variable}_error"

    if error_col not in df.columns:
        logger.warning("No error column '%s' in data", error_col)
        return pd.DataFrame()

    cols_to_keep = [error_col, "lead_hours"]
    if "model" in df.columns:
        cols_to_keep.append("model")

    data = df[cols_to_keep].dropna().copy()

    if model:
        data = data[data["model"] == model]
    if lead_hours is not None:
        data = data[data["lead_hours"] == lead_hours]

    if data.empty:
        return pd.DataFrame()

    data["abs_error"] = data[error_col].abs()
    top = data.nlargest(n, "abs_error")

    logger.info(
        "Found top %d absolute errors for %s (mean abs error: %.2f)",
        n,
        variable,
        top["abs_error"].mean(),
    )
    return top


def find_extreme_misses(
    df: pd.DataFrame,
    variable: str,
    threshold: float,
    direction: str = "both",
    model: str | None = None,
) -> pd.DataFrame:
    """Find extreme misses above a threshold.

    An "extreme miss" is a forecast where the absolute error exceeds
    a specified threshold. Useful for identifying catastrophic failures.

    Args:
        df: Verification DataFrame.
        variable: Variable name.
        threshold: Absolute error threshold.
        direction: "over" (overprediction), "under" (underprediction), "both".
        model: Optional model filter.

    Returns:
        DataFrame with extreme misses.
    """
    error_col = f"{variable}_error"

    if error_col not in df.columns:
        return pd.DataFrame()

    cols_to_keep = [error_col, "lead_hours"]
    if "model" in df.columns:
        cols_to_keep.append("model")

    data = df[cols_to_keep].dropna().copy()

    if model:
        data = data[data["model"] == model]

    # Filter by threshold
    if direction == "over":
        data = data[data[error_col] >= threshold]
    elif direction == "under":
        data = data[data[error_col] <= -threshold]
    else:  # both
        data = data[data[error_col].abs() >= threshold]

    data["abs_error"] = data[error_col].abs()
    data = data.sort_values("abs_error", ascending=False)

    logger.info(
        "Found %d extreme misses (|error| >= %.2f) for %s",
        len(data),
        threshold,
        variable,
    )
    return data


# ---------------------------------------------------------------------------
# Error quantiles
# ---------------------------------------------------------------------------


def compute_error_quantiles(
    df: pd.DataFrame | pd.Series,
    variable: str,
    percentiles: list[int] | None = None,
    lead_hours: int | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Compute error quantiles for a variable.

    Provides a detailed view of the error distribution:
    - P10, P25, P50 (median), P75, P90
    - P5, P95 for tail analysis

    Args:
        df: Verification DataFrame or Series of errors.
        variable: Variable name.
        percentiles: Percentiles to compute. Defaults to [5, 10, 25, 50, 75, 90, 95].
        lead_hours: Optional lead time filter.
        model: Optional model filter.

    Returns:
        DataFrame with percentile values.
    """
    if percentiles is None:
        percentiles = [5, 10, 25, 50, 75, 90, 95]

    error_col = f"{variable}_error"

    # Handle both DataFrame and Series input
    if isinstance(df, pd.Series):
        data = df.dropna()
    elif error_col not in df.columns:
        return pd.DataFrame()
    else:
        data = df[error_col].dropna()

    if model:
        data = df[(df["model"] == model)][error_col].dropna()
    if lead_hours is not None:
        if model:
            data = df[(df["model"] == model) & (df["lead_hours"] == lead_hours)][error_col].dropna()
        else:
            data = df[df["lead_hours"] == lead_hours][error_col].dropna()

    if data.empty:
        return pd.DataFrame()

    results = []
    for p in percentiles:
        val = np.percentile(data, p)
        results.append(
            {
                "percentile": p,
                "value": val,
            }
        )

    result_df = pd.DataFrame(results)

    logger.info(
        "Computed error quantiles for %s: %s",
        variable,
        result_df.to_dict("records"),
    )
    return result_df


def compute_lead_time_quantiles(
    df: pd.DataFrame,
    variable: str,
    percentiles: list[int] | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Compute error quantiles grouped by lead time.

    Shows how the error distribution changes with lead time:
    - Does the median increase?
    - Do the tails widen?
    - Does skewness change?

    Args:
        df: Verification DataFrame.
        variable: Variable name.
        percentiles: Percentiles to compute.
        model: Optional model filter.

    Returns:
        DataFrame with lead_hours, percentile, value columns.
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    error_col = f"{variable}_error"

    if error_col not in df.columns:
        return pd.DataFrame()

    data = df[[error_col, "lead_hours"]].dropna()

    if model:
        data = df[(df["model"] == model)][[error_col, "lead_hours"]].dropna()

    if data.empty:
        return pd.DataFrame()

    rows = []
    for lt in sorted(data["lead_hours"].unique()):
        lt_data = data[data["lead_hours"] == lt][error_col]
        if lt_data.empty:
            continue
        for p in percentiles:
            val = np.percentile(lt_data, p)
            rows.append(
                {
                    "lead_hours": lt,
                    "percentile": p,
                    "value": val,
                }
            )

    result_df = pd.DataFrame(rows)
    logger.info(
        "Computed lead-time quantiles for %s: %d combinations",
        variable,
        len(result_df),
    )
    return result_df


# ---------------------------------------------------------------------------
# Extreme event frequency
# ---------------------------------------------------------------------------


def compute_extreme_event_frequency(
    df: pd.DataFrame | pd.Series,
    variable: str,
    threshold: float,
    model: str | None = None,
) -> dict:
    """Compute frequency of extreme error events.

    Args:
        df: Verification DataFrame or Series of errors.
        variable: Variable name.
        threshold: Absolute error threshold.
        model: Optional model filter.

    Returns:
        Dictionary with:
            - total: total observations
            - extreme_count: number exceeding threshold
            - frequency: fraction of extremes
            - mean_error_when_extreme: average error magnitude during extremes
    """
    error_col = f"{variable}_error"

    if isinstance(df, pd.Series):
        data = df.dropna()
    elif error_col not in df.columns:
        return {
            "total": 0,
            "extreme_count": 0,
            "frequency": 0.0,
            "mean_error_when_extreme": float("nan"),
        }
    else:
        data = df[error_col].dropna()

    if model:
        data = df[(df["model"] == model)][error_col].dropna()

    if data.empty:
        return {
            "total": 0,
            "extreme_count": 0,
            "frequency": 0.0,
            "mean_error_when_extreme": float("nan"),
        }

    total = len(data)
    extremes = data.abs() >= threshold
    extreme_count = extremes.sum()
    frequency = extreme_count / total if total > 0 else 0.0
    mean_extreme_error = data[extremes].abs().mean() if extreme_count > 0 else float("nan")

    result = {
        "total": total,
        "extreme_count": int(extreme_count),
        "frequency": float(frequency),
        "mean_error_when_extreme": float(mean_extreme_error)
        if not np.isnan(mean_extreme_error)
        else None,
    }

    logger.info(
        "Extreme event frequency for %s (|error| >= %.2f): %.1f%%",
        variable,
        threshold,
        100 * frequency,
    )
    return result
