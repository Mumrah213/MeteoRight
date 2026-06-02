"""Data loading helpers for forecast verification data.

Provides clean, typed DataFrames from parquet files produced by Phase 3
(weather_metrics) and Phase 2 (weather_verifier).

Loading order:
1. Metrics parquet files (aggregated metrics) — for plotting
2. Verification parquet files (raw verification) — for distribution/extreme analysis

Never mutates source data.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name references — imported from P3 schemas for consistency
# ---------------------------------------------------------------------------

# Metric columns
METRIC_NAME_COL = "metric_name"
METRIC_VALUE_COL = "metric_value"
VARIABLE_COL = "variable"
SAMPLE_SIZE_COL = "sample_size"
TOTAL_ROWS_COL = "total_rows"
MISSING_COUNT_COL = "missing_count"

# Dimension columns
LEAD_HOURS_COL = "lead_hours"
MODEL_COL = "model"
MONTH_COL = "month"
SEASON_COL = "season"

# Verification columns
ERROR_SUFFIX = "_error"
FORECAST_PREFIX = "forecast_"
OBSERVED_PREFIX = "observed_"


# ---------------------------------------------------------------------------
# Loading aggregated metrics
# ---------------------------------------------------------------------------


def load_metrics(
    paths: str | Path | list[str | Path],
) -> pd.DataFrame:
    """Load aggregated metrics parquet files.

    Reads metrics output from Phase 3 (weather_metrics), concatenates
    all matching files, and returns a single DataFrame.

    Expected schema (long format):
        [dimension_cols..., variable, metric_name, metric_value,
         sample_size, total_rows, missing_count]

    Args:
        paths: Single path, glob pattern, or list of paths to metrics parquet files.

    Returns:
        Concatenated DataFrame with all metric rows.

    Raises:
        FileNotFoundError: If no files match the glob pattern.
    """
    resolved = _resolve_paths(paths)

    if not resolved:
        raise FileNotFoundError(f"No metrics parquet files found matching: {paths}")

    logger.info("Loading %d metrics file(s)", len(resolved))

    frames: list[pd.DataFrame] = []
    for path in resolved:
        df = pd.read_parquet(path)
        logger.debug("  Loaded %d rows from %s", len(df), path)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    logger.info("Total: %d metric rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Loading verification data
# ---------------------------------------------------------------------------


def load_verification(
    paths: str | Path | list[str | Path],
) -> pd.DataFrame:
    """Load canonical verification parquet files.

    Reads verification data from Phase 2 (weather_verifier), concatenates
    all matching files, and returns a single DataFrame.

    Expected schema (wide format):
        [forecast_issue_time, forecast_target_time, observation_time,
         lead_hours, model, latitude, longitude, elevation,
         forecast_{var}, observed_{var}, {var}_error, ...]

    Args:
        paths: Single path, glob pattern, or list of paths to verification files.

    Returns:
        Concatenated DataFrame with all verification rows, sorted by provenance.
    """
    resolved = _resolve_paths(paths)

    if not resolved:
        raise FileNotFoundError(f"No verification parquet files found matching: {paths}")

    logger.info("Loading %d verification file(s)", len(resolved))

    frames: list[pd.DataFrame] = []
    for path in resolved:
        df = pd.read_parquet(path)
        logger.debug("  Loaded %d rows from %s", len(df), path)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    # Sort by provenance columns for consistency
    sort_cols = [
        c for c in ["forecast_issue_time", "forecast_target_time", "model"] if c in result.columns
    ]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)

    logger.info("Total: %d verification rows", len(result))
    return result


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------


def filter_metrics(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    metrics: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Filter a metrics DataFrame by variable, metric, and/or model.

    Args:
        df: Metrics DataFrame from load_metrics().
        variables: List of variable names to include (e.g. ["temperature_2m"]).
            If None, all variables are included.
        metrics: List of metric names to include (e.g. ["mae", "rmse"]).
            If None, all metrics are included.
        models: List of model names to include (e.g. ["icon_eu", "gfs"]).
            If None, all models are included.

    Returns:
        Filtered DataFrame.
    """
    result = df.copy()

    if variables and VARIABLE_COL in result.columns:
        result = result[result[VARIABLE_COL].isin(variables)]
    if metrics and METRIC_NAME_COL in result.columns:
        result = result[result[METRIC_NAME_COL].isin(metrics)]
    if models and MODEL_COL in result.columns:
        result = result[result[MODEL_COL].isin(models)]

    logger.info(
        "Filtered metrics: %d → %d rows",
        len(df),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Pivot helpers
# ---------------------------------------------------------------------------


def pivot_metrics(
    df: pd.DataFrame,
    index_cols: list[str],
    columns_cols: list[str],
    values_col: str = METRIC_VALUE_COL,
) -> pd.DataFrame:
    """Pivot a long-format metrics DataFrame to wide format.

    Useful for plotting where each column should be a separate series.

    Args:
        df: Long-format metrics DataFrame.
        index_cols: Columns to use as index (e.g. ["lead_hours"]).
        columns_cols: Columns to pivot into columns (e.g. ["variable", "metric_name"]).
        values_col: Column containing values (default: metric_value).

    Returns:
        Wide-format DataFrame suitable for plotting.
    """
    valid_index = [c for c in index_cols if c in df.columns]
    valid_columns = [c for c in columns_cols if c in df.columns]

    if not valid_index or not valid_columns:
        logger.warning(
            "pivot_metrics: no valid index (%s) or columns (%s) in DataFrame",
            valid_index,
            valid_columns,
        )
        return df

    try:
        pivot = df.pivot_table(
            index=valid_index,
            columns=valid_columns,
            values=values_col,
            aggfunc="first",
        )
        # Flatten multi-level column index
        if isinstance(pivot.columns, pd.MultiIndex):
            pivot.columns = ["_".join(str(c) for c in col) for col in pivot.columns]
        return pivot
    except Exception as e:
        logger.warning("pivot_metrics failed: %s — returning original", e)
        return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_paths(paths: str | Path | list[str | Path]) -> list[Path]:
    """Resolve paths, supporting glob patterns.

    Args:
        paths: Single path, glob pattern, or list of paths.

    Returns:
        List of resolved Path objects, sorted.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]

    result: list[Path] = []
    for p in paths:
        p_str = str(p)
        if any(c in p_str for c in "*?["):
            matched = sorted(glob.glob(p_str))
            if not matched:
                logger.warning("Glob pattern matched no files: %s", p_str)
            else:
                result.extend(Path(m) for m in matched)
        else:
            result.append(Path(p))

    return result
