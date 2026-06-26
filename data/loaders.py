"""Loaders for verification and metrics Parquet artifacts.

Used by ``meteoright analyze`` / ``meteoright demo`` to read the outputs of the
metrics and verification steps back into DataFrames. Verification reading reuses
:func:`metrics.io.read_verification` (single source of truth for path resolution
and provenance sorting); metrics reading shares the same path expansion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from metrics.io import _resolve_paths, read_verification


def load_verification(paths: str | Path | list[str | Path]) -> pd.DataFrame:
    """Load canonical verification rows from a file, directory, or glob.

    Thin wrapper over :func:`metrics.io.read_verification` so the analysis CLI
    has a stable import surface independent of where the reader lives.
    """
    return read_verification(paths)


def load_metrics(paths: str | Path | list[str | Path]) -> pd.DataFrame:
    """Load aggregated metric rows from a file, directory, or glob.

    Metrics files are long-format (one row per dimension/variable/metric), so
    they are simply concatenated — no provenance sort applies.
    """
    resolved = _resolve_paths(paths)
    if not resolved:
        raise FileNotFoundError(f"No metrics parquet files found for: {paths!r}")

    frames = [pd.read_parquet(p) for p in resolved]
    return pd.concat(frames, ignore_index=True)


def filter_metrics(
    df: pd.DataFrame,
    variables: list[str] | None = None,
    metrics: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Filter a long-format metrics DataFrame by variable, metric, and model.

    Args:
        df: Long-format metrics frame (columns include ``variable`` and
            ``metric_name``, optionally ``model``).
        variables: Keep only these variables. ``None`` keeps all.
        metrics: Keep only these metric names (e.g. ``["mae", "rmse"]``).
            ``None`` keeps all.
        models: Keep only these models. ``None`` keeps all.

    Returns:
        The filtered frame (a copy). Missing filter columns are tolerated so a
        partially-populated frame does not raise.
    """
    out = df
    if variables is not None and "variable" in out.columns:
        out = out[out["variable"].isin(variables)]
    if metrics is not None and "metric_name" in out.columns:
        out = out[out["metric_name"].isin(metrics)]
    if models is not None and "model" in out.columns:
        out = out[out["model"].isin(models)]
    return out.reset_index(drop=True)


def pivot_metrics(
    df: pd.DataFrame,
    index_cols: list[str],
    columns_cols: list[str],
    value_col: str = "metric_value",
) -> pd.DataFrame:
    """Pivot a long-format metrics frame into a wide table.

    Args:
        df: Long-format metrics frame.
        index_cols: Columns to keep as the row index.
        columns_cols: Columns whose values become new columns.
        value_col: Column supplying the pivoted values.

    Returns:
        A wide DataFrame (index reset to columns). An empty input, or
        index/column names absent from the frame, return the input unchanged
        rather than raising — callers treat pivoting as best-effort reshaping.
    """
    if df.empty:
        return df.copy()
    if not set(index_cols).issubset(df.columns) or not set(columns_cols).issubset(df.columns):
        return df.copy()
    if value_col not in df.columns:
        return df.copy()

    pivoted = df.pivot_table(
        index=index_cols,
        columns=columns_cols,
        values=value_col,
        aggfunc="first",
    )
    return pivoted.reset_index()
