"""Core verification metrics for weather analysis.

Computes MAE, RMSE, bias and grouping helpers for interactive exploration.

All metrics operate on aligned DataFrames from src.analysis.alignment.align().

Usage
-----
>>> from src.analysis.metrics import mae_by_leadtime, rmse_by_model, bias_by_variable
>>> aligned = align(forecast, truth)
>>> mae_lt = mae_by_leadtime(aligned)
>>> rmse_mod = rmse_by_model(aligned)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Grouping helpers ────────────────────────────────────────────────────────


def mae_by_leadtime(df: pd.DataFrame) -> pd.DataFrame:
    """MAE grouped by lead_hours.

    Returns DataFrame with columns: lead_hours, mae, sample_size
    """
    result = mae(df, group_by="lead_hours")
    if isinstance(result, pd.DataFrame):
        return result.rename(columns={"group_key": "lead_hours"})
    return pd.DataFrame({"lead_hours": [0], "mae": [result], "sample_size": [len(df)]})


def rmse_by_leadtime(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE grouped by lead_hours.

    Returns DataFrame with columns: lead_hours, rmse, sample_size
    """
    result = rmse(df, group_by="lead_hours")
    if isinstance(result, pd.DataFrame):
        return result.rename(columns={"group_key": "lead_hours"})
    return pd.DataFrame({"lead_hours": [0], "rmse": [result], "sample_size": [len(df)]})


def bias_by_leadtime(df: pd.DataFrame) -> pd.DataFrame:
    """Bias grouped by lead_hours.

    Returns DataFrame with columns: lead_hours, bias, sample_size
    """
    result = bias(df, group_by="lead_hours")
    if isinstance(result, pd.DataFrame):
        return result.rename(columns={"group_key": "lead_hours"})
    return pd.DataFrame({"lead_hours": [0], "bias": [result], "sample_size": [len(df)]})


def mae_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """MAE grouped by model.

    Returns DataFrame with columns: model, mae, sample_size
    """
    result = mae(df, group_by="model")
    if isinstance(result, pd.DataFrame):
        return result.rename(columns={"group_key": "model"})
    return pd.DataFrame({"model": ["all"], "mae": [result], "sample_size": [len(df)]})


def rmse_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE grouped by model.

    Returns DataFrame with columns: model, rmse, sample_size
    """
    result = rmse(df, group_by="model")
    if isinstance(result, pd.DataFrame):
        return result.rename(columns={"group_key": "model"})
    return pd.DataFrame({"model": ["all"], "rmse": [result], "sample_size": [len(df)]})


def bias_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Bias grouped by model.

    Returns DataFrame with columns: model, bias, sample_size
    """
    result = bias(df, group_by="model")
    if isinstance(result, pd.DataFrame):
        return result.rename(columns={"group_key": "model"})
    return pd.DataFrame({"model": ["all"], "bias": [result], "sample_size": [len(df)]})


def mae_by_variable(df: pd.DataFrame) -> pd.DataFrame:
    """MAE grouped by variable.

    Returns DataFrame with columns: variable, mae, sample_size
    """
    result = mae(df, group_by="variable")
    if isinstance(result, pd.DataFrame):
        return result.rename(columns={"group_key": "variable"})
    return pd.DataFrame({"variable": ["all"], "mae": [result], "sample_size": [len(df)]})


def rmse_by_variable(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE grouped by variable.

    Returns DataFrame with columns: variable, rmse, sample_size
    """
    result = rmse(df, group_by="variable")
    if isinstance(result, pd.DataFrame):
        return result.rename(columns={"group_key": "variable"})
    return pd.DataFrame({"variable": ["all"], "rmse": [result], "sample_size": [len(df)]})


def bias_by_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Bias grouped by variable.

    Returns DataFrame with columns: variable, bias, sample_size
    """
    result = bias(df, group_by="variable")
    if isinstance(result, pd.DataFrame):
        return result.rename(columns={"group_key": "variable"})
    return pd.DataFrame({"variable": ["all"], "bias": [result], "sample_size": [len(df)]})


# ── Convenience: compute all metrics at once ────────────────────────────────


def compute_all_metrics(
    df: pd.DataFrame,
    group_by: str | None = None,
) -> dict[str, float | pd.DataFrame]:
    """Compute MAE, RMSE, and bias in one call.

    Parameters
    ----------
    df : DataFrame
        Aligned DataFrame.
    group_by : str or None
        Column to group by.

    Returns
    -------
    Dict with keys: "mae", "rmse", "bias"
    Values are either floats (if group_by is None) or DataFrames.
    """
    return {
        "mae": mae(df, group_by),
        "rmse": rmse(df, group_by),
        "bias": bias(df, group_by),
    }


# ── Internal helpers ────────────────────────────────────────────────────────


def _get_matched(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to matched records with non-null values."""
    if "match_status" in df.columns:
        df = df[df["match_status"] == "matched"]
    return df.dropna(subset=["forecast_value", "observation_value"])


def _empty_result(group_by: str | None) -> pd.DataFrame:
    """Return an empty DataFrame with appropriate columns."""
    if group_by:
        return pd.DataFrame({group_by: [], "metric": [], "value": []})
    return pd.DataFrame({"metric": ["no_data"], "value": [np.nan]})


# ── Grouped metric implementations ─────────────────────────────────────────


def _mae_grouped(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """MAE grouped implementation."""
    matched = _get_matched(df)

    if "abs_error" in matched.columns:
        errors = matched["abs_error"]
    elif "forecast_value" in matched.columns and "observation_value" in matched.columns:
        errors = (matched["forecast_value"] - matched["observation_value"]).abs()
    else:
        return pd.DataFrame({group_by: [], "mae": [], "sample_size": []})

    errors = errors.dropna()
    matched_idx = errors.index
    if len(matched_idx) == 0:
        return pd.DataFrame({group_by: [], "mae": [], "sample_size": []})

    grouped = df.loc[matched_idx].groupby(group_by)

    results = []
    for group_key, group_df in grouped:
        mask = group_df.index.isin(matched_idx)
        group_errors = errors.loc[group_df.index[mask]].dropna()

        if len(group_errors) == 0:
            continue

        results.append(
            {
                "group_key": group_key,
                "mae": float(group_errors.mean()),
                "sample_size": len(group_errors),
            }
        )

    return pd.DataFrame(results)


def _rmse_grouped(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """RMSE grouped implementation."""
    matched = _get_matched(df)

    if "forecast_value" in matched.columns and "observation_value" in matched.columns:
        errors = matched["forecast_value"] - matched["observation_value"]
    else:
        return pd.DataFrame({group_by: [], "rmse": [], "sample_size": []})

    errors = errors.dropna()
    matched_idx = errors.index
    if len(matched_idx) == 0:
        return pd.DataFrame({group_by: [], "rmse": [], "sample_size": []})

    grouped = df.loc[matched_idx].groupby(group_by)

    results = []
    for group_key, group_df in grouped:
        mask = group_df.index.isin(matched_idx)
        group_errors = errors.loc[group_df.index[mask]].dropna()

        if len(group_errors) == 0:
            continue

        results.append(
            {
                "group_key": group_key,
                "rmse": float((group_errors**2).mean() ** 0.5),
                "sample_size": len(group_errors),
            }
        )

    return pd.DataFrame(results)


def _bias_grouped(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Bias grouped implementation."""
    matched = _get_matched(df)

    if "forecast_value" in matched.columns and "observation_value" in matched.columns:
        errors = matched["forecast_value"] - matched["observation_value"]
    else:
        return pd.DataFrame({group_by: [], "bias": [], "sample_size": []})

    errors = errors.dropna()
    matched_idx = errors.index
    if len(matched_idx) == 0:
        return pd.DataFrame({group_by: [], "bias": [], "sample_size": []})

    grouped = df.loc[matched_idx].groupby(group_by)

    results = []
    for group_key, group_df in grouped:
        mask = group_df.index.isin(matched_idx)
        group_errors = errors.loc[group_df.index[mask]].dropna()

        if len(group_errors) == 0:
            continue

        results.append(
            {
                "group_key": group_key,
                "bias": float(group_errors.mean()),
                "sample_size": len(group_errors),
            }
        )

    return pd.DataFrame(results)


# Override the mae/rmse/bias functions with proper implementations
def mae(df: pd.DataFrame, group_by: str | None = None) -> pd.DataFrame | float:
    """Compute Mean Absolute Error."""
    matched = _get_matched(df)

    if "abs_error" in matched.columns:
        errors = matched["abs_error"]
    elif "forecast_value" in matched.columns and "observation_value" in matched.columns:
        errors = (matched["forecast_value"] - matched["observation_value"]).abs()
    else:
        logger.warning("No error column found in DataFrame")
        return _empty_result(group_by)

    errors = errors.dropna()

    if group_by is None:
        return float(errors.mean()) if len(errors) > 0 else np.nan

    if group_by == "lead_hours":
        return _mae_grouped(df, group_by).rename(columns={"group_key": "lead_hours"})
    elif group_by == "model":
        return _mae_grouped(df, group_by).rename(columns={"group_key": "model"})
    elif group_by == "variable":
        return _mae_grouped(df, group_by).rename(columns={"group_key": "variable"})

    return _mae_grouped(df, group_by)


def rmse(df: pd.DataFrame, group_by: str | None = None) -> pd.DataFrame | float:
    """Compute Root Mean Squared Error."""
    matched = _get_matched(df)

    if "forecast_value" in matched.columns and "observation_value" in matched.columns:
        errors = matched["forecast_value"] - matched["observation_value"]
    else:
        logger.warning("No forecast/observation columns in DataFrame")
        return _empty_result(group_by)

    errors = errors.dropna()

    if group_by is None:
        return float((errors**2).mean() ** 0.5) if len(errors) > 0 else np.nan

    if group_by == "lead_hours":
        return _rmse_grouped(df, group_by).rename(columns={"group_key": "lead_hours"})
    elif group_by == "model":
        return _rmse_grouped(df, group_by).rename(columns={"group_key": "model"})
    elif group_by == "variable":
        return _rmse_grouped(df, group_by).rename(columns={"group_key": "variable"})

    return _rmse_grouped(df, group_by)


def bias(df: pd.DataFrame, group_by: str | None = None) -> pd.DataFrame | float:
    """Compute bias (mean error)."""
    matched = _get_matched(df)

    if "forecast_value" in matched.columns and "observation_value" in matched.columns:
        errors = matched["forecast_value"] - matched["observation_value"]
    else:
        logger.warning("No forecast/observation columns in DataFrame")
        return _empty_result(group_by)

    errors = errors.dropna()

    if group_by is None:
        return float(errors.mean()) if len(errors) > 0 else np.nan

    if group_by == "lead_hours":
        return _bias_grouped(df, group_by).rename(columns={"group_key": "lead_hours"})
    elif group_by == "model":
        return _bias_grouped(df, group_by).rename(columns={"group_key": "model"})
    elif group_by == "variable":
        return _bias_grouped(df, group_by).rename(columns={"group_key": "variable"})

    return _bias_grouped(df, group_by)
