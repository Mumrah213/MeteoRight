"""Forecast-observation alignment utilities.

Aligns forecast and observation DataFrames on common keys:
  - valid_time (forecast_target_time == observation_time)
  - variable name
  - location (latitude, longitude)

Preserves leadtime and model metadata from forecasts.

Expects long-format frames: forecasts carrying ``forecast_target_time``,
``variable`` and ``forecast_value``; observations carrying ``variable`` and
``observation_value``. Both are produced by ``verification.loaders``.

Usage
-----
>>> from verification.loaders import load_forecasts, load_observations
>>> from analysis.alignment import align

>>> data_dir = Path("data/copenhagen_demo")
>>> forecast = load_forecasts(data_dir, "ecmwf_ifs_single", "2026-05-20", "2026-05-22")
>>> truth = load_observations(data_dir, "2026-05-20", "2026-05-22")
>>> aligned = align(forecast, truth)
>>> aligned.head()
   forecast_issue_time forecast_target_time  lead_hours  model  latitude  ...
   observation_value   match_status        ...
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from verification.location import validate_location_compatibility

logger = logging.getLogger(__name__)


# ── Canonical alignment schema ──────────────────────────────────────────────

# Columns expected in aligned output
ALIGNED_COLUMNS = [
    "forecast_issue_time",  # Model initialization time
    "forecast_target_time",  # Valid time (same as observation_time when matched)
    "lead_hours",  # Forecast lead time in hours
    "model",  # Model name
    "latitude",  # Location latitude
    "longitude",  # Location longitude
    "location_id",  # Canonical location id, if available
    "variable",  # Variable name
    "observation_value",  # Truth / observed value
    "forecast_value",  # Predicted value
    "error",  # forecast_value - observation_value
    "abs_error",  # |forecast_value - observation_value|
    "match_status",  # "matched" | "forecast_only" | "observation_only"
]


# ── Core alignment ──────────────────────────────────────────────────────────


def align(
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    valid_time_col: str = "observation_time",
    target_time_col: str = "forecast_target_time",
    max_location_distance_km: float = 50.0,
) -> pd.DataFrame:
    """Align forecast and observation DataFrames.

    Merges on valid_time = observation_time, variable, and location.
    Each forecast run is aligned separately against observations.

    Parameters
    ----------
    forecast_df : DataFrame
        From load_forecast(). Expected columns include:
        forecast_target_time, lead_hours, model, variable, value
    truth_df : DataFrame
        From load_historical(). Expected columns include:
        observation_time, variable, value
    valid_time_col : str
        Column name in truth_df for the observation time.
    target_time_col : str
        Column name in forecast_df for the forecast valid time.
    lat_lon_tolerance : float
        Tolerance in degrees for latitude/longitude matching
        (to account for different grid resolutions between APIs).

    Returns
    -------
    DataFrame with columns:
        forecast_issue_time, forecast_target_time, lead_hours, model,
        latitude, longitude, variable, observation_value, forecast_value,
        error, abs_error, match_status

    Examples
    --------
    >>> forecast = load_forecast("icon_eu", "temperature_2m", start="2024-01")
    >>> truth = load_historical("temperature_2m", start="2024-01")
    >>> aligned = align(forecast, truth)
    """
    if forecast_df.empty and truth_df.empty:
        return pd.DataFrame(columns=ALIGNED_COLUMNS)

    validate_location_compatibility(
        forecast_df,
        truth_df,
        max_distance_km=max_location_distance_km,
    )

    # ── Prepare forecast DataFrame ─────────────────────────────────────
    fcst = forecast_df.copy()

    # Standardize column names for value
    if "value" not in fcst.columns:
        raise ValueError("Forecast DataFrame missing 'value' column")

    fcst = fcst.rename(columns={"value": "forecast_value"})

    # Ensure we have the required columns
    fcst_cols = [
        c
        for c in ALIGNED_COLUMNS
        if c != "observation_value" and c != "error" and c != "abs_error" and c != "match_status"
    ]
    fcst_cols = [c for c in fcst_cols if c in fcst.columns]
    fcst = fcst[fcst_cols].copy()

    # ── Prepare truth DataFrame ────────────────────────────────────────
    obs = truth_df.copy()

    if "value" not in obs.columns:
        raise ValueError("Observation DataFrame missing 'value' column")

    obs = obs.rename(columns={"value": "observation_value"})
    obs = obs.rename(columns={valid_time_col: "forecast_target_time"})

    # Select relevant columns
    obs_cols = [
        "forecast_target_time",
        "latitude",
        "longitude",
        "location_id",
        "variable",
        "observation_value",
    ]
    obs_cols = [c for c in obs_cols if c in obs.columns]
    obs = obs[obs_cols].copy()

    # ── Merge ──────────────────────────────────────────────────────────
    # Single-location datasets match on time + variable after a strict
    # location compatibility check. Multi-location datasets must have a
    # location_id and match on it as well.
    merge_keys = ["forecast_target_time", "variable"]
    if "location_id" in fcst.columns and "location_id" in obs.columns:
        merge_keys.append("location_id")

    # Left join: all forecasts + matching observations
    aligned = fcst.merge(obs, on=merge_keys, how="left", suffixes=("", "_obs"))

    # If observation columns weren't found, create them as None
    for col in ["observation_value"]:
        if col not in aligned.columns:
            aligned[col] = None

    # ── Compute errors ─────────────────────────────────────────────────
    aligned["match_status"] = "matched"

    # Mark unmatched
    obs_mask = aligned["observation_value"].isna()
    fcst_only_mask = aligned["forecast_target_time"].isna()

    aligned.loc[obs_mask & ~fcst_only_mask, "match_status"] = "forecast_only"
    aligned.loc[fcst_only_mask, "match_status"] = "observation_only"

    # Compute error (only for matched records)
    aligned["error"] = None
    aligned["abs_error"] = None

    match_mask = aligned["match_status"] == "matched"
    if match_mask.any():
        fcst_vals = aligned.loc[match_mask, "forecast_value"].astype(float)
        obs_vals = aligned.loc[match_mask, "observation_value"].astype(float)
        aligned.loc[match_mask, "error"] = fcst_vals - obs_vals
        aligned.loc[match_mask, "abs_error"] = (fcst_vals - obs_vals).abs()

    # ── Select canonical output columns ────────────────────────────────
    output_cols = [c for c in ALIGNED_COLUMNS if c in aligned.columns]
    result = aligned[output_cols].copy()

    # Sort by time, model, variable for consistency
    sort_cols = [c for c in ["forecast_target_time", "model", "variable"] if c in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)

    logger.info(
        "Aligned %d forecast rows × %d observation rows → %d aligned rows",
        len(forecast_df),
        len(truth_df),
        len(result),
    )

    return result


# ── Convenience: align and compute metrics in one step ──────────────────────


def align_and_evaluate(
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Align forecast and truth, then compute summary metrics.

    Parameters
    ----------
    forecast_df : DataFrame
        From load_forecast().
    truth_df : DataFrame
        From load_historical().

    Returns
    -------
    aligned : DataFrame
        Aligned DataFrame from align().
    summary : dict
        Summary statistics:
            total_rows, matched_rows, forecast_only, observation_only,
            mae, rmse, bias, coverage_fraction
    """
    aligned = align(forecast_df, truth_df)

    matched = aligned[aligned["match_status"] == "matched"].copy()

    summary: dict[str, Any] = {
        "total_rows": len(aligned),
        "matched_rows": len(matched),
        "forecast_only": len(aligned[aligned["match_status"] == "forecast_only"]),
        "observation_only": len(aligned[aligned["match_status"] == "observation_only"]),
    }

    if len(matched) > 0:
        errors = matched["error"].dropna()
        abs_errors = matched["abs_error"].dropna()

        summary["mae"] = float(abs_errors.mean()) if len(abs_errors) > 0 else None
        summary["rmse"] = float((errors**2).mean() ** 0.5) if len(errors) > 0 else None
        summary["bias"] = float(errors.mean()) if len(errors) > 0 else None
        summary["coverage_fraction"] = len(matched) / max(len(aligned), 1)
    else:
        summary["mae"] = None
        summary["rmse"] = None
        summary["bias"] = None
        summary["coverage_fraction"] = 0.0

    return aligned, summary


# ── Grouping helpers ────────────────────────────────────────────────────────


def group_by_leadtime(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Group aligned DataFrame by lead_hours.

    Args:
        df: Aligned DataFrame from align().

    Returns:
        Dict mapping lead_hours → DataFrame subset.
    """
    if "lead_hours" not in df.columns:
        return {"all": df}

    grouped: dict[int, pd.DataFrame] = {}
    for lt, group in df.groupby("lead_hours"):
        grouped[int(lt)] = group.reset_index(drop=True)
    return grouped


def group_by_model(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group aligned DataFrame by model.

    Args:
        df: Aligned DataFrame from align().

    Returns:
        Dict mapping model name → DataFrame subset.
    """
    if "model" not in df.columns:
        return {"all": df}

    grouped: dict[str, pd.DataFrame] = {}
    for model, group in df.groupby("model"):
        grouped[str(model)] = group.reset_index(drop=True)
    return grouped


def group_by_variable(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group aligned DataFrame by variable.

    Args:
        df: Aligned DataFrame from align().

    Returns:
        Dict mapping variable name → DataFrame subset.
    """
    if "variable" not in df.columns:
        return {"all": df}

    grouped: dict[str, pd.DataFrame] = {}
    for var, group in df.groupby("variable"):
        grouped[str(var)] = group.reset_index(drop=True)
    return grouped


def filter_matched(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only matched records.

    Args:
        df: Aligned DataFrame from align().

    Returns:
        DataFrame with only match_status == "matched" rows.
    """
    if "match_status" not in df.columns:
        return df.copy()
    return df[df["match_status"] == "matched"].reset_index(drop=True)


def filter_forecast_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to forecast-only (no observation).

    Args:
        df: Aligned DataFrame from align().

    Returns:
        DataFrame with only match_status == "forecast_only" rows.
    """
    if "match_status" not in df.columns:
        return df.copy()
    return df[df["match_status"] == "forecast_only"].reset_index(drop=True)


def filter_observation_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to observation-only (no forecast).

    Args:
        df: Aligned DataFrame from align().

    Returns:
        DataFrame with only match_status == "observation_only" rows.
    """
    if "match_status" not in df.columns:
        return df.copy()
    return df[df["match_status"] == "observation_only"].reset_index(drop=True)
