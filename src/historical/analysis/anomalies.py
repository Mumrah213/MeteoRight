"""Anomaly detection for forecast verification.

Identifies unusual forecast behavior:
- Groups with unusually high errors (relative to their lead time)
- Seasonal anomalies (bias significantly different from annual average)
- Outlier detection using statistical methods

Usage
-----
>>> from weather_analysis.analysis.anomalies import (
...     detect_high_error_groups,
...     detect_seasonal_bias_anomalies,
... )
>>> anomalies = detect_high_error_groups(df, "temperature_2m")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# High error detection
# ---------------------------------------------------------------------------


def detect_high_error_groups(
    df: pd.DataFrame,
    variable: str,
    metric: str = "mae",
    n_sigma: float = 2.0,
    model: str | None = None,
    groupby: list[str] | None = None,
) -> pd.DataFrame:
    """Detect groups with unusually high errors.

    Uses z-scores to identify groups whose metric value exceeds
    the mean by more than n_sigma standard deviations.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metric: Metric name.
        n_sigma: Number of standard deviations for threshold.
        model: Optional model filter.
        groupby: Columns to group by. If None, uses lead_hours.

    Returns:
        DataFrame with flagged high-error groups.
    """
    data = df[(df["variable"] == variable) & (df["metric_name"] == metric)].copy()

    if model:
        data = data[data["model"] == model]

    if data.empty:
        return pd.DataFrame()

    if groupby is None:
        groupby = ["lead_hours"]

    # Compute global statistics
    global_mean = data["metric_value"].mean()
    global_std = data["metric_value"].std()

    if global_std == 0 or np.isnan(global_std):
        logger.info("No variance in %s for %s — no anomalies detected", metric, variable)
        return pd.DataFrame()

    # Compute group means
    group_means = data.groupby(groupby)["metric_value"].mean().reset_index()
    group_means.columns = groupby + ["group_mean"]

    # Compute z-scores
    group_means["z_score"] = (group_means["group_mean"] - global_mean) / global_std
    group_means["is_anomaly"] = group_means["z_score"].abs() > n_sigma

    anomalies = group_means[group_means["is_anomaly"]].copy()

    logger.info(
        "Detected %d high-error groups for %s / %s (threshold: %.1f sigma)",
        len(anomalies),
        variable,
        metric,
        n_sigma,
    )
    return anomalies


# ---------------------------------------------------------------------------
# Seasonal bias anomalies
# ---------------------------------------------------------------------------


def detect_seasonal_bias_anomalies(
    df: pd.DataFrame,
    variable: str,
    n_sigma: float = 2.0,
    model: str | None = None,
) -> pd.DataFrame:
    """Detect seasons with unusually biased forecasts.

    Compares each season's bias to the annual average bias.
    Seasons with bias significantly different from the annual
    average are flagged as anomalies.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        n_sigma: Number of standard deviations for threshold.
        model: Optional model filter.

    Returns:
        DataFrame with seasonal bias anomalies.
    """
    data = df[(df["variable"] == variable) & (df["metric_name"] == "bias")].copy()

    if model:
        data = data[data["model"] == model]

    if "season" not in data.columns:
        return pd.DataFrame()

    # Compute seasonal means
    seasonal_bias = data.groupby("season")["metric_value"].mean()

    # Compute global statistics
    global_mean = seasonal_bias.mean()
    global_std = seasonal_bias.std()

    if global_std == 0 or np.isnan(global_std):
        return pd.DataFrame()

    # Z-scores
    seasonal_bias_df = seasonal_bias.reset_index()
    seasonal_bias_df.columns = ["season", "bias"]
    seasonal_bias_df["z_score"] = (seasonal_bias_df["bias"] - global_mean) / global_std
    seasonal_bias_df["is_anomaly"] = seasonal_bias_df["z_score"].abs() > n_sigma

    anomalies = seasonal_bias_df[seasonal_bias_df["is_anomaly"]]

    logger.info(
        "Detected %d seasonal bias anomalies for %s",
        len(anomalies),
        variable,
    )
    return anomalies


# ---------------------------------------------------------------------------
# Outlier detection using IQR method
# ---------------------------------------------------------------------------


def detect_outliers_iqr(
    df: pd.DataFrame,
    variable: str,
    metric: str = "mae",
    iqr_multiplier: float = 1.5,
    model: str | None = None,
) -> pd.DataFrame:
    """Detect outliers using the IQR method.

    Identifies groups where the metric value falls outside
    [Q1 - k*IQR, Q3 + k*IQR].

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metric: Metric name.
        iqr_multiplier: IQR multiplier (1.5 for mild outliers, 3.0 for extreme).
        model: Optional model filter.

    Returns:
        DataFrame with outlier groups.
    """
    data = df[(df["variable"] == variable) & (df["metric_name"] == metric)].copy()

    if model:
        data = data[data["model"] == model]

    if data.empty:
        return pd.DataFrame()

    q1 = data["metric_value"].quantile(0.25)
    q3 = data["metric_value"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    outliers = data[(data["metric_value"] < lower) | (data["metric_value"] > upper)]

    logger.info(
        "Detected %d outliers for %s / %s using IQR method (multiplier=%.1f)",
        len(outliers),
        variable,
        metric,
        iqr_multiplier,
    )
    return outliers
