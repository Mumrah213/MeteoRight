"""Seasonal pattern analysis.

Quantifies seasonal forecast behavior:
- Seasonal bias patterns
- Seasonal skill differences
- Month-by-month degradation comparisons
- Seasonal error distribution shifts

Uses meteorological seasons:
    DJF: December, January, February
    MAM: March, April, May
    JJA: June, July, August
    SON: September, October, November

Usage
-----
>>> from weather_analysis.analysis.seasonal_patterns import (
...     seasonal_bias_summary,
...     seasonal_skill_comparison,
... )
>>> bias = seasonal_bias_summary(df, "temperature_2m")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..plots.seasonal import SEASON_ORDER

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seasonal bias summary
# ---------------------------------------------------------------------------


def seasonal_bias_summary(
    df: pd.DataFrame,
    variable: str,
    model: str | None = None,
) -> pd.DataFrame:
    """Compute bias by season.

    Shows whether forecasts are systematically over- or under-predicting
    in different seasons.

    Args:
        df: Metrics DataFrame (long format).
        variable: Variable name.
        model: Optional model filter.

    Returns:
        DataFrame with season, bias, sample_size columns.
    """
    data = df[(df["variable"] == variable) & (df["metric_name"] == "bias")].copy()

    if model:
        data = data[data["model"] == model]

    if "season" not in data.columns:
        logger.warning("No 'season' column in data for seasonal bias summary")
        return pd.DataFrame()

    result = (
        data.groupby("season")
        .agg(
            bias=("metric_value", "mean"),
            sample_size=("sample_size", "sum"),
        )
        .reindex(SEASON_ORDER)
    )

    logger.info("Seasonal bias for %s: %s", variable, result.to_dict("records"))
    return result


def seasonal_skill_summary(
    df: pd.DataFrame,
    variable: str,
    metrics: list[str] | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Compute skill metrics by season.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metrics: Metric names. Defaults to ["mae", "rmse", "bias"].
        model: Optional model filter.

    Returns:
        DataFrame with season, metric_name, metric_value, sample_size.
    """
    if metrics is None:
        metrics = ["mae", "rmse", "bias"]

    data = df[(df["variable"] == variable) & (df["metric_name"].isin(metrics))].copy()

    if model:
        data = data[data["model"] == model]

    if "season" not in data.columns:
        return pd.DataFrame()

    result = (
        data.groupby(["season", "metric_name"])
        .agg(
            metric_value=("metric_value", "mean"),
            sample_size=("sample_size", "sum"),
        )
        .reindex(
            pd.MultiIndex.from_product([SEASON_ORDER, metrics], names=["season", "metric_name"])
        )
    )

    return result.reset_index()


# ---------------------------------------------------------------------------
# Seasonal degradation comparison
# ---------------------------------------------------------------------------


def seasonal_degradation_comparison(
    df: pd.DataFrame,
    variable: str,
    model: str | None = None,
) -> pd.DataFrame:
    """Compare lead-time degradation across seasons.

    For each season, computes the degradation rate (slope of MAE vs lead_hours).
    This answers: "Do forecasts degrade faster in some seasons?"

    Args:
        df: Metrics DataFrame (must have 'season' column).
        variable: Variable name.
        model: Optional model filter.

    Returns:
        DataFrame with season, degradation_rate columns.
    """
    data = df[(df["variable"] == variable) & (df["metric_name"] == "mae")].copy()

    if model:
        data = data[data["model"] == model]

    if "season" not in data.columns:
        return pd.DataFrame()

    results = []
    for season in SEASON_ORDER:
        season_data = data[data["season"] == season].sort_values("lead_hours")

        if len(season_data) < 3:
            results.append(
                {
                    "season": season,
                    "degradation_rate": float("nan"),
                    "n_points": len(season_data),
                }
            )
            continue

        x = season_data["lead_hours"].values
        y = season_data["metric_value"].values

        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 3:
            results.append(
                {
                    "season": season,
                    "degradation_rate": float("nan"),
                    "n_points": mask.sum(),
                }
            )
            continue

        coeffs = np.polyfit(x[mask], y[mask], 1)
        results.append(
            {
                "season": season,
                "degradation_rate": float(coeffs[0]),
                "n_points": int(mask.sum()),
            }
        )

    result_df = pd.DataFrame(results)
    logger.info(
        "Seasonal degradation comparison for %s: %s", variable, result_df.to_dict("records")
    )
    return result_df


# ---------------------------------------------------------------------------
# Month-by-month analysis
# ---------------------------------------------------------------------------


def monthly_skill_summary(
    df: pd.DataFrame,
    variable: str,
    metrics: list[str] | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Compute skill metrics by month.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metrics: Metric names.
        model: Optional model filter.

    Returns:
        DataFrame with month, metric_name, metric_value, sample_size.
    """
    if metrics is None:
        metrics = ["mae", "rmse", "bias"]

    data = df[(df["variable"] == variable) & (df["metric_name"].isin(metrics))].copy()

    if model:
        data = data[data["model"] == model]

    if "month" not in data.columns:
        return pd.DataFrame()

    result = data.groupby(["month", "metric_name"]).agg(
        metric_value=("metric_value", "mean"),
        sample_size=("sample_size", "sum"),
    )

    return result.reset_index()
