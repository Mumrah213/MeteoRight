"""Seasonal analysis plots.

Visualizes how forecast metrics vary across the year:
- Month-by-month breakdowns
- Meteorological season comparisons (DJF, MAM, JJA, SON)
- Seasonal degradation comparisons

Meteorological seasons:
    DJF: December, January, February
    MAM: March, April, May
    JJA: June, July, August
    SON: September, October, November

Usage
-----
>>> from weather_analysis.plots.seasonal import (
...     plot_seasonal_metrics,
...     plot_monthly_metrics,
... )
>>> fig = plot_seasonal_metrics(df, "temperature_2m", "mae")
>>> fig.savefig("outputs/figures/mae_by_season_temperature_2m.png")
"""


import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import (
    get_metric_color,
    get_variable_units,
    label_axes,
    make_figure,
    save_figure,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------


def _has_meaningful_variation(values: np.ndarray, threshold: float = 1e-9) -> bool:
    """Check if values have meaningful variation (not all zeros or constant).

    Returns False if:
    - All values are NaN
    - All non-NaN values are the same (within threshold)
    - Standard deviation is below threshold
    """
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return False
    if len(valid) == 1:
        return True  # Single point is still meaningful

    # Check if all values are effectively zero or constant
    std = np.std(valid)
    if std < threshold:
        # All values are the same - check if they're all zero
        if np.abs(valid).max() < threshold:
            return False  # All zeros
    return True


# ---------------------------------------------------------------------------
# Season mapping
# ---------------------------------------------------------------------------

SEASON_MAP: dict[int, str] = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

MONTH_SEASON_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

MONTH_ORDER = list(range(1, 13))


# ---------------------------------------------------------------------------
# Season plots
# ---------------------------------------------------------------------------


def plot_seasonal_metrics(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    models: list[str] | None = None,
) -> plt.Figure | None:
    """Plot metrics by meteorological season.

    Args:
        df: Metrics DataFrame (long format).
        variable: Variable name.
        metric: Metric name (e.g. "mae", "rmse", "bias").
        models: List of model names. If None, all models.

    Returns:
        Matplotlib Figure object, or None if data has no meaningful variation.
    """
    data = df[(df["variable"] == variable) & (df["metric_name"] == metric)].copy()

    if data.empty:
        logger.info("No data for variable='%s', metric='%s'. Skipping.", variable, metric)
        return None

    # Check for meaningful variation
    if not _has_meaningful_variation(data["metric_value"].values):
        logger.info(
            "Skipping %s/%s seasonal plot: no meaningful variation (all zeros or constant)",
            variable,
            metric,
        )
        return None

    if models:
        data = data[data["model"].isin(models)]

    # Group by season, preserving order
    data["season"] = data.get("season", pd.Series(index=data.index))
    if "season" not in data.columns:
        logger.info("No season column in data for %s. Skipping.", variable)
        return None

    season_data = (
        data.groupby("season", sort=False)
        .agg(
            metric_value=("metric_value", "mean"),
            sample_size=("sample_size", "sum"),
        )
        .reindex(SEASON_ORDER)
    )

    fig, ax = make_figure(
        title=f"{metric.upper()} by Season — {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, "Season", f"{metric.upper()} ({units})")

    colors = [get_metric_color(metric)] * len(SEASON_ORDER)

    ax.bar(
        range(len(SEASON_ORDER)),
        season_data["metric_value"].values,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(range(len(SEASON_ORDER)))
    ax.set_xticklabels(SEASON_ORDER)

    # Add value labels on bars
    for i, (val, n) in enumerate(
        zip(season_data["metric_value"].values, season_data["sample_size"].values, strict=False)
    ):
        if not np.isnan(val):
            ax.text(
                i,
                val + 0.02 * max(season_data["metric_value"].dropna().max(), 0.1),
                f"{val:.2f}\nn={n}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    if models and len(models) > 1:
        ax.legend(title="Model", loc="upper left")

    return fig


def plot_monthly_metrics(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    models: list[str] | None = None,
) -> plt.Figure | None:
    """Plot metrics by month.

    Args:
        df: Metrics DataFrame (long format).
        variable: Variable name.
        metric: Metric name.
        models: List of model names.

    Returns:
        Matplotlib Figure object, or None if data has no meaningful variation.
    """
    data = df[(df["variable"] == variable) & (df["metric_name"] == metric)].copy()

    if data.empty:
        logger.info("No data for variable='%s', metric='%s'. Skipping.", variable, metric)
        return None

    # Check for meaningful variation
    if not _has_meaningful_variation(data["metric_value"].values):
        logger.info(
            "Skipping %s/%s monthly plot: no meaningful variation (all zeros or constant)",
            variable,
            metric,
        )
        return None

    if models:
        data = data[data["model"].isin(models)]

    if "month" not in data.columns:
        logger.info("No month column in data for %s. Skipping.", variable)
        return None

    month_data = (
        data.groupby("month", sort=True)
        .agg(
            metric_value=("metric_value", "mean"),
            sample_size=("sample_size", "sum"),
        )
        .reindex(MONTH_ORDER)
    )

    fig, ax = make_figure(
        title=f"{metric.upper()} by Month — {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, "Month", f"{metric.upper()} ({units})")

    month_names = [MONTH_SEASON_NAMES[m] for m in MONTH_ORDER]

    ax.plot(
        range(12),
        month_data["metric_value"].values,
        marker="o",
        markersize=6,
        linewidth=2,
        color=get_metric_color(metric),
    )

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names)

    # Add value labels
    for i, val in enumerate(month_data["metric_value"].values):
        if not np.isnan(val):
            ax.text(
                i,
                val + 0.02 * max(month_data["metric_value"].dropna().max(), 0.1),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    return fig


def plot_seasonal_degradation(
    df: pd.DataFrame,
    variable: str,
    models: list[str] | None = None,
    metrics: list[str] | None = None,
) -> plt.Figure | None:
    """Plot lead-time degradation curves grouped by season.

    Shows how forecast degradation differs across seasons:
    - Each season is a separate line
    - Useful for identifying seasonal skill differences

    Args:
        df: Metrics DataFrame (must have 'season' column).
        variable: Variable name.
        models: List of model names.
        metrics: List of metric names. Defaults to ["mae"].

    Returns:
        Matplotlib Figure object, or None if data has no meaningful variation.
    """
    if metrics is None:
        metrics = ["mae"]

    data = df[(df["variable"] == variable) & (df["metric_name"].isin(metrics))].copy()

    if data.empty:
        logger.info("No data for variable='%s'. Skipping.", variable)
        return None

    # Check for meaningful variation
    if not _has_meaningful_variation(data["metric_value"].values):
        logger.info(
            "Skipping %s seasonal degradation plot: no meaningful variation (all zeros or constant)",
            variable,
        )
        return None

    if "season" not in data.columns:
        logger.info("No season column in data for %s. Skipping.", variable)
        return None

    if models:
        data = data[data["model"].isin(models)]

    fig, ax = make_figure(
        title=f"{metrics[0].upper()} by Lead Time — Seasonal Comparison: {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, "Lead time (hours)", f"{metrics[0].upper()} ({units})")

    season_colors = {
        "DJF": "#1f77b4",  # Blue (winter)
        "MAM": "#2ca02c",  # Green (spring)
        "JJA": "#d62728",  # Red (summer)
        "SON": "#ff7f0e",  # Orange (autumn)
    }

    for season in SEASON_ORDER:
        season_data = data[data["season"] == season].sort_values("lead_hours")

        if season_data.empty:
            continue

        lead_hours = season_data["lead_hours"].values
        values = season_data["metric_value"].values

        ax.plot(
            lead_hours,
            values,
            color=season_colors.get(season, "gray"),
            marker="o",
            markersize=5,
            linewidth=2,
            label=season,
        )

    ax.legend(title="Season", loc="upper left")
    ax.set_xlim(left=0)

    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_seasonal_metrics(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    output_dir: str | Path,
    models: list[str] | None = None,
) -> Path | None:
    """Generate and save a seasonal metrics plot.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metric: Metric name.
        output_dir: Output directory.
        models: Optional list of models.

    Returns:
        Path to the saved figure, or None if plot was skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_seasonal_metrics(df, variable, metric, models=models)

    if fig is None:
        return None

    model_suffix = ""
    if models:
        model_suffix = "_" + "_vs_".join(models)

    filename = f"{metric}_by_season_{variable}{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path


def save_monthly_metrics(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    output_dir: str | Path,
    models: list[str] | None = None,
) -> Path | None:
    """Generate and save a monthly metrics plot.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metric: Metric name.
        output_dir: Output directory.
        models: Optional list of models.

    Returns:
        Path to the saved figure, or None if plot was skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_monthly_metrics(df, variable, metric, models=models)

    if fig is None:
        return None

    model_suffix = ""
    if models:
        model_suffix = "_" + "_vs_".join(models)

    filename = f"{metric}_by_month_{variable}{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path
