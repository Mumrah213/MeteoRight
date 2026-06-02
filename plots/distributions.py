"""Error distribution analysis plots.

Visualizes the distribution of forecast errors:
- Histograms with KDE overlay
- Boxplots by lead time
- Percentile summaries
- Quantile plots

These plots reveal:
- Whether errors are symmetric or skewed
- Presence of heavy tails (catastrophic misses)
- Error variance changes with lead time
- Systematic asymmetries

Usage
-----
>>> from weather_analysis.plots.distributions import (
...     plot_error_histogram,
...     plot_error_boxplot,
...     plot_error_percentiles,
... )
>>> fig = plot_error_histogram(df, "temperature_2m", 24)
>>> fig.savefig("outputs/figures/error_histogram_temperature_2m_24h.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    pass

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
# Histogram plots
# ---------------------------------------------------------------------------


def plot_error_histogram(
    errors: pd.Series,
    variable: str,
    bins: int = 50,
    kde: bool = True,
    title: str = "",
) -> plt.Figure | None:
    """Plot a histogram of forecast errors.

    Shows the distribution of errors for a single variable (optionally
    filtered by lead time). Includes KDE overlay and key statistics.

    Args:
        errors: Series of forecast errors (forecast - observed).
        variable: Variable name for labeling.
        bins: Number of histogram bins.
        kde: Whether to overlay a KDE curve.
        title: Override title. If None, auto-generated.

    Returns:
        Matplotlib Figure object, or None if data has no meaningful variation.
    """
    clean = errors.dropna()

    if clean.empty:
        logger.info("No error data for %s. Skipping histogram.", variable)
        return None

    # Check for meaningful variation
    if not _has_meaningful_variation(clean.values):
        logger.info(
            "Skipping %s histogram: no meaningful variation (all zeros or constant)", variable
        )
        return None

    fig, ax = make_figure(
        title=title or f"Error Distribution — {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, f"Error ({units})", "Frequency")

    # Histogram
    ax.hist(
        clean,
        bins=bins,
        color=get_metric_color("error_hist"),
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        density=False,
    )

    # KDE overlay
    if kde:
        from scipy.stats import gaussian_kde

        try:
            kde_x = np.linspace(clean.min(), clean.max(), 200)
            kde_y = gaussian_kde(clean)(kde_x)
            ax.plot(
                kde_x,
                kde_y * len(clean) * (clean.max() - clean.min()) / bins,
                color="red",
                linewidth=2,
                label="KDE",
            )
        except Exception:
            pass  # KDE may fail with very small samples

    # Zero line
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Statistics annotation
    stats_text = (
        f"Mean: {clean.mean():.2f}\n"
        f"Median: {clean.median():.2f}\n"
        f"Std: {clean.std():.2f}\n"
        f"n: {len(clean)}"
    )
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Percentile markers
    for p in (10, 25, 50, 75, 90):
        val = np.percentile(clean, p)
        ax.axvline(x=val, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    return fig


# ---------------------------------------------------------------------------
# Boxplot plots
# ---------------------------------------------------------------------------


def plot_error_boxplot(
    df: pd.DataFrame,
    variable: str,
    lead_times: list[int] | None = None,
    model: str | None = None,
) -> plt.Figure | None:
    """Plot error boxplots by lead time.

    Shows how error distributions change with lead time:
    - Median (box line)
    - IQR (box)
    - Whiskers (1.5×IQR)
    - Outliers (points)

    Useful for:
    - Detecting increasing error variance with lead time
    - Identifying asymmetric error distributions
    - Spotting outliers/catastrophic misses

    Args:
        df: Verification DataFrame (must contain {variable}_error and lead_hours).
        variable: Variable name.
        lead_times: Specific lead times to include. If None, all lead times.
        model: Specific model to filter. If None, all models.

    Returns:
        Matplotlib Figure object, or None if data has no meaningful variation.
    """
    error_col = f"{variable}_error"
    if error_col not in df.columns:
        logger.info("No error column '%s' in data. Skipping boxplot.", error_col)
        return None

    data = df[[error_col, "lead_hours"]].dropna()

    if model:
        data = df[df["model"] == model][[error_col, "lead_hours"]].dropna()

    if lead_times:
        data = data[data["lead_hours"].isin(lead_times)]

    if data.empty:
        logger.info("No error data for %s. Skipping boxplot.", variable)
        return None

    # Check for meaningful variation
    if not _has_meaningful_variation(data[error_col].values):
        logger.info(
            "Skipping %s boxplot: no meaningful variation (all zeros or constant)", variable
        )
        return None

    fig, ax = make_figure(
        title=f"Error Distribution by Lead Time — {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, "Lead time (hours)", f"Error ({units})")

    # Group by lead time
    grouped = data.groupby("lead_hours")[error_col]

    # Get unique lead times sorted
    lead_times_sorted = sorted(data["lead_hours"].unique())

    # Boxplot
    ax.boxplot(
        [grouped.get_group(lt).values for lt in lead_times_sorted],
        positions=lead_times_sorted,
        widths=0.6,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor=get_metric_color("error_hist"), alpha=0.5, edgecolor="black"),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
        flierprops=dict(marker="o", markerfacecolor="red", markersize=3, alpha=0.5),
    )

    ax.set_xlim(left=min(lead_times_sorted) - 1, right=max(lead_times_sorted) + 1)

    # Add median values as text
    for lt in lead_times_sorted:
        group = grouped.get_group(lt)
        if not group.empty:
            median_val = group.median()
            ax.text(lt, median_val, f"{median_val:.1f}", ha="center", va="bottom", fontsize=7)

    return fig


# ---------------------------------------------------------------------------
# Percentile plots
# ---------------------------------------------------------------------------


def plot_error_percentiles(
    df: pd.DataFrame,
    variable: str,
    lead_times: list[int] | None = None,
    percentiles: list[int] | None = None,
    model: str | None = None,
) -> plt.Figure | None:
    """Plot error percentiles by lead time.

    Shows how different parts of the error distribution change with lead time:
    - P10, P25, P50 (median), P75, P90
    - Useful for understanding tail behavior

    Args:
        df: Verification DataFrame.
        variable: Variable name.
        lead_times: Specific lead times to include.
        percentiles: Percentiles to plot. Defaults to [10, 25, 50, 75, 90].
        model: Specific model to filter.

    Returns:
        Matplotlib Figure object, or None if data has no meaningful variation.
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    error_col = f"{variable}_error"
    if error_col not in df.columns:
        logger.info("No error column '%s' in data. Skipping percentiles.", error_col)
        return None

    data = df[[error_col, "lead_hours"]].dropna()

    if model:
        data = df[df["model"] == model][[error_col, "lead_hours"]].dropna()

    if lead_times:
        data = data[data["lead_hours"].isin(lead_times)]

    if data.empty:
        logger.info("No error data for %s. Skipping percentiles.", variable)
        return None

    # Check for meaningful variation
    if not _has_meaningful_variation(data[error_col].values):
        logger.info(
            "Skipping %s percentiles: no meaningful variation (all zeros or constant)", variable
        )
        return None

    fig, ax = make_figure(
        title=f"Error Percentiles by Lead Time — {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, "Lead time (hours)", f"Error ({units})")

    lead_times_sorted = sorted(data["lead_hours"].unique())

    for p in percentiles:
        values = []
        for lt in lead_times_sorted:
            group = data[data["lead_hours"] == lt][error_col]
            if not group.empty:
                values.append(np.percentile(group, p))
            else:
                values.append(np.nan)

        label = f"P{p}"
        if p == 50:
            label = "Median"
        elif p == 25:
            label = "P25"
        elif p == 75:
            label = "P75"

        ax.plot(
            lead_times_sorted,
            values,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=label,
        )

    # Zero line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.legend(loc="upper left")
    ax.set_xlim(left=0)

    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_error_histogram(
    df: pd.DataFrame,
    variable: str,
    lead_hours: int,
    output_dir: str | Path,
    model: str | None = None,
) -> Path | None:
    """Generate and save an error histogram plot.

    Args:
        df: Verification DataFrame.
        variable: Variable name.
        lead_hours: Lead time to filter.
        output_dir: Output directory.
        model: Optional model filter.

    Returns:
        Path to the saved figure, or None if plot was skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    error_col = f"{variable}_error"
    data = df[df["lead_hours"] == lead_hours][error_col]

    if model:
        data = df[(df["lead_hours"] == lead_hours) & (df["model"] == model)][error_col]

    fig = plot_error_histogram(
        data, variable, title=f"Error Distribution — {variable} @ {lead_hours}h"
    )

    if fig is None:
        return None

    model_suffix = f"_{model}" if model else ""
    filename = f"error_histogram_{variable}_{lead_hours}h{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path


def save_error_boxplot(
    df: pd.DataFrame,
    variable: str,
    output_dir: str | Path,
    lead_times: list[int] | None = None,
    model: str | None = None,
) -> Path | None:
    """Generate and save an error boxplot.

    Args:
        df: Verification DataFrame.
        variable: Variable name.
        output_dir: Output directory.
        lead_times: Optional lead time filter.
        model: Optional model filter.

    Returns:
        Path to the saved figure, or None if plot was skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_error_boxplot(df, variable, lead_times=lead_times, model=model)

    if fig is None:
        return None

    model_suffix = f"_{model}" if model else ""
    lt_suffix = f"_lt{'_'.join(str(lt_val) for lt_val in lead_times)}" if lead_times else ""
    filename = f"error_boxplot_{variable}{lt_suffix}{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path
