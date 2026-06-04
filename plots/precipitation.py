"""Precipitation event analysis plots.

Deterministic event analysis for precipitation forecasts:
- Rain/no-rain thresholding (e.g., ≥1mm/h, ≥5mm/h)
- Hit/miss statistics
- False alarm statistics
- Event frequency plots

Does NOT yet implement:
- Probabilistic verification
- Brier score
- CRPS

Usage
-----
>>> from weather_analysis.plots.precipitation import (
...     plot_hit_miss_by_threshold,
...     plot_event_frequency,
... )
>>> fig = plot_hit_miss_by_threshold(df, "precipitation", thresholds=[1, 5, 10])
>>> fig.savefig("outputs/figures/hit_miss_by_threshold.png")
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
# Event detection helpers
# ---------------------------------------------------------------------------


def classify_events(
    df: pd.DataFrame,
    variable: str,
    threshold: float,
) -> pd.DataFrame:
    """Classify forecast-observation pairs into event categories.

    For deterministic precipitation forecasts, a pair is classified as:
    - Hit (H):  forecast ≥ threshold AND observed ≥ threshold
    - Miss (M):  forecast < threshold AND observed ≥ threshold
    - False alarm (FA): forecast ≥ threshold AND observed < threshold
    - Correct rejection (CR): forecast < threshold AND observed < threshold

    Args:
        df: Verification DataFrame with forecast_{var} and observed_{var}.
        variable: Variable name.
        threshold: Rainfall threshold (e.g., 1.0 mm/h).

    Returns:
        DataFrame with added 'event_category' column.
    """
    data = df.copy()

    forecast_col = f"forecast_{variable}"
    observed_col = f"observed_{variable}"

    if forecast_col not in data.columns or observed_col not in data.columns:
        logger.warning(
            "Missing columns for precipitation analysis: %s, %s",
            forecast_col,
            observed_col,
        )
        return data

    # Drop rows with NaN values
    data = data.dropna(subset=[forecast_col, observed_col])

    # Classify events
    conditions = [
        (data[forecast_col] >= threshold) & (data[observed_col] >= threshold),
        (data[forecast_col] < threshold) & (data[observed_col] >= threshold),
        (data[forecast_col] >= threshold) & (data[observed_col] < threshold),
        (data[forecast_col] < threshold) & (data[observed_col] < threshold),
    ]
    choices = ["Hit", "Miss", "FalseAlarm", "CorrectRejection"]

    data["event_category"] = np.select(conditions, choices, default="Unknown")

    logger.info(
        "Classified %d precipitation pairs at threshold %.1f mm/h",
        len(data),
        threshold,
    )
    return data


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_hit_miss_by_threshold(
    df: pd.DataFrame,
    variable: str,
    thresholds: list[float] | None = None,
    model: str | None = None,
) -> plt.Figure:
    """Plot hit rate and false alarm rate by rainfall threshold.

    Shows how forecast skill changes as the rainfall threshold increases:
    - Hit rate: fraction of observed events correctly predicted
    - False alarm rate: fraction of predicted events that didn't occur
    - Critical success index: combined skill metric

    Args:
        df: Verification DataFrame.
        variable: Variable name (e.g. "precipitation").
        thresholds: Rainfall thresholds to analyze. Defaults to [0.1, 1, 5, 10, 25].
        model: Optional model filter.

    Returns:
        Matplotlib Figure object.
    """
    if thresholds is None:
        thresholds = [0.1, 1, 5, 10, 25]

    forecast_col = f"forecast_{variable}"
    observed_col = f"observed_{variable}"

    data = df.dropna(subset=[forecast_col, observed_col]).copy()

    if model:
        data = data[data["model"] == model]

    fig, ax = make_figure(
        title=f"Event Detection Skill by Threshold — {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, f"Rainfall threshold ({units})", "Rate")

    hit_rates = []
    false_alarm_rates = []
    csi_scores = []

    for threshold in thresholds:
        # Observed events
        observed_event = data[observed_col] >= threshold
        observed_n = observed_event.sum()

        # Forecast events
        forecast_event = data[forecast_col] >= threshold

        # Hit: observed event AND forecast event
        hits = (observed_event & forecast_event).sum()

        # Miss: observed event BUT no forecast event
        misses = (observed_event & ~forecast_event).sum()

        # False alarm: forecast event BUT no observed event
        false_alarms = (~observed_event & forecast_event).sum()

        # Correct rejection
        correct_rejections = (~observed_event & ~forecast_event).sum()

        # Rates
        hit_rate = hits / observed_n if observed_n > 0 else np.nan
        fa_rate = false_alarms / (~observed_event).sum() if (~observed_event).sum() > 0 else np.nan
        csi = (
            hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
        )

        hit_rates.append(hit_rate)
        false_alarm_rates.append(fa_rate)
        csi_scores.append(csi)

        logger.debug(
            "Threshold %.1f: H=%d, M=%d, FA=%d, CR=%d | HR=%.3f, FAR=%.3f, CSI=%.3f",
            threshold,
            hits,
            misses,
            false_alarms,
            correct_rejections,
            hit_rate,
            fa_rate,
            csi,
        )

    # Plot
    ax.plot(
        thresholds,
        hit_rates,
        marker="o",
        linewidth=2,
        color=get_metric_color("mae"),
        label="Hit Rate",
    )
    ax.plot(
        thresholds,
        false_alarm_rates,
        marker="s",
        linewidth=2,
        color=get_metric_color("bias"),
        label="False Alarm Rate",
    )
    ax.plot(
        thresholds,
        csi_scores,
        marker="^",
        linewidth=2,
        color=get_metric_color("rmse"),
        label="Critical Success Index",
    )

    ax.legend(loc="upper left")
    ax.set_xlim(left=min(thresholds) * 0.9)

    return fig


def plot_event_frequency(
    df: pd.DataFrame,
    variable: str,
    thresholds: list[float] | None = None,
    model: str | None = None,
) -> plt.Figure:
    """Plot observed vs forecast event frequency by threshold.

    Shows whether the model overpredicts or underpredicts event frequency
    at different rainfall thresholds.

    Args:
        df: Verification DataFrame.
        variable: Variable name.
        thresholds: Rainfall thresholds.
        model: Optional model filter.

    Returns:
        Matplotlib Figure object.
    """
    if thresholds is None:
        thresholds = [0.1, 1, 5, 10, 25]

    forecast_col = f"forecast_{variable}"
    observed_col = f"observed_{variable}"

    data = df.dropna(subset=[forecast_col, observed_col]).copy()

    if model:
        data = data[data["model"] == model]

    fig, ax = make_figure(
        title=f"Event Frequency by Threshold — {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, f"Rainfall threshold ({units})", "Event frequency (%)")

    observed_freq = []
    forecast_freq = []

    for threshold in thresholds:
        obs_n = (data[observed_col] >= threshold).sum()
        fcst_n = (data[forecast_col] >= threshold).sum()
        total = len(data)

        observed_freq.append(100 * obs_n / total)
        forecast_freq.append(100 * fcst_n / total)

    ax.bar(
        [t - 0.3 for t in thresholds],
        observed_freq,
        width=0.6,
        color=get_metric_color("mae"),
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="Observed frequency",
    )
    ax.bar(
        [t + 0.3 for t in thresholds],
        forecast_freq,
        width=0.6,
        color=get_metric_color("rmse"),
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="Forecast frequency",
    )

    ax.legend(loc="upper left")
    ax.set_xlim(left=min(thresholds) - 0.5)

    return fig


def plot_event_contingency(
    df: pd.DataFrame,
    variable: str,
    threshold: float,
    model: str | None = None,
) -> plt.Figure:
    """Plot a contingency table heatmap.

    Shows the 2×2 contingency table for a given threshold:
    - Hit, Miss, False Alarm, Correct Rejection

    Args:
        df: Verification DataFrame.
        variable: Variable name.
        threshold: Rainfall threshold.
        model: Optional model filter.

    Returns:
        Matplotlib Figure object.
    """
    data = classify_events(df, variable, threshold)

    if model:
        data = data[data["model"] == model]

    fig, ax = make_figure(
        title=f"Contingency Table — {variable} ≥ {threshold} mm/h",
    )

    label_axes(ax, "", "")

    # Contingency table
    categories = ["Hit", "Miss", "FalseAlarm", "CorrectRejection"]
    counts = data["event_category"].value_counts().reindex(categories, fill_value=0)

    # 2×2 layout
    hit = counts["Hit"]
    miss = counts["Miss"]
    fa = counts["FalseAlarm"]
    cr = counts["CorrectRejection"]

    table_data = [
        [hit, miss],
        [fa, cr],
    ]

    im = ax.imshow(table_data, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Observed Event", "Observed No Event"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Forecast Event", "Forecast No Event"])

    # Add value labels
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{int(table_data[i][j])}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, label="Count")

    # Add overall statistics
    total = hit + miss + fa + cr
    hit_rate = hit / (hit + miss) if (hit + miss) > 0 else 0
    fa_rate = fa / (fa + cr) if (fa + cr) > 0 else 0
    csi = hit / (hit + miss + fa) if (hit + miss + fa) > 0 else 0

    stats_text = f"Total: {total}\nHit Rate: {hit_rate:.3f}\nFA Rate: {fa_rate:.3f}\nCSI: {csi:.3f}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_hit_miss_by_threshold(
    df: pd.DataFrame,
    variable: str,
    output_dir: str | Path,
    thresholds: list[float] | None = None,
    model: str | None = None,
) -> Path:
    """Generate and save a hit/miss by threshold plot.

    Args:
        df: Verification DataFrame.
        variable: Variable name.
        output_dir: Output directory.
        thresholds: Optional rainfall thresholds.
        model: Optional model filter.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_hit_miss_by_threshold(df, variable, thresholds=thresholds, model=model)

    model_suffix = f"_{model}" if model else ""
    filename = f"hit_miss_by_threshold_{variable}{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path


def save_event_frequency(
    df: pd.DataFrame,
    variable: str,
    output_dir: str | Path,
    thresholds: list[float] | None = None,
    model: str | None = None,
) -> Path:
    """Generate and save an event frequency plot.

    Args:
        df: Verification DataFrame.
        variable: Variable name.
        output_dir: Output directory.
        thresholds: Optional rainfall thresholds.
        model: Optional model filter.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_event_frequency(df, variable, thresholds=thresholds, model=model)

    model_suffix = f"_{model}" if model else ""
    filename = f"event_frequency_{variable}{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path
