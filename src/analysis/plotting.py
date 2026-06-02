"""Visualization helpers for weather analysis.

Simple, reusable plotting utilities optimized for interactive exploration.
Prefers matplotlib with seaborn support.

Usage
-----
>>> from src.analysis.plotting import forecast_vs_truth, mae_vs_leadtime
>>> aligned = align(forecast, truth)
>>> forecast_vs_truth(aligned[:1000])        # quick overlay plot
>>> mae_vs_leadtime(mae_by_leadtime(aligned))  # MAE degradation curve
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ── Defaults ────────────────────────────────────────────────────────────────


def _get_figsize(figsize: tuple[int, int] | None = None) -> tuple[int, int]:
    """Return default figure size for notebook display."""
    if figsize is not None:
        return figsize
    return (12, 6)


def _empty_plot(figsize: tuple[int, int] | None = None) -> plt.Figure:
    """Return a blank figure with a no-data message.

    Used as a fallback when the plotting function has no data to render.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=_get_figsize(figsize))
    ax.text(
        0.5,
        0.5,
        "No data available",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=14,
        color="gray",
    )
    ax.set_visible(False)
    plt.tight_layout()
    return fig


# ── Forecast vs Truth ──────────────────────────────────────────────────────


def forecast_vs_truth(
    aligned_df: pd.DataFrame,
    variable: str | None = None,
    model: str | None = None,
    lead_hours: int | None = None,
    figsize: tuple[int, int] | None = None,
    sample_size: int = 1000,
    ax: plt.Axes | None = None,
) -> plt.Figure | plt.Axes:
    """Plot forecast vs truth as line overlay.

    Filters to matched records and plots forecast_value vs observation_value
    over time. Useful for quick visual sanity checking.

    Parameters
    ----------
    aligned_df : DataFrame
        From align().
    variable : str or None
        Filter to specific variable.
    model : str or None
        Filter to specific model.
    lead_hours : int or None
        Filter to specific lead time.
    figsize : tuple or None
        Figure size (width, height).
    sample_size : int
        Maximum rows to plot (for large datasets).
    ax : Axes or None
        Matplotlib axes to plot on.

    Returns
    -------
    matplotlib Figure or Axes object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for plotting")
        raise

    # Filter data
    df = aligned_df.copy()

    if variable and "variable" in df.columns:
        df = df[df["variable"] == variable]
    if model and "model" in df.columns:
        df = df[df["model"] == model]
    if lead_hours is not None and "lead_hours" in df.columns:
        df = df[df["lead_hours"] == lead_hours]

    # Only matched records
    df = df[df["match_status"] == "matched"].copy()

    # Sample if too large
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).sort_values("forecast_target_time")

    # Determine time column
    time_col = None
    for col in ["forecast_target_time", "observation_time"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        logger.warning("No time column found")
        return _empty_plot(figsize)

    df = df.sort_values(time_col)

    fig, ax = plt.subplots(figsize=_get_figsize(figsize))

    # Plot forecast and truth
    ax.plot(
        df[time_col],
        df["forecast_value"],
        label="Forecast",
        alpha=0.7,
        linewidth=1,
        color="blue",
    )
    ax.plot(
        df[time_col],
        df["observation_value"],
        label="Observation",
        alpha=0.7,
        linewidth=1,
        color="green",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    title = "Forecast vs Truth"
    if variable:
        title += f" — {variable}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()

    return fig


# ── MAE vs Leadtime ─────────────────────────────────────────────────────────


def mae_vs_leadtime(
    mae_df: pd.DataFrame,
    variable: str | None = None,
    model: str | None = None,
    figsize: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | plt.Axes:
    """Plot MAE as a function of leadtime.

    The classic forecast degradation curve. Shows how error grows with lead time.

    Parameters
    ----------
    mae_df : DataFrame
        From mae_by_leadtime(). Expected columns: lead_hours, mae, sample_size.
    variable : str or None
        Variable label for the title.
    model : str or None
        Model label for the title.
    figsize : tuple or None
        Figure size.
    ax : Axes or None
        Matplotlib axes.

    Returns
    -------
    matplotlib Figure or Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise

    df = mae_df.copy()
    if "lead_hours" not in df.columns:
        raise ValueError("MAE DataFrame must have 'lead_hours' column")

    # Sort by lead time
    df = df.sort_values("lead_hours")

    fig, ax = plt.subplots(figsize=_get_figsize(figsize))

    ax.plot(
        df["lead_hours"],
        df["mae"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="red",
        label="MAE",
    )

    # Fill area under curve for visual emphasis
    ax.fill_between(
        df["lead_hours"],
        df["mae"],
        alpha=0.15,
        color="red",
    )

    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("MAE")
    title = "MAE vs Leadtime"
    if variable:
        title += f" — {variable}"
    if model:
        title += f" — {model}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add sample size annotations
    if "sample_size" in df.columns and len(df) < 50:
        for _, row in df.iterrows():
            ax.annotate(
                f"n={int(row['sample_size'])}",
                (row["lead_hours"], row["mae"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()

    return fig


# ── RMSE vs MAE Scatter ────────────────────────────────────────────────────


def error_comparison(
    metrics_df: pd.DataFrame,
    figsize: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | plt.Axes:
    """Scatter plot: RMSE vs MAE by leadtime.

    RMSE should be >= MAE (by mathematical property).
    Large gaps indicate heavy-tailed error distributions.

    Parameters
    ----------
    metrics_df : DataFrame
        Must have columns: lead_hours, mae, rmse (from compute_all_metrics).
    figsize : tuple or None
        Figure size.
    ax : Axes or None
        Matplotlib axes.

    Returns
    -------
    matplotlib Figure or Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise

    df = metrics_df.copy()

    fig, ax = plt.subplots(figsize=_get_figsize(figsize))

    ax.scatter(
        df["mae"],
        df["rmse"],
        s=df.get("sample_size", 100) * 2,
        alpha=0.6,
        color="blue",
    )

    # Add leadtime labels
    if "lead_hours" in df.columns:
        for _, row in df.iterrows():
            ax.annotate(
                f"{int(row['lead_hours'])}h",
                (row["mae"], row["rmse"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    # Reference line: RMSE = MAE (should be above this)
    max_val = max(df["rmse"].max(), df["mae"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "--", alpha=0.3, color="gray", label="RMSE = MAE")

    ax.set_xlabel("MAE")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs MAE by Lead Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


# ── Error Distribution ──────────────────────────────────────────────────────


def error_histogram(
    aligned_df: pd.DataFrame,
    variable: str | None = None,
    model: str | None = None,
    bins: int = 50,
    figsize: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | plt.Axes:
    """Histogram of forecast errors.

    Shows the distribution of (forecast - observation).
    Mean = bias. Median = 0 is ideal.

    Parameters
    ----------
    aligned_df : DataFrame
        From align().
    variable : str or None
        Filter to variable.
    model : str or None
        Filter to model.
    bins : int
        Number of histogram bins.
    figsize : tuple or None
        Figure size.
    ax : Axes or None
        Matplotlib axes.

    Returns
    -------
    matplotlib Figure or Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise

    df = aligned_df.copy()

    if variable and "variable" in df.columns:
        df = df[df["variable"] == variable]
    if model and "model" in df.columns:
        df = df[df["model"] == model]

    # Only matched records
    df = df[df["match_status"] == "matched"].copy()

    # Compute errors if not present
    if "error" in df.columns:
        errors = df["error"].dropna()
    elif "forecast_value" in df.columns and "observation_value" in df.columns:
        errors = df["forecast_value"] - df["observation_value"]
        errors = errors.dropna()
    else:
        logger.warning("No error data available")
        return _empty_plot(figsize)

    fig, ax = plt.subplots(figsize=_get_figsize(figsize))

    ax.hist(
        errors,
        bins=bins,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )

    # Mark mean (bias)
    bias_val = errors.mean()
    ax.axvline(
        bias_val, color="red", linestyle="--", linewidth=2, label=f"Mean (bias) = {bias_val:.2f}"
    )

    # Mark zero
    ax.axvline(0, color="green", linestyle="-", linewidth=1, alpha=0.5, label="Zero")

    ax.set_xlabel("Error (Forecast - Observation)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    return fig


# ── Missing Data Visualization ──────────────────────────────────────────────


def missing_data_plot(
    aligned_df: pd.DataFrame,
    figsize: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | plt.Axes:
    """Visualize alignment gaps (missing forecasts or observations).

    Plots match_status distribution over time.

    Parameters
    ----------
    aligned_df : DataFrame
        From align().
    figsize : tuple or None
        Figure size.
    ax : Axes or None
        Matplotlib axes.

    Returns
    -------
    matplotlib Figure or Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise

    df = aligned_df.copy()

    if "match_status" not in df.columns:
        logger.warning("No match_status column — nothing to plot")
        return _empty_plot(figsize)

    # Determine time column
    time_col = None
    for col in ["forecast_target_time", "observation_time"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        time_col = "forecast_target_time"

    # Count by status over time (sample for large datasets)
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)

    df = df.sort_values(time_col)

    # Create a time bin and count statuses
    df["_bin"] = pd.cut(
        df[time_col],
        bins=min(50, len(df) // 10),
        labels=False,
    )

    status_counts = df.groupby(["_bin", "match_status"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=_get_figsize(figsize))

    status_counts.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#4CAF50", "#FFC107", "#F44336"],  # green, amber, red
    )

    ax.set_xlabel("Time Bins")
    ax.set_ylabel("Count")
    ax.set_title("Alignment Status Over Time")
    ax.legend(title="Status")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    return fig


# ── Bias Map (if spatial data available) ────────────────────────────────────


def bias_spatial(
    aligned_df: pd.DataFrame,
    figsize: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | plt.Axes:
    """Plot bias as a function of leadtime and model.

    Simple line plot: bias by leadtime for each model.

    Parameters
    ----------
    aligned_df : DataFrame
        From align().
    figsize : tuple or None
        Figure size.
    ax : Axes or None
        Matplotlib axes.

    Returns
    -------
    matplotlib Figure or Axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise

    df = aligned_df.copy()

    # Compute bias by leadtime and model
    matched = df[df["match_status"] == "matched"].copy()

    if "forecast_value" in matched.columns and "observation_value" in matched.columns:
        matched["error"] = matched["forecast_value"] - matched["observation_value"]
    elif "error" in matched.columns:
        pass  # Already computed
    else:
        logger.warning("No error data available")
        return _empty_plot(figsize)

    has_model = "model" in matched.columns
    has_leadtime = "lead_hours" in matched.columns

    if not has_leadtime:
        logger.warning("No lead_hours column — cannot plot bias by leadtime")
        return _empty_plot(figsize)

    fig, ax = plt.subplots(figsize=_get_figsize(figsize))

    if has_model:
        for model_name, model_df in matched.groupby("model"):
            grouped = model_df.groupby("lead_hours")["error"].mean()
            ax.plot(
                grouped.index,
                grouped.values,
                marker="o",
                linewidth=2,
                label=str(model_name),
            )
    else:
        grouped = matched.groupby("lead_hours")["error"].mean()
        ax.plot(
            grouped.index,
            grouped.values,
            marker="o",
            linewidth=2,
            label="All models",
        )

    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("Bias (Forecast - Observation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
