"""Model comparison plots.

Visualizes forecast skill differences between models:
- Overlay plots: multiple models on same axes
- Grouped plots: side-by-side comparison
- MAE and RMSE by lead time
- Bias comparison

Usage
-----
>>> from weather_analysis.plots.model_comparison import (
...     plot_model_comparison,
...     plot_model_bias_comparison,
... )
>>> fig = plot_model_comparison(df, "temperature_2m", "mae")
>>> fig.savefig("outputs/figures/mae_by_lead_time_icon_vs_gfs.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from .utils import (
    annotate_sample_size,
    get_metric_color,
    get_model_color,
    get_model_linestyle,
    get_variable_units,
    label_axes,
    make_figure,
    make_multi_panel_figure,
    save_figure,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Overlay plots
# ---------------------------------------------------------------------------


def plot_model_comparison(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    models: list[str] | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> plt.Figure:
    """Plot multiple models' metrics vs lead time on the same axes.

    This is the primary comparison plot: it overlays models so that
    skill differences are immediately visible.

    Args:
        df: Metrics DataFrame (long format).
        variable: Variable name.
        metric: Metric name (e.g. "mae", "rmse").
        models: List of model names. If None, all models in data are used.
        log_x: Logarithmic x-axis.
        log_y: Logarithmic y-axis.

    Returns:
        Matplotlib Figure object.
    """
    # Filter to variable and metric
    data = df[(df["variable"] == variable) & (df["metric_name"] == metric)].copy()

    if data.empty:
        fig, ax = make_figure(title=f"No data for {variable} / {metric}")
        return fig

    # Filter to specific models
    if models:
        data = data[data["model"].isin(models)]

    model_list = sorted(str(m) for m in data["model"].dropna().unique())

    fig, ax = make_figure(
        title=f"{metric.upper()} by Lead Time — Model Comparison: {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, "Lead time (hours)", f"{metric.upper()} ({units})")

    for idx, model in enumerate(model_list):
        model_data = data[data["model"] == model].sort_values("lead_hours")

        if model_data.empty:
            continue

        lead_hours = model_data["lead_hours"].values
        values = model_data["metric_value"].values
        sample_sizes = model_data["sample_size"].values

        color = get_model_color(idx)
        linestyle = get_model_linestyle(idx)

        ax.plot(
            lead_hours,
            values,
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=6,
            linewidth=2,
            label=model,
        )

        # Annotate sample size
        if len(sample_sizes) > 0:
            annotate_sample_size(ax, int(sample_sizes[-1]))

    ax.legend(title="Model", loc="upper left")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if not log_x:
        ax.set_xlim(left=0)

    return fig


def plot_model_bias_comparison(
    df: pd.DataFrame,
    variable: str,
    models: list[str] | None = None,
) -> plt.Figure:
    """Plot bias vs lead time for multiple models.

    Bias shows systematic overprediction (positive) or underprediction
    (negative). This plot reveals which models have systematic biases
    and how those biases change with lead time.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        models: List of model names.

    Returns:
        Matplotlib Figure object.
    """
    data = df[(df["variable"] == variable) & (df["metric_name"] == "bias")].copy()

    if data.empty:
        fig, ax = make_figure(title=f"No bias data for {variable}")
        return fig

    if models:
        data = data[data["model"].isin(models)]

    model_list = sorted(str(m) for m in data["model"].dropna().unique())

    fig, ax = make_figure(
        title=f"Bias by Lead Time — Model Comparison: {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, "Lead time (hours)", f"Bias ({units})")

    for idx, model in enumerate(model_list):
        model_data = data[data["model"] == model].sort_values("lead_hours")

        if model_data.empty:
            continue

        lead_hours = model_data["lead_hours"].values
        values = model_data["metric_value"].values

        color = get_model_color(idx)
        linestyle = get_model_linestyle(idx)

        ax.plot(
            lead_hours,
            values,
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=6,
            linewidth=2,
            label=model,
        )

    # Add zero line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.legend(title="Model", loc="upper left")
    ax.set_xlim(left=0)

    return fig


# ---------------------------------------------------------------------------
# Grouped (side-by-side) plots
# ---------------------------------------------------------------------------


def plot_model_grouped(
    df: pd.DataFrame,
    variable: str,
    metrics: list[str],
    models: list[str] | None = None,
) -> plt.Figure:
    """Plot metrics side-by-side for each model.

    Creates a multi-panel figure with one row per model and one column
    per metric. This is useful when overlaying many metrics on the same
    axes would be cluttered.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metrics: List of metric names.
        models: List of model names.

    Returns:
        Matplotlib Figure object.
    """
    if models:
        df = df[df["model"].isin(models)]

    model_list = sorted(str(m) for m in df["model"].dropna().unique())
    n_models = len(model_list)

    fig, axes = make_multi_panel_figure(
        n_rows=n_models,
        n_cols=len(metrics),
        width=6.0,
        height=4.0,
        sharex=True,
    )

    for row, model in enumerate(model_list):
        model_data = df[df["model"] == model].copy()

        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            mask = (model_data["variable"] == variable) & (model_data["metric_name"] == metric)
            data = model_data[mask].sort_values("lead_hours")

            if data.empty:
                ax.set_title(f"{model}\n{metric.upper()}")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue

            lead_hours = data["lead_hours"].values
            values = data["metric_value"].values
            color = get_metric_color(metric)

            ax.plot(lead_hours, values, color=color, marker="o", markersize=5, linewidth=2)

            units = get_variable_units(variable)
            ax.set_ylabel(f"{metric.upper()} ({units})")
            ax.set_title(f"{model}\n{metric.upper()}")
            ax.grid(True, alpha=0.3, linestyle="-")
            ax.tick_params(axis="both", which="major", labelsize=9)

            if row == n_models - 1:
                ax.set_xlabel("Lead time (hours)")

    fig.suptitle(f"{variable} — Model Comparison (Grouped)", fontsize=14, fontweight="bold", y=1.02)

    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_model_comparison(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    output_dir: str | Path,
    models: list[str] | None = None,
) -> Path:
    """Generate and save a model comparison plot.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metric: Metric name.
        output_dir: Output directory.
        models: Optional list of models.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_model_comparison(df, variable, metric, models=models)

    model_suffix = ""
    if models:
        model_suffix = "_" + "_vs_".join(models)

    filename = f"{metric}_by_lead_time_{variable}{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path


def save_model_bias_comparison(
    df: pd.DataFrame,
    variable: str,
    output_dir: str | Path,
    models: list[str] | None = None,
) -> Path:
    """Generate and save a bias comparison plot.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        output_dir: Output directory.
        models: Optional list of models.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_model_bias_comparison(df, variable, models=models)

    model_suffix = ""
    if models:
        model_suffix = "_" + "_vs_".join(models)

    filename = f"bias_by_lead_time_{variable}{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path
