"""Lead-time degradation curves.

The canonical weather verification visualization: how forecast accuracy
changes with lead time.

Produces:
- MAE vs lead_hours
- RMSE vs lead_hours
- Bias vs lead_hours

Supports:
- Single or multiple variables
- Single or multiple models
- Automatic color assignment

Usage
-----
>>> from weather_analysis.plots.lead_time import plot_lead_time_degradation
>>> fig = plot_lead_time_degradation(df, "temperature_2m", "mae")
>>> fig.savefig("outputs/figures/mae_by_lead_time_temperature_2m.png")
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import (
    annotate_sample_size,
    get_metric_color,
    get_model_color,
    get_model_linestyle,
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
# Plotting functions
# ---------------------------------------------------------------------------


def plot_lead_time_degradation(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    models: list[str] | None = None,
    include_std: bool = False,
    log_x: bool = False,
    log_y: bool = False,
) -> plt.Figure | None:
    """Plot MAE/RMSE/Bias vs lead time for a single variable.

    This is the canonical forecast verification plot. It shows how forecast
    accuracy degrades as lead time increases.

    Args:
        df: Metrics DataFrame (long format from Phase 3).
            Must contain columns: lead_hours, variable, metric_name,
            metric_value, sample_size.
        variable: Variable name (e.g. "temperature_2m").
        metric: Metric name (e.g. "mae", "rmse", "bias").
        models: List of model names to plot. If None, all models are plotted.
        include_std: If True, also plot error standard deviation.
        log_x: If True, use logarithmic x-axis.
        log_y: If True, use logarithmic y-axis.

    Returns:
        Matplotlib Figure object, or None if data has no meaningful variation.
    """
    # Filter to the target variable and metric
    mask = (df["variable"] == variable) & (df["metric_name"] == metric)
    if include_std:
        mask = mask | ((df["variable"] == variable) & (df["metric_name"] == "std"))
    data = df[mask].copy()

    if data.empty:
        logger.info("No data for variable='%s', metric='%s'. Skipping.", variable, metric)
        return None

    # Check for meaningful variation
    if not _has_meaningful_variation(data["metric_value"].values):
        logger.info(
            "Skipping %s/%s lead_time plot: no meaningful variation (all zeros or constant)",
            variable,
            metric,
        )
        return None

    # Filter to specific models if requested
    if models:
        data = data[data["model"].isin(models)]

    # Get unique models
    model_list = (
        sorted(str(m) for m in data["model"].dropna().unique())
        if "model" in data.columns
        else ["_all_"]
    )

    fig, ax = make_figure(title=f"{metric.upper()} by Lead Time — {variable}")

    units = get_variable_units(variable)
    label_axes(ax, "Lead time (hours)", f"{metric.upper()} ({units})")

    for idx, model in enumerate(model_list):
        if model == "_all_" or "model" not in data.columns:
            model_data = data.sort_values("lead_hours")
        else:
            model_data = data[data["model"] == model].sort_values("lead_hours")

        if model_data.empty:
            continue

        lead_hours = model_data["lead_hours"].values
        values = model_data["metric_value"].values
        sample_sizes = model_data["sample_size"].values

        color = get_model_color(idx)
        linestyle = "-" if len(model_list) == 1 else get_model_linestyle(idx)

        ax.plot(
            lead_hours,
            values,
            color=color,
            linestyle=linestyle,
            marker="o",
            markersize=5,
            linewidth=2,
            label=model if len(model_list) > 1 else None,
        )

        # Annotate sample size for the last point
        if len(lead_hours) > 0:
            annotate_sample_size(ax, int(sample_sizes[-1]))

    if len(model_list) > 1:
        ax.legend(title="Model", loc="upper left")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # Set x-axis to start at 0 if not log scale
    if not log_x:
        ax.set_xlim(left=0)

    return fig


def plot_lead_time_all_metrics(
    df: pd.DataFrame,
    variable: str,
    models: list[str] | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> plt.Figure | None:
    """Plot MAE, RMSE, and Bias vs lead time on the same axes.

    Useful for comparing error magnitude and systematic bias.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        models: List of model names to plot. If None, all models.
        log_x: Logarithmic x-axis.
        log_y: Logarithmic y-axis.

    Returns:
        Matplotlib Figure object, or None if data has no meaningful variation.
    """
    metrics = ["mae", "rmse", "bias"]

    if models:
        df = df[df["model"].isin(models)]

    # Check if any metric has meaningful variation
    has_data = False
    for metric in metrics:
        mask = (df["variable"] == variable) & (df["metric_name"] == metric)
        data = df[mask]
        if not data.empty and _has_meaningful_variation(data["metric_value"].values):
            has_data = True
            break

    if not has_data:
        logger.info("Skipping %s lead_time_all_metrics: no meaningful variation in data", variable)
        return None

    fig, ax = make_figure(
        title=f"Error Metrics by Lead Time — {variable}",
    )

    units = get_variable_units(variable)
    label_axes(ax, "Lead time (hours)", f"Value ({units})")

    for metric in metrics:
        mask = (df["variable"] == variable) & (df["metric_name"] == metric)
        data = df[mask].sort_values("lead_hours")

        if data.empty:
            continue

        lead_hours = data["lead_hours"].values
        values = data["metric_value"].values
        color = get_metric_color(metric)

        ax.plot(
            lead_hours,
            values,
            color=color,
            marker="o",
            markersize=5,
            linewidth=2,
            label=metric.upper(),
        )

    ax.legend(title="Metric", loc="upper left")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if not log_x:
        ax.set_xlim(left=0)

    return fig


def plot_lead_time_multi_variable(
    df: pd.DataFrame,
    variables: list[str],
    metric: str,
    models: list[str] | None = None,
    log_x: bool = False,
) -> plt.Figure:
    """Plot a single metric for multiple variables on the same axes.

    Args:
        df: Metrics DataFrame.
        variables: List of variable names.
        metric: Metric name.
        models: List of model names.
        log_x: Logarithmic x-axis.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = make_figure(title=f"{metric.upper()} by Lead Time — Multi-Variable")

    if models:
        df = df[df["model"].isin(models)]

    for idx, var in enumerate(variables):
        mask = (df["variable"] == var) & (df["metric_name"] == metric)
        data = df[mask].sort_values("lead_hours")

        if data.empty:
            continue

        lead_hours = data["lead_hours"].values
        values = data["metric_value"].values
        color = get_model_color(idx)

        units = get_variable_units(var)
        ax.plot(
            lead_hours,
            values,
            color=color,
            marker="o",
            markersize=5,
            linewidth=2,
            label=f"{var} ({units})",
        )

    ax.legend(title="Variable", loc="upper left")

    if log_x:
        ax.set_xscale("log")
    ax.set_xlim(left=0)

    return fig


# ---------------------------------------------------------------------------
# Save helpers with standardized filenames
# ---------------------------------------------------------------------------


def save_lead_time_degradation(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    output_dir: str | Path,
    models: list[str] | None = None,
    include_std: bool = False,
) -> Path | None:
    """Generate and save a lead-time degradation plot.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metric: Metric name.
        output_dir: Output directory.
        models: Optional list of models.
        include_std: Also plot standard deviation.

    Returns:
        Path to the saved figure, or None if plot was skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_lead_time_degradation(
        df,
        variable,
        metric,
        models=models,
        include_std=include_std,
    )

    if fig is None:
        return None

    # Build filename
    model_suffix = ""
    if models:
        model_suffix = "_" + "_vs_".join(models)

    filename = f"{metric}_by_lead_time_{variable}{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path


def save_lead_time_all_metrics(
    df: pd.DataFrame,
    variable: str,
    output_dir: str | Path,
    models: list[str] | None = None,
) -> Path | None:
    """Generate and save a multi-metric lead-time plot.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        output_dir: Output directory.
        models: Optional list of models.

    Returns:
        Path to the saved figure, or None if plot was skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_lead_time_all_metrics(df, variable, models=models)

    if fig is None:
        return None

    model_suffix = ""
    if models:
        model_suffix = "_" + "_vs_".join(models)

    filename = f"mae_rmse_bias_by_lead_time_{variable}{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path


def save_lead_time_multi_variable(
    df: pd.DataFrame,
    variables: list[str],
    metric: str,
    output_dir: str | Path,
    models: list[str] | None = None,
) -> Path:
    """Generate and save a multi-variable lead-time plot.

    Args:
        df: Metrics DataFrame.
        variables: List of variable names.
        metric: Metric name.
        output_dir: Output directory.
        models: Optional list of models.

    Returns:
        Path to the saved figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_lead_time_multi_variable(df, variables, metric, models=models)

    model_suffix = ""
    if models:
        model_suffix = "_" + "_vs_".join(models)

    var_suffix = "_".join(variables)
    filename = f"{metric}_by_lead_time_{var_suffix}{model_suffix}.png"
    output_path = output_dir / filename

    save_figure(fig, output_path)
    logger.info("Saved: %s", output_path)
    return output_path
