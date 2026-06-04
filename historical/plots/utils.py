"""Shared plotting primitives for forecast verification visualization.

All plotting functions in this package use these utilities for consistency:
- Figure creation with standard sizes
- Axis labeling with units
- Legend placement
- Sample size annotation
- Color conventions

This module is the single source of truth for visual styling.
"""


import logging

import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color conventions
# ---------------------------------------------------------------------------

COLORS = {
    "mae": "#1f77b4",  # Blue
    "rmse": "#ff7f0e",  # Orange
    "bias": "#2ca02c",  # Green
    "std": "#d62728",  # Red
    "error_hist": "#999999",  # Grey
}

# Default color cycle for models (overrides matplotlib defaults)
MODEL_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Grey
]

# Line styles for models
MODEL_LINESTYLES = ["-", "--", "-.", ":"]

# ---------------------------------------------------------------------------
# Attribution for data sources
# Open-Meteo data requires CC BY 4.0 attribution per
# https://open-meteo.com/en/license
ATTRIBUTION_TEXT = "Data: Open-Meteo.com (CC BY 4.0)"
ATTRIBUTION_FULL = (
    "Weather data provided by Open-Meteo.com "
    "under CC BY 4.0 license "
    "(https://open-meteo.com/en/license)"
)

# ---------------------------------------------------------------------------
# Figure creation
# ---------------------------------------------------------------------------


def make_figure(
    width: float = 8.0,
    height: float = 6.0,
    dpi: int = 150,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with standard scientific styling.

    Args:
        width: Figure width in inches.
        height: Figure height in inches.
        dpi: Dots per inch.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Tuple of (figure, axes).
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    # Font settings
    fig.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.15)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    ax.tick_params(axis="both", which="major", labelsize=10)

    # Grid for readability
    ax.grid(True, alpha=0.3, linestyle="-")

    return fig, ax


def make_multi_panel_figure(
    n_rows: int = 2,
    n_cols: int = 2,
    width: float = 8.0,
    height: float = 6.0,
    dpi: int = 150,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a multi-panel figure with standard scientific styling.

    Args:
        n_rows: Number of rows of subplots.
        n_cols: Number of columns of subplots.
        width: Figure width in inches (per panel).
        height: Figure height in inches (per panel).
        dpi: Dots per inch.
        sharex: Share x-axis between panels.
        sharey: Share y-axis between panels.

    Returns:
        Tuple of (figure, axes_array).
    """
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(width * n_cols, height * n_rows),
        dpi=dpi,
        sharex=sharex,
        sharey=sharey,
    )
    axes = np.atleast_2d(axes)

    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.92,
        bottom=0.08,
        hspace=0.3,
        wspace=0.25,
    )

    return fig, axes


# ---------------------------------------------------------------------------
# Axis labeling
# ---------------------------------------------------------------------------


def label_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str = "",
) -> plt.Axes:
    """Apply standard axis labels to an axes object.

    Args:
        ax: Matplotlib axes.
        xlabel: X-axis label with units.
        ylabel: Y-axis label with units.
        title: Optional title.

    Returns:
        ax (for chaining).
    """
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, alpha=0.3, linestyle="-")
    return ax


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------


def add_legend(
    ax: plt.Axes,
    title: str = "",
    loc: str = "upper left",
    fontsize: int = 10,
) -> plt.Axes:
    """Add a legend with standard styling.

    Args:
        ax: Matplotlib axes.
        title: Legend title.
        loc: Legend location.
        fontsize: Legend font size.

    Returns:
        ax (for chaining).
    """
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(
            handles,
            labels,
            title=title,
            loc=loc,
            fontsize=fontsize,
            framealpha=0.9,
            edgecolor="gray",
        )
    return ax


# ---------------------------------------------------------------------------
# Sample size annotation
# ---------------------------------------------------------------------------


def annotate_sample_size(
    ax: plt.Axes,
    n: int,
    total: int = 0,
    fontsize: int = 9,
    loc: str = "upper right",
) -> plt.Axes:
    """Annotate a plot with sample size information.

    Args:
        ax: Matplotlib axes.
        n: Number of observations used.
        total: Total number of observations (including missing).
        fontsize: Annotation font size.
        loc: Text location.

    Returns:
        ax (for chaining).
    """
    if total > 0 and total != n:
        text = f"n = {n} / {total}"
    else:
        text = f"n = {n}"
    ax.annotate(
        text,
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        fontsize=fontsize,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    return ax


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def get_metric_color(metric_name: str) -> str:
    """Get the standard color for a metric.

    Args:
        metric_name: Metric name (e.g. "mae", "rmse", "bias", "std").

    Returns:
        Hex color string.
    """
    return COLORS.get(metric_name.lower(), "#1f77b4")


def get_model_color(model_index: int) -> str:
    """Get the color for a model by index.

    Args:
        model_index: Zero-based index of the model.

    Returns:
        Hex color string.
    """
    return MODEL_COLORS[model_index % len(MODEL_COLORS)]


def get_model_linestyle(model_index: int) -> str:
    """Get the line style for a model by index.

    Args:
        model_index: Zero-based index of the model.

    Returns:
        Line style string.
    """
    return MODEL_LINESTYLES[model_index % len(MODEL_LINESTYLES)]


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

# Common unit mappings for variables
VARIABLE_UNITS: dict[str, str] = {
    "temperature_2m": "°C",
    "temperature_2m_error": "°C",
    "precipitation": "mm/h",
    "precipitation_error": "mm/h",
    "wind_speed": "m/s",
    "wind_speed_error": "m/s",
    "relative_humidity": "%",
    "relative_humidity_error": "%",
    "pressure": "hPa",
    "pressure_error": "hPa",
}


def get_variable_units(variable: str) -> str:
    """Get the standard unit for a variable.

    Args:
        variable: Variable name.

    Returns:
        Unit string, or "units" if unknown.
    """
    return VARIABLE_UNITS.get(variable, VARIABLE_UNITS.get(f"{variable}_error", "units"))


# ---------------------------------------------------------------------------
# Save figure helper
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    output_path: str,
    bbox_inches: str = "tight",
    attribution: str = ATTRIBUTION_TEXT,
    attribution_location: str = "lower right",
) -> None:
    """Save a figure to file with standard settings.

    Args:
        fig: Matplotlib figure.
        output_path: Output file path.
        bbox_inches: Bounding box for saving ("tight" recommended).
        attribution: Attribution text to display on the figure.
        attribution_location: Where to place the attribution annotation.
    """
    # Add attribution as a small text annotation in the corner
    fig.text(
        0.98 if attribution_location == "lower right" else 0.02,
        0.02,
        attribution,
        transform=fig.transFigure,
        fontsize=7,
        ha="right" if attribution_location == "lower right" else "left",
        va="bottom",
        color="gray",
        alpha=0.7,
    )
    fig.savefig(output_path, bbox_inches=bbox_inches, dpi=150)
    plt.close(fig)
    logger.debug("Saved figure: %s", output_path)


# ---------------------------------------------------------------------------
# Close all figures
# ---------------------------------------------------------------------------


def close_all() -> None:
    """Close all open matplotlib figures.

    Call this after generating figures to free memory.
    """
    plt.close("all")
