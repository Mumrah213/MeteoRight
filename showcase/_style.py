"""Shared visual tokens for the showcase figures.

One palette across every figure so the folder reads as a single system. The
values are the validated light-mode set: adjacent-pair CVD separation checked
against the #fcfcfb surface, with the de-emphasis grey reserved for context
marks that must recede behind an accent.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"
DEEMPHASIS = "#898781"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots, in fixed order. Never cycle past the end — fold the tail
# into "Other" or facet instead.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# Sequential ramp (one hue, light to dark) for magnitude heatmaps.
SEQUENTIAL = "Blues"

# Diverging ramp for signed quantities, neutral grey at the midpoint.
DIVERGING = "BrBG"


def style_axes(ax, *, grid_axis: str = "both") -> None:
    """Apply the recessive chrome every showcase axes shares."""
    ax.set_facecolor(SURFACE)
    if grid_axis != "none":
        ax.grid(axis=grid_axis, color=GRIDLINE, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def panel_title(ax, text: str, *, pad: int = 10) -> None:
    """Left-aligned panel heading."""
    ax.set_title(text, fontsize=11, color=INK_PRIMARY, fontweight="600", loc="left", pad=pad)


def figure_caption(fig, title: str, subtitle: str, footnote: str) -> None:
    """Figure-level title block and footnote, aligned to the left margin."""
    fig.suptitle(
        title,
        fontsize=13.5,
        color=INK_PRIMARY,
        fontweight="600",
        x=0.006,
        ha="left",
        y=0.965,
    )
    fig.text(0.006, 0.885, subtitle, fontsize=9.5, color=INK_SECONDARY, ha="left")
    fig.text(0.006, 0.035, footnote, fontsize=8.5, color=INK_MUTED, ha="left")
