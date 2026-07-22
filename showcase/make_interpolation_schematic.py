#!/usr/bin/env python3
"""Generate the grid-interpolation schematic.

Explains what `--interpolate` actually does, in three steps:

1. Native: the observation is compared with the nearest grid point, which may
   be kilometres away and over a different surface.
2. Bilinear: the four surrounding grid points are weighted by how much of the
   cell each one "owns", giving an estimate at the observation itself.
3. The weights, drawn to scale.

Unlike the other showcase figures this one is a diagram, not a measurement:
the grid is an idealised 0.25° cell and the values are chosen to make the
arithmetic legible. The caption says so.

Usage:
    python showcase/make_interpolation_schematic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "showcase"))

from _style import (
    ACCENT,
    BASELINE,
    DEEMPHASIS,
    GRIDLINE,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL,
    SURFACE,
    figure_caption,
    panel_title,
)

OUTPUT = REPO_ROOT / "showcase/interpolation_schematic.png"

# One idealised grid cell. Corner values are illustrative, chosen so the
# weighted result is not a round number and the arithmetic is visible.
CELL = (0.0, 0.25)
CORNER_VALUES = {(0, 0): 8.0, (1, 0): 12.0, (0, 1): 14.0, (1, 1): 20.0}
SITE = (0.38, 0.30)  # fractional position within the cell (u, v)


def _weights(u: float, v: float) -> dict[tuple[int, int], float]:
    """Bilinear weights for the four corners of the unit cell."""
    return {
        (0, 0): (1 - u) * (1 - v),
        (1, 0): u * (1 - v),
        (0, 1): (1 - u) * v,
        (1, 1): u * v,
    }


def _draw_cell(ax, *, show_site: bool = True) -> None:
    """The 0.25° cell, its four corners, and the observation site."""
    low, high = CELL
    ax.add_patch(
        Rectangle(
            (low, low),
            high - low,
            high - low,
            facecolor=SURFACE,
            edgecolor=BASELINE,
            linewidth=1.2,
            zorder=1,
        )
    )
    for (i, j), value in CORNER_VALUES.items():
        x, y = low + i * (high - low), low + j * (high - low)
        ax.plot(
            x,
            y,
            marker="s",
            markersize=11,
            color=DEEMPHASIS,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=4,
        )
        ax.annotate(
            f"{value:.0f}",
            (x, y),
            xytext=(-15 if i == 0 else 15, -15 if j == 0 else 15),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
            fontweight="600",
        )
    if show_site:
        u, v = SITE
        ax.plot(
            low + u * (high - low),
            low + v * (high - low),
            marker="o",
            markersize=10,
            color=ACCENT,
            markeredgecolor=SURFACE,
            markeredgewidth=1.8,
            zorder=5,
        )


def _style_cell(ax) -> None:
    low, high = CELL
    pad = 0.075
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal")
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> int:
    low, high = CELL
    span = high - low
    u, v = SITE
    site_x, site_y = low + u * span, low + v * span
    weights = _weights(u, v)
    blended = sum(weights[k] * CORNER_VALUES[k] for k in CORNER_VALUES)
    nearest_key = min(
        CORNER_VALUES,
        key=lambda k: np.hypot(low + k[0] * span - site_x, low + k[1] * span - site_y),
    )
    nearest_value = CORNER_VALUES[nearest_key]

    fig = plt.figure(figsize=(13.0, 4.6))
    fig.patch.set_facecolor(SURFACE)
    layout = fig.add_gridspec(1, 3, wspace=0.16, left=0.03, right=0.985, top=0.71, bottom=0.13)

    # ── 1. Native: nearest grid point wins ───────────────────────────────
    ax = fig.add_subplot(layout[0, 0])
    _draw_cell(ax)
    nx, ny = low + nearest_key[0] * span, low + nearest_key[1] * span
    ax.annotate(
        "",
        xy=(nx, ny),
        xytext=(site_x, site_y),
        arrowprops={"arrowstyle": "->", "color": INK_PRIMARY, "linewidth": 1.6},
        zorder=6,
    )
    ax.annotate(
        "nearest",
        ((site_x + nx) / 2, (site_y + ny) / 2),
        xytext=(14, 6),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax.text(
        low + span / 2,
        low - 0.063,
        f"forecast = {nearest_value:.0f}   (the other three corners are ignored)",
        ha="center",
        fontsize=9,
        color=INK_PRIMARY,
    )
    _style_cell(ax)
    panel_title(ax, "Native: snap to the nearest grid point")

    # ── 2. Bilinear: all four corners contribute ─────────────────────────
    ax2 = fig.add_subplot(layout[0, 1])
    _draw_cell(ax2)
    for (i, j), weight in weights.items():
        x, y = low + i * span, low + j * span
        ax2.plot(
            [site_x, x],
            [site_y, y],
            color=ACCENT,
            linewidth=0.8 + 5.0 * weight,
            alpha=0.5,
            zorder=3,
        )
    ax2.axvline(site_x, color=GRIDLINE, linewidth=1, zorder=2)
    ax2.axhline(site_y, color=GRIDLINE, linewidth=1, zorder=2)
    ax2.text(
        low + span / 2,
        low - 0.063,
        f"forecast = {blended:.2f}   (line width = weight)",
        ha="center",
        fontsize=9,
        color=INK_PRIMARY,
    )
    _style_cell(ax2)
    panel_title(ax2, "Bilinear: weight all four by overlap")

    # ── 3. The weights, drawn as the sub-rectangles they are ─────────────
    ax3 = fig.add_subplot(layout[0, 2])
    # Each corner's weight is the area of the diagonally opposite sub-cell.
    patches = {
        (0, 0): (site_x, site_y, high - site_x, high - site_y),
        (1, 0): (low, site_y, site_x - low, high - site_y),
        (0, 1): (site_x, low, high - site_x, site_y - low),
        (1, 1): (low, low, site_x - low, site_y - low),
    }
    cmap = plt.get_cmap(SEQUENTIAL)
    for key, (x, y, width, height) in patches.items():
        weight = weights[key]
        ax3.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=cmap(0.15 + 0.75 * weight),
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=2,
            )
        )
        if width > 0.02 and height > 0.02:
            ax3.text(
                x + width / 2,
                y + height / 2,
                f"{weight:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="600",
                color=SURFACE if weight > 0.45 else INK_PRIMARY,
            )
    _draw_cell(ax3)
    ax3.text(
        low + span / 2,
        low - 0.063,
        "each corner's weight is the area diagonally opposite it",
        ha="center",
        fontsize=9,
        color=INK_PRIMARY,
    )
    _style_cell(ax3)
    panel_title(ax3, "The weights sum to 1")

    figure_caption(
        fig,
        "How grid interpolation places a forecast at the observation",
        "A model grid point (grey squares) rarely coincides with an observation site (blue circle). "
        "Bilinear interpolation estimates the forecast where the observation actually is.",
        "Schematic, not measured: one idealised 0.25° cell with illustrative corner values. "
        "The measured effect is in interpolation_effect.png and interpolation_map.png.\n"
        "Reproduce with: python showcase/make_interpolation_schematic.py",
    )

    fig.savefig(OUTPUT, dpi=160, facecolor=SURFACE)
    print(f"Wrote {OUTPUT}")
    print(f"  nearest = {nearest_value:.1f}, bilinear = {blended:.3f}")
    print(f"  weights: {', '.join(f'{k}={w:.3f}' for k, w in weights.items())}")
    print(f"  sum = {sum(weights.values()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
