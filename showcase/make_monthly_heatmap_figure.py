#!/usr/bin/env python3
"""Generate the month-by-lead-time skill heatmap.

Answers a question the lead-time curves cannot: are some months simply harder
to forecast than others, and does that difficulty depend on lead time?

Shows native and interpolated side by side on a shared colour scale, so the
two panels are directly comparable — the point is that interpolation shifts
the whole surface, not one cell.

Usage:
    python showcase/make_monthly_heatmap_figure.py
"""

from __future__ import annotations

import calendar
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "showcase"))

from _data import build_error_table
from _style import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL,
    SURFACE,
    figure_caption,
    panel_title,
)

OUTPUT = REPO_ROOT / "showcase/monthly_lead_skill.png"
VARIABLE = "wind_speed_10m"


def _grid(table, column: str, months: list[int], leads: list[int]) -> np.ndarray:
    """Mean absolute error per (month, lead) cell."""
    matrix = np.full((len(months), len(leads)), np.nan)
    for row, month in enumerate(months):
        for col, lead in enumerate(leads):
            mask = (table["month"] == month) & (table["lead_hours"] == lead)
            values = table.loc[mask, column].abs()
            if len(values):
                matrix[row, col] = values.mean()
    return matrix


def main() -> int:
    try:
        table = build_error_table([VARIABLE])
    except FileNotFoundError:
        print("Missing grid sample; run showcase/make_grid_sample.py first", file=sys.stderr)
        return 1

    months = sorted(table["month"].unique())
    leads = sorted(table["lead_hours"].unique())

    native = _grid(table, f"{VARIABLE}_native_error", months, leads)
    interp = _grid(table, f"{VARIABLE}_interp_error", months, leads)
    vmax = float(np.nanmax([native, interp]))

    fig = plt.figure(figsize=(12.4, 5.4))
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(1, 2, wspace=0.16, left=0.075, right=0.90, top=0.74, bottom=0.16)

    axes = []
    for index, (matrix, title) in enumerate(
        [(native, "Native grid point"), (interp, "Interpolated to observation")]
    ):
        ax = fig.add_subplot(grid[0, index])
        image = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto", zorder=2)
        ax.set_xticks(range(len(leads)), [f"{lead}h" for lead in leads], fontsize=9)
        if index == 0:
            ax.set_yticks(range(len(months)), [calendar.month_abbr[m] for m in months], fontsize=9)
            ax.set_ylabel("Target month", fontsize=9.5, color=INK_SECONDARY)
        else:
            ax.set_yticks(range(len(months)), [""] * len(months))
        ax.set_xlabel("Lead time", fontsize=9.5, color=INK_SECONDARY)

        # Label every cell: the grid is small enough that the numbers are the
        # point, and a colour alone cannot be read precisely.
        for row in range(len(months)):
            for col in range(len(leads)):
                value = matrix[row, col]
                if np.isnan(value):
                    continue
                ax.text(
                    col,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=SURFACE if value > vmax * 0.6 else INK_PRIMARY,
                )

        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_MUTED, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_title(ax, title)
        axes.append(ax)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.030, pad=0.02)
    colorbar.set_label("Wind-speed MAE (km/h)", fontsize=9, color=INK_SECONDARY)
    colorbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    colorbar.outline.set_visible(False)

    worst_month = calendar.month_abbr[months[int(np.nanargmax(np.nanmean(interp, axis=1)))]]
    best_month = calendar.month_abbr[months[int(np.nanargmin(np.nanmean(interp, axis=1)))]]
    overall = 100 * (np.nanmean(native) - np.nanmean(interp)) / np.nanmean(native)

    figure_caption(
        fig,
        "Wind-speed skill by month and lead time — Copenhagen, one year",
        f"Both panels share one colour scale. {worst_month} is the hardest month, "
        f"{best_month} the easiest; interpolation lowers MAE by {overall:.1f}% overall.",
        "Mean absolute error over 23 observation sites, ECMWF IFS 0.25° grid.  "
        "Data: Open-Meteo (CC BY 4.0).\n"
        "Reproduce with: python showcase/make_monthly_heatmap_figure.py",
    )

    fig.savefig(OUTPUT, dpi=160, facecolor=SURFACE)
    print(f"Wrote {OUTPUT}")
    print(f"  native mean {np.nanmean(native):.3f} → interpolated {np.nanmean(interp):.3f} km/h")
    print(f"  hardest month {worst_month}, easiest {best_month}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
