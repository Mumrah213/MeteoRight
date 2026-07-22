#!/usr/bin/env python3
"""Generate the wind-error structure showcase figure.

Three views of the same error field, each answering a different question:

- Distribution: what does a typical error look like, and is it biased?
- Percentiles by lead time: does the tail grow faster than the median?
- Cross-variable correlation: do the same sites fail across variables?

Replaces the separate wind_error_distributions / _quantiles /
_percentile_heatmap figures, which showed overlapping cuts of one dataset.

Usage:
    python showcase/make_error_structure_figure.py
"""

from __future__ import annotations

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
    ACCENT,
    DEEMPHASIS,
    DIVERGING,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    figure_caption,
    panel_title,
    style_axes,
)

OUTPUT = REPO_ROOT / "showcase/wind_error_structure.png"

VARIABLES = ["wind_speed_10m", "wind_gusts_10m", "temperature_2m", "wind_direction_10m"]
VARIABLE_LABELS = {
    "wind_speed_10m": "Wind speed",
    "wind_gusts_10m": "Wind gusts",
    "temperature_2m": "Temperature",
    "wind_direction_10m": "Wind direction",
}
PERCENTILES = [50, 75, 90, 95]


def main() -> int:
    try:
        table = build_error_table(VARIABLES)
    except FileNotFoundError:
        print("Missing grid sample; run showcase/make_grid_sample.py first", file=sys.stderr)
        return 1

    fig = plt.figure(figsize=(13.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.12],
        wspace=0.34,
        left=0.055,
        right=0.955,
        top=0.72,
        bottom=0.21,
    )

    # ── 1. Error distribution, native vs interpolated ────────────────────
    ax = fig.add_subplot(grid[0, 0])
    native = table["wind_speed_10m_native_error"].dropna()
    interp = table["wind_speed_10m_interp_error"].dropna()
    bins = np.linspace(-12, 12, 49)

    ax.hist(native, bins=bins, color=DEEMPHASIS, alpha=0.85, label="Native grid", zorder=2)
    ax.hist(
        interp,
        bins=bins,
        histtype="step",
        color=ACCENT,
        linewidth=2,
        label="Interpolated",
        zorder=3,
    )
    ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
    style_axes(ax, grid_axis="y")
    panel_title(ax, "Wind-speed error distribution")
    ax.set_xlabel("Forecast − observed (km/h)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("Observations", fontsize=9.5, color=INK_SECONDARY)
    legend = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    ax.text(
        0.03,
        0.72,
        f"bias {native.mean():+.2f} → {interp.mean():+.2f} km/h",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK_MUTED,
    )

    # ── 2. Error percentiles by lead time ────────────────────────────────
    ax2 = fig.add_subplot(grid[0, 1])
    leads = sorted(table["lead_hours"].unique())
    absolute = table.assign(abs_error=table["wind_speed_10m_interp_error"].abs())

    for index, percentile in enumerate(PERCENTILES):
        values = [
            absolute.loc[absolute.lead_hours == lead, "abs_error"].quantile(percentile / 100)
            for lead in leads
        ]
        shade = 0.25 + 0.75 * index / (len(PERCENTILES) - 1)
        ax2.plot(
            leads,
            values,
            linewidth=2,
            marker="o",
            markersize=5,
            color=ACCENT,
            alpha=shade,
            zorder=3,
        )
        ax2.annotate(
            f"P{percentile}",
            (leads[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_SECONDARY,
            va="center",
        )

    style_axes(ax2)
    panel_title(ax2, "The tail grows faster than the median")
    ax2.set_xlabel("Lead time (hours)", fontsize=9.5, color=INK_SECONDARY)
    ax2.set_ylabel("Absolute wind-speed error (km/h)", fontsize=9.5, color=INK_SECONDARY)
    ax2.set_xlim(min(leads) - 2, max(leads) + 8)

    # ── 3. Cross-variable error correlation across sites ─────────────────
    ax3 = fig.add_subplot(grid[0, 2])
    per_site = table.groupby(["site_lat", "site_lon"]).apply(
        lambda g: {v: g[f"{v}_interp_error"].abs().mean() for v in VARIABLES},
        include_groups=False,
    )
    site_errors = np.array([[row[v] for v in VARIABLES] for row in per_site])
    correlation = np.corrcoef(site_errors, rowvar=False)

    image = ax3.imshow(correlation, cmap=DIVERGING, vmin=-1, vmax=1, zorder=2)
    labels = [VARIABLE_LABELS[v] for v in VARIABLES]
    ax3.set_xticks(range(len(labels)), labels, fontsize=8.5, rotation=30, ha="right")
    ax3.set_yticks(range(len(labels)), labels, fontsize=8.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = correlation[i, j]
            ax3.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=INK_PRIMARY if abs(value) < 0.6 else SURFACE,
                fontweight="600",
            )
    ax3.set_facecolor(SURFACE)
    ax3.tick_params(colors=INK_MUTED, length=0)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    panel_title(ax3, "Shared difficulty across variables")
    colorbar = fig.colorbar(image, ax=ax3, fraction=0.040, pad=0.06)
    colorbar.set_label("Correlation of per-site MAE", fontsize=8.5, color=INK_SECONDARY)
    colorbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    colorbar.outline.set_visible(False)

    sites = len(per_site)
    figure_caption(
        fig,
        "Wind error structure after interpolation — Copenhagen, one year",
        f"{len(table):,} verification pairs across {sites} observation sites and "
        f"{len(leads)} lead times, ECMWF IFS 0.25° grid.",
        "Distribution and percentiles use interpolated forecasts; the correlation panel asks whether "
        "a site that is hard for wind is also hard for temperature.\n"
        "Data: Open-Meteo (CC BY 4.0). Reproduce with: "
        "python showcase/make_error_structure_figure.py",
    )

    fig.savefig(OUTPUT, dpi=160, facecolor=SURFACE)
    print(f"Wrote {OUTPUT}")
    print(f"  native bias {native.mean():+.3f} → interpolated {interp.mean():+.3f} km/h")
    print(f"  sites {sites}, pairs {len(table):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
