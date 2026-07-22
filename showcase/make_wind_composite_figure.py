#!/usr/bin/env python3
"""Generate the wind verification composite.

A 3x3 view of the same interpolated verification, one question per row:

  row 1  how error grows with lead time, per variable (MAE, RMSE, bias)
  row 2  where wind-speed error sits, at three lead bands
  row 3  how variable that error is across lead time, per variable

Row 2 shows magnitude on a shared colour scale so the three bands read as a
progression. Row 3 shows the coefficient of variation of each site's MAE
across lead times: low means a site is consistently hard, high means its
difficulty depends on how far ahead the forecast was made.

Usage:
    python showcase/make_wind_composite_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "showcase"))

from _data import build_error_table, load_grid_sample
from _style import (
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL,
    SERIES,
    SURFACE,
    panel_title,
    style_axes,
)
from make_interpolation_map_figure import (
    COPENHAGEN,
    MALMO,
    draw_coastline,
    load_coastline,
)

OUTPUT = REPO_ROOT / "showcase/wind_composite.png"

VARIABLES = [
    ("wind_speed_10m", "Wind speed", "km/h", SERIES[0]),
    ("wind_gusts_10m", "Wind gusts", "km/h", SERIES[2]),
    ("wind_direction_10m", "Wind direction", "°", SERIES[4]),
]
MAP_VARIABLE = "wind_speed_10m"

# The bundled sample runs to 47 h, so these are the honest bands — not the
# 0-1 / 3-4 / 6-7 day split a week-long run would allow.
LEAD_BANDS = [(0, 12, "0–12 h"), (12, 36, "12–36 h"), (36, 48, "36–47 h")]

GRID_RESOLUTION = 200


def lead_time_series(table, variable: str) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """MAE, RMSE and bias per lead time for one variable."""
    column = f"{variable}_interp_error"
    leads = sorted(table["lead_hours"].unique())
    mae, rmse, bias = [], [], []
    for lead in leads:
        errors = table.loc[table.lead_hours == lead, column].dropna().to_numpy()
        mae.append(np.abs(errors).mean())
        rmse.append(np.sqrt((errors**2).mean()))
        bias.append(errors.mean())
    return leads, np.array(mae), np.array(rmse), np.array(bias)


def site_field(sites_lon, sites_lat, values, bounds):
    """Smooth surface through scattered site values, filling the panel."""
    (x0, x1), (y0, y1) = bounds
    mesh_lon, mesh_lat = np.meshgrid(
        np.linspace(x0, x1, GRID_RESOLUTION), np.linspace(y0, y1, GRID_RESOLUTION)
    )
    # Anchor the corners at the field mean so the whole box lies inside the
    # convex hull; cubic returns NaN outside it and any backfill leaves a seam.
    anchor_lon = np.array([x0, x1, x0, x1, (x0 + x1) / 2, (x0 + x1) / 2, x0, x1])
    anchor_lat = np.array([y0, y0, y1, y1, y0, y1, (y0 + y1) / 2, (y0 + y1) / 2])
    padded = (
        np.concatenate([sites_lon, anchor_lon]),
        np.concatenate([sites_lat, anchor_lat]),
    )
    padded_values = np.concatenate([values, np.full(anchor_lon.size, values.mean())])

    field = griddata(padded, padded_values, (mesh_lon, mesh_lat), method="cubic")
    # Cubic overshoots between sparse points; never imply an unobserved value.
    return mesh_lon, mesh_lat, np.clip(field, values.min(), values.max())


def draw_map_panel(ax, bounds, mesh, field, sites, values, cmap, vmin, vmax, coastline):
    """One map panel: smooth surface, coastline, city markers, site dots."""
    mesh_lon, mesh_lat = mesh
    levels = np.linspace(vmin, vmax, 21)
    ax.contourf(mesh_lon, mesh_lat, field, levels=levels, cmap=cmap, extend="both", zorder=1)
    ax.contour(
        mesh_lon,
        mesh_lat,
        field,
        levels=levels[::4],
        colors=[INK_PRIMARY],
        linewidths=0.4,
        alpha=0.25,
        zorder=2,
    )
    draw_coastline(ax, coastline)
    image = ax.scatter(
        sites["site_lon"],
        sites["site_lat"],
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=54,
        edgecolors=INK_PRIMARY,
        linewidths=0.7,
        zorder=4,
    )
    for label, (lat, lon) in {"Copenhagen": COPENHAGEN, "Malmö": MALMO}.items():
        ax.plot(lon, lat, marker="*", markersize=11, color=INK_PRIMARY, zorder=5)
        ax.annotate(
            label,
            (lon, lat),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK_SECONDARY,
            zorder=5,
        )
    style_axes(ax)
    ax.grid(False)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_xlabel("Longitude (°E)", fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel("Latitude (°N)", fontsize=8.5, color=INK_SECONDARY)
    return image


def _caption(fig, title: str, subtitle: str, footnote: str) -> None:
    """Title block for a tall figure.

    _style.figure_caption places the subtitle for a single-row figure; at three
    rows that lands on top of the first panel, so the offsets are set here.
    """
    fig.suptitle(
        title,
        fontsize=14,
        color=INK_PRIMARY,
        fontweight="600",
        x=0.006,
        ha="left",
        y=0.975,
    )
    fig.text(0.006, 0.947, subtitle, fontsize=9.5, color=INK_SECONDARY, ha="left")
    fig.text(0.006, 0.012, footnote, fontsize=8.5, color=INK_MUTED, ha="left")


def main() -> int:
    names = [name for name, _, _, _ in VARIABLES]
    try:
        table = build_error_table(names)
    except FileNotFoundError:
        print("Missing grid sample; run showcase/make_grid_sample.py first", file=sys.stderr)
        return 1

    forecasts, _ = load_grid_sample()
    grid = forecasts[["latitude", "longitude"]].drop_duplicates()
    coastline = load_coastline()

    sites = table[["site_lat", "site_lon"]].drop_duplicates().reset_index(drop=True)
    margin = 0.06
    bounds = (
        (
            min(sites.site_lon.min(), grid.longitude.min()) - margin,
            max(sites.site_lon.max(), grid.longitude.max()) + margin,
        ),
        (
            min(sites.site_lat.min(), grid.latitude.min()) - margin,
            max(sites.site_lat.max(), grid.latitude.max()) + margin,
        ),
    )

    fig = plt.figure(figsize=(13.4, 13.6))
    fig.patch.set_facecolor(SURFACE)
    layout = fig.add_gridspec(
        3, 3, hspace=0.34, wspace=0.28, left=0.065, right=0.885, top=0.855, bottom=0.095
    )

    # ── Row 1: error against lead time ───────────────────────────────────
    for column, (variable, title, unit, colour) in enumerate(VARIABLES):
        ax = fig.add_subplot(layout[0, column])
        leads, mae, rmse, bias = lead_time_series(table, variable)

        ax.plot(
            leads,
            rmse,
            color=colour,
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            label="RMSE",
            zorder=3,
        )
        ax.plot(
            leads,
            mae,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            label="MAE",
            zorder=4,
        )
        ax.plot(
            leads,
            bias,
            color=INK_MUTED,
            linewidth=1.5,
            linestyle=":",
            marker="^",
            markersize=5,
            label="Bias",
            zorder=3,
        )
        ax.axhline(0, color=GRIDLINE, linewidth=1, zorder=1)

        style_axes(ax)
        panel_title(ax, title)
        ax.set_xlabel("Lead time (hours)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel(f"Error ({unit})", fontsize=9, color=INK_SECONDARY)
        legend = ax.legend(frameon=False, fontsize=8, loc="upper left")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    # ── Row 2: wind-speed error by lead band, shared scale ───────────────
    band_values = []
    for low, high, _ in LEAD_BANDS:
        window = table[(table.lead_hours >= low) & (table.lead_hours < high)]
        per_site = window.groupby(["site_lat", "site_lon"])[f"{MAP_VARIABLE}_interp_error"].apply(
            lambda s: s.abs().mean()
        )
        band_values.append(
            per_site.reindex(list(zip(sites.site_lat, sites.site_lon, strict=True))).to_numpy()
        )

    vmin = float(min(v.min() for v in band_values))
    vmax = float(max(v.max() for v in band_values))

    map_image = None
    for column, ((_, _, label), values) in enumerate(zip(LEAD_BANDS, band_values, strict=True)):
        ax = fig.add_subplot(layout[1, column])
        mesh_lon, mesh_lat, field = site_field(
            sites.site_lon.to_numpy(), sites.site_lat.to_numpy(), values, bounds
        )
        map_image = draw_map_panel(
            ax,
            bounds,
            (mesh_lon, mesh_lat),
            field,
            sites,
            values,
            SEQUENTIAL,
            vmin,
            vmax,
            coastline,
        )
        panel_title(ax, label)

    row2_axes = fig.axes[3:6]
    box = row2_axes[-1].get_position()
    cax = fig.add_axes([0.898, box.y0, 0.013, box.height])
    colorbar = fig.colorbar(map_image, cax=cax)
    colorbar.set_label("Wind-speed MAE (km/h)", fontsize=8.5, color=INK_SECONDARY)
    colorbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    colorbar.outline.set_visible(False)

    # ── Row 3: how much each site's error depends on lead time ───────────
    cv_values = []
    for variable, _, _, _ in VARIABLES:
        per = table.groupby(["site_lat", "site_lon", "lead_hours"])[
            f"{variable}_interp_error"
        ].apply(lambda s: s.abs().mean())
        cv = per.groupby(level=[0, 1]).agg(lambda s: s.std() / s.mean())
        cv_values.append(
            cv.reindex(list(zip(sites.site_lat, sites.site_lon, strict=True))).to_numpy()
        )

    cv_min = float(min(v.min() for v in cv_values))
    cv_max = float(max(v.max() for v in cv_values))

    cv_image = None
    for column, ((_, title, _, _), values) in enumerate(zip(VARIABLES, cv_values, strict=True)):
        ax = fig.add_subplot(layout[2, column])
        mesh_lon, mesh_lat, field = site_field(
            sites.site_lon.to_numpy(), sites.site_lat.to_numpy(), values, bounds
        )
        cv_image = draw_map_panel(
            ax,
            bounds,
            (mesh_lon, mesh_lat),
            field,
            sites,
            values,
            "PuOr_r",
            cv_min,
            cv_max,
            coastline,
        )
        panel_title(ax, f"{title} — spread across lead time")

    row3_box = fig.axes[7].get_position()
    cv_cax = fig.add_axes([0.898, row3_box.y0, 0.013, row3_box.height])
    cv_bar = fig.colorbar(cv_image, cax=cv_cax)
    cv_bar.set_label("CV of MAE across lead time (σ/μ)", fontsize=8.5, color=INK_SECONDARY)
    cv_bar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    cv_bar.outline.set_visible(False)

    leads = sorted(table["lead_hours"].unique())
    _caption(
        fig,
        "Wind forecast verification — interpolated to the observation grid",
        f"{len(sites)} observation sites, {len(table):,} verification pairs, "
        f"lead times {min(leads)}–{max(leads)} h. ECMWF IFS 0.25° grid, Copenhagen, one year.",
        "Forecasts are interpolated onto each observation location before any error is computed. "
        "Row 2 shares one colour scale so the three lead bands are comparable; row 3 is the "
        "coefficient of variation of each site's MAE\nacross lead times — low means consistently "
        "hard, high means difficulty depends on lead time.  Data: Open-Meteo (CC BY 4.0). "
        "Reproduce with: python showcase/make_wind_composite_figure.py",
    )

    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)
    print(f"Wrote {OUTPUT}")
    for variable, title, unit, _ in VARIABLES:
        _, mae, rmse, bias = lead_time_series(table, variable)
        print(
            f"  {title:<16} MAE {mae[0]:6.2f} → {mae[-1]:6.2f} {unit:<5} "
            f"RMSE {rmse[0]:6.2f} → {rmse[-1]:6.2f}  bias {bias[0]:+.2f} → {bias[-1]:+.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
