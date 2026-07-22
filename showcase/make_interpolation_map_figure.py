#!/usr/bin/env python3
"""Generate the spatial view of what grid interpolation does.

The aggregate figure says interpolation lowers error; this one says *where*.
Three panels:

1. The geometry — model grid points against the observation sites they are
   verified at, with a line showing how far each observation sits from the
   grid point that represents it.
2. Where the error is — native MAE across the region, largest over water
   where the nearest grid point is also furthest away.
3. What predicts it — improvement against grid-to-observation distance,
   separating sea-level sites from inland ones.

The coastline comes from Natural Earth, clipped to the region and committed by
showcase/make_coastline.py, so the figure needs no network access at plot time
and no geospatial dependency.

Usage:
    python showcase/make_interpolation_map_figure.py
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
    ACCENT,
    BASELINE,
    DEEMPHASIS,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL,
    SURFACE,
    figure_caption,
    panel_title,
    style_axes,
)

OUTPUT = REPO_ROOT / "showcase/interpolation_map.png"
VARIABLE = "wind_speed_10m"

COPENHAGEN = (55.676, 12.568)
MALMO = (55.605, 13.000)

# Natural Earth 10 m coastline, clipped to the region by
# showcase/make_coastline.py. Land and sea is the reason a grid point can
# misrepresent a site, so the geography belongs on the map.
COASTLINE = REPO_ROOT / "examples/geo/coastline.geojson"


def load_coastline() -> list[np.ndarray]:
    """Coastline segments as [lon, lat] arrays, or [] when unavailable.

    Reads GeoJSON directly rather than pulling in geopandas/cartopy: the only
    geometry needed is LineString and Polygon rings, and a showcase figure
    should not add a heavyweight geospatial dependency.
    """
    if not COASTLINE.exists():
        return []

    import json

    data = json.loads(COASTLINE.read_text())
    segments: list[np.ndarray] = []

    def walk(geometry: dict) -> None:
        kind = geometry.get("type")
        coords = geometry.get("coordinates")
        if kind == "LineString":
            segments.append(np.asarray(coords, dtype=float))
        elif kind in {"MultiLineString", "Polygon"}:
            segments.extend(np.asarray(part, dtype=float) for part in coords)
        elif kind == "MultiPolygon":
            for polygon in coords:
                segments.extend(np.asarray(ring, dtype=float) for ring in polygon)
        elif kind == "GeometryCollection":
            for child in geometry.get("geometries", []):
                walk(child)

    if data.get("type") == "FeatureCollection":
        for feature in data["features"]:
            walk(feature.get("geometry") or {})
    else:
        walk(data)

    return [s for s in segments if s.ndim == 2 and s.shape[0] > 1]


def draw_coastline(ax, segments: list[np.ndarray]) -> None:
    """Draw coastline segments as recessive chrome, behind the data."""
    for segment in segments:
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            color=INK_MUTED,
            linewidth=0.9,
            alpha=0.9,
            zorder=2.5,
            solid_capstyle="round",
        )


def _frame_to_data(ax, sites, grid, margin: float = 0.06):
    """Clamp a map panel to the data extent, and report the bounds used.

    The coastline runs past the study area on purpose, so it reaches the panel
    edges; without this it would stretch the axes and shrink the grid. The
    returned bounds let a filled surface cover exactly the same box, so the
    panel has no empty margin.
    """
    lons = [*sites["site_lon"], *grid["longitude"]]
    lats = [*sites["site_lat"], *grid["latitude"]]
    xlim = (min(lons) - margin, max(lons) + margin)
    ylim = (min(lats) - margin, max(lats) + margin)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return xlim, ylim


def per_site_table():
    """Per-site native/interpolated MAE, distance and elevation."""
    from verification.interpolate import haversine_km

    forecasts, observations = load_grid_sample()
    table = build_error_table([VARIABLE])

    elevation = observations.drop_duplicates(["latitude", "longitude"])[
        ["latitude", "longitude", "elevation"]
    ]
    grid = forecasts[["latitude", "longitude"]].drop_duplicates()

    sites = (
        table.groupby(["site_lat", "site_lon"])
        .agg(
            native=(f"{VARIABLE}_native_error", lambda s: s.abs().mean()),
            interpolated=(f"{VARIABLE}_interp_error", lambda s: s.abs().mean()),
            distance_km=("site_distance_km", "first"),
        )
        .reset_index()
        .merge(elevation, left_on=["site_lat", "site_lon"], right_on=["latitude", "longitude"])
    )
    sites["improvement"] = sites["native"] - sites["interpolated"]
    sites["at_sea_level"] = sites["elevation"] <= 0

    # The grid point each site is verified against natively.
    nearest = [
        min(
            grid.itertuples(),
            key=lambda r, la=row.site_lat, lo=row.site_lon: haversine_km(
                la, lo, r.latitude, r.longitude
            ),
        )
        for row in sites.itertuples()
    ]
    sites["grid_lat"] = [n.latitude for n in nearest]
    sites["grid_lon"] = [n.longitude for n in nearest]
    return sites, grid


def main() -> int:
    try:
        sites, grid = per_site_table()
    except FileNotFoundError:
        print("Missing grid sample; run showcase/make_grid_sample.py first", file=sys.stderr)
        return 1

    coastline = load_coastline()

    fig = plt.figure(figsize=(13.6, 5.3))
    fig.patch.set_facecolor(SURFACE)
    layout = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.15, 1.0],
        wspace=0.34,
        left=0.055,
        right=0.955,
        top=0.72,
        bottom=0.27,
    )

    # ── 1. Geometry: grid points, sites, and the gap between them ────────
    ax = fig.add_subplot(layout[0, 0])
    draw_coastline(ax, coastline)
    for row in sites.itertuples():
        ax.plot(
            [row.site_lon, row.grid_lon],
            [row.site_lat, row.grid_lat],
            color=BASELINE,
            linewidth=1,
            zorder=2,
        )
    ax.scatter(
        grid["longitude"],
        grid["latitude"],
        s=46,
        marker="s",
        color=DEEMPHASIS,
        edgecolors=SURFACE,
        linewidths=1.5,
        zorder=3,
        label="Model grid point",
    )
    ax.scatter(
        sites["site_lon"],
        sites["site_lat"],
        s=40,
        color=ACCENT,
        edgecolors=SURFACE,
        linewidths=1.5,
        zorder=4,
        label="Observation site",
    )
    style_axes(ax)
    _frame_to_data(ax, sites, grid)
    panel_title(ax, f"Forecast and site sit {sites['distance_km'].median():.1f} km apart (median)")
    ax.set_xlabel("Longitude (°E)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("Latitude (°N)", fontsize=9.5, color=INK_SECONDARY)
    legend = ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncols=2,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    # ── 2. Native error across the region ────────────────────────────────
    # Magnitude, not the native-minus-interpolated difference. That difference
    # is within ±0.1 km/h at 15 of 23 sites, so mapping it paints mostly noise
    # and lets two outliers dictate the whole surface. Native MAE is a real
    # field: it is 3.82 km/h at sea-level sites against 2.71 inland, which is
    # the pattern worth showing — the model does worst over water, where its
    # nearest grid point is also furthest away.
    ax2 = fig.add_subplot(layout[0, 1])
    ax2_bounds = _frame_to_data(ax2, sites, grid)

    # Continuous surface between the sites, as the earlier composites did.
    # This is display smoothing only — the measurements are the 23 site
    # markers drawn on top; everything between them is interpolated for
    # readability and carries no independent evidence.
    #
    # Cubic rather than linear: linear over scattered points produces flat
    # triangular facets with visible straight edges.
    field_lon = np.linspace(*ax2_bounds[0], 240)
    field_lat = np.linspace(*ax2_bounds[1], 240)
    mesh_lon, mesh_lat = np.meshgrid(field_lon, field_lat)
    points = (sites["site_lon"].to_numpy(), sites["site_lat"].to_numpy())
    values = sites["native"].to_numpy()

    # Anchor the surface at the panel corners with the field mean so the whole
    # box lies inside the convex hull. Without this, cubic returns NaN outside
    # the hull and any backfill leaves a visible seam where the two meet.
    (x0, x1), (y0, y1) = ax2_bounds
    anchor_lon = np.array([x0, x1, x0, x1, (x0 + x1) / 2, (x0 + x1) / 2, x0, x1])
    anchor_lat = np.array([y0, y0, y1, y1, y0, y1, (y0 + y1) / 2, (y0 + y1) / 2])
    padded_lon = np.concatenate([points[0], anchor_lon])
    padded_lat = np.concatenate([points[1], anchor_lat])
    padded_values = np.concatenate([values, np.full(anchor_lon.size, values.mean())])

    field = griddata((padded_lon, padded_lat), padded_values, (mesh_lon, mesh_lat), method="cubic")
    # Cubic can overshoot past the observed range between sparse points; clamp
    # it so the surface never implies a value no site recorded.
    field = np.clip(field, values.min(), values.max())

    levels = np.linspace(values.min(), values.max(), 25)
    ax2.contourf(mesh_lon, mesh_lat, field, levels=levels, cmap=SEQUENTIAL, extend="both", zorder=1)
    ax2.contour(
        mesh_lon,
        mesh_lat,
        field,
        levels=levels[::4],
        colors=[INK_PRIMARY],
        linewidths=0.4,
        alpha=0.28,
        zorder=2,
    )
    draw_coastline(ax2, coastline)
    scatter = ax2.scatter(
        sites["site_lon"],
        sites["site_lat"],
        c=sites["native"],
        cmap=SEQUENTIAL,
        vmin=values.min(),
        vmax=values.max(),
        s=72,
        edgecolors=INK_PRIMARY,
        linewidths=0.8,
        zorder=4,
    )
    for label, (lat, lon) in {"Copenhagen": COPENHAGEN, "Malmö": MALMO}.items():
        ax2.plot(lon, lat, marker="*", markersize=13, color=INK_PRIMARY, zorder=4)
        ax2.annotate(
            label,
            (lon, lat),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    style_axes(ax2)
    _frame_to_data(ax2, sites, grid)
    ax2.grid(False)
    panel_title(ax2, "The distance is large for coastal areas")
    ax2.set_xlabel("Longitude (°E)", fontsize=9.5, color=INK_SECONDARY)
    ax2.set_ylabel("Latitude (°N)", fontsize=9.5, color=INK_SECONDARY)
    colorbar = fig.colorbar(scatter, ax=ax2, fraction=0.043, pad=0.03)
    colorbar.set_label("Native wind-speed MAE (km/h)", fontsize=8.5, color=INK_SECONDARY)
    colorbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    colorbar.outline.set_visible(False)

    # ── 3. Distance is what predicts it ──────────────────────────────────
    ax3 = fig.add_subplot(layout[0, 2])
    ax3.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    for at_sea_level, marker, label in [
        (True, "o", "At sea level"),
        (False, "^", "Inland"),
    ]:
        subset = sites[sites["at_sea_level"] == at_sea_level]
        ax3.scatter(
            subset["distance_km"],
            subset["improvement"],
            s=62,
            marker=marker,
            color=ACCENT if at_sea_level else DEEMPHASIS,
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=3,
            label=label,
        )
    style_axes(ax3)
    panel_title(ax3, "Error vs. fore/obs distance.")
    ax3.set_xlabel("Dist. Grid point to observation site (km)", fontsize=9.5, color=INK_SECONDARY)
    ax3.set_ylabel("MAE reduction (km/h)", fontsize=9.5, color=INK_SECONDARY)
    legend3 = ax3.legend(frameon=False, fontsize=8.5, loc="upper left")
    for text in legend3.get_texts():
        text.set_color(INK_SECONDARY)

    distance_correlation = sites[["distance_km", "improvement"]].corr().iloc[0, 1]
    elevation_correlation = sites[["elevation", "improvement"]].corr().iloc[0, 1]
    sea_level_gain = sites.loc[sites["at_sea_level"], "improvement"].mean()
    inland_gain = sites.loc[~sites["at_sea_level"], "improvement"].mean()
    improved = int((sites["improvement"] > 0).sum())

    figure_caption(
        fig,
        "Effect of grid interpolation — Copenhagen, one year",
        f"{len(sites)} observation sites against a 24-point ECMWF IFS 0.25° grid. "
        f"Wind speed forecast improves at {improved} of {len(sites)} sites.",
        f"Sea-level sites gain more on average ({sea_level_gain:.2f} vs {inland_gain:.2f} km/h), but "
        f"that follows from distance (r={distance_correlation:.2f}) rather than the coast itself "
        f"(elevation r={elevation_correlation:.2f}) —\nthose sites simply sit further from their grid "
        "point.  Data: Open-Meteo (CC BY 4.0). Reproduce with: "
        "python showcase/make_interpolation_map_figure.py",
    )

    fig.savefig(OUTPUT, dpi=160, facecolor=SURFACE)
    print(f"Wrote {OUTPUT}")
    print(f"  improved at {improved}/{len(sites)} sites")
    print(f"  sea level {sea_level_gain:+.3f} km/h vs inland {inland_gain:+.3f} km/h")
    print(f"  r(distance)={distance_correlation:.3f}  r(elevation)={elevation_correlation:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
