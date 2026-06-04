"""Visualization for spatial forecast error across a grid of points.

Creates scatter plots with interpolation contours showing forecast error
(MAE) at each grid point, overlaid on a map. Grid positions are snapped
to the ideal 0.25° ECMWF IFS/ERA5 grid (not the actual API-returned coordinates).

Usage
-----
>>> from forecast.plot_error_grid import plot_error_map
>>> plot_error_map(
...     data_dir="/path/to/grid_runs/data",
...     variable="temperature_2m",
...     metric="mae",
...     output="/tmp/error_map.png",
... )
"""


import math
from pathlib import Path
from typing import Sequence

import numpy as np


def _compute_mae(
    forecasts: list[Path],
    observations: list[Path],
    variable: str,
    lead_hours: int = 0,
) -> float:
    """Compute MAE between forecasts and observations for a single grid point.

    Args:
        forecasts: List of forecast parquet file paths.
        observations: List of observation parquet file paths.
        variable: Variable name (e.g., "temperature_2m").
        lead_hours: Forecast lead hour to use (0 = analysis).

    Returns:
        Mean absolute error.
    """
    import pyarrow.dataset as ds

    fcst_tables = [ds.dataset(str(f), format="parquet").to_table() for f in forecasts]
    obs_tables = [ds.dataset(str(o), format="parquet").to_table() for o in observations]

    fcst = fcst_tables[0].to_pandas() if fcst_tables else None
    obs = obs_tables[0].to_pandas() if obs_tables else None

    if fcst is None or obs is None:
        return np.nan

    fcst_0 = fcst[fcst["lead_hours"] == lead_hours]
    if len(fcst_0) == 0 or len(obs) == 0:
        return np.nan

    fcst_val = fcst_0[variable].values
    obs_val = obs[variable].values

    # Align by length
    min_len = min(len(fcst_val), len(obs_val))
    if min_len == 0:
        return np.nan

    return float(np.mean(np.abs(fcst_val[:min_len] - obs_val[:min_len])))


def _snap_to_025_grid(lat: float, lon: float) -> tuple[float, float]:
    """Snap lat/lon to the nearest 0.25° grid position.

    The ECMWF IFS/ERA5 grid has origin at (-90°, -180°) with 0.25° spacing.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        Tuple of (snapped_lat, snapped_lon).
    """
    snapped_lat = round(lat / 0.25) * 0.25
    snapped_lon = round(lon / 0.25) * 0.25
    # Clamp to valid ranges
    snapped_lat = max(-90.0, min(90.0, snapped_lat))
    snapped_lon = ((snapped_lon + 180.0) % 360.0) - 180.0
    return snapped_lat, snapped_lon


def _load_grid_data(
    data_dir: str | Path,
    variable: str,
    metric: str = "mae",
    lead_hours: int = 0,
) -> list[dict]:
    """Load all grid point data and compute error metrics.

    Grid positions are snapped to the ideal 0.25° ECMWF IFS/ERA5 grid
    (not the actual API-returned coordinates).

    Args:
        data_dir: Root data directory containing grid subdirectories.
        variable: Variable name (e.g., "temperature_2m", "wind_speed_10m").
        metric: Error metric ("mae" currently supported).
        lead_hours: Forecast lead hour to use.

    Returns:
        List of dicts with grid_id, lat, lon, and error values.
    """
    from pathlib import Path as P

    data_dir = P(data_dir)
    grid_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    results = []
    for grid_dir in grid_dirs:
        obs_path = grid_dir / "observations"
        fcst_path = grid_dir / "forecasts"

        if not obs_path.exists() or not fcst_path.exists():
            continue

        # Load observations to get lat/lon
        import pyarrow.dataset as ds

        obs_ds = ds.dataset(str(obs_path), format="parquet")
        obs = obs_ds.to_table().to_pandas()

        lat = float(obs["latitude"].iloc[0])
        lon = float(obs["longitude"].iloc[0])

        # Snap to ideal 0.25° grid
        lat, lon = _snap_to_025_grid(lat, lon)

        # Compute error
        if metric == "mae":
            error = _compute_mae(
                list(fcst_path.glob("**/*.parquet")),
                list(obs_path.glob("**/*.parquet")),
                variable,
                lead_hours,
            )
        else:
            error = np.nan

        results.append({
            "grid_id": grid_dir.name,
            "lat": lat,
            "lon": lon,
            f"error_{metric}": error,
        })

    return results


def plot_error_map(
    data_dir: str | Path,
    variable: str,
    *,
    metrics: Sequence[str] | None = None,
    lead_hours: int = 0,
    output: str | Path | None = None,
    title: str | None = None,
    basemap_source: str = "CartoDB.Voyager",
) -> Path:
    """Plot forecast error across a grid of points with interpolation contours.

    Requires: matplotlib, scipy, pyarrow, contextily (optional for basemap).

    Args:
        data_dir: Root data directory containing grid subdirectories.
        variable: Variable name (e.g., "temperature_2m", "wind_speed_10m").
        metrics: Error metrics to plot. Defaults to ["mae"].
        lead_hours: Forecast lead hour to use (0 = analysis).
        output: Output file path (PNG). If None, returns figure.
        title: Plot title.
        basemap_source: Contextily basemap source string.

    Returns:
        Path to the saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.interpolate import griddata

    if metrics is None:
        metrics = ["mae"]

    n = len(metrics)
    if n == 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        axes = [ax]
    else:
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

    # Load data
    points_data = _load_grid_data(data_dir, variable, metrics[0], lead_hours)
    if not points_data:
        raise ValueError(f"No grid data found in {data_dir}")

    lons = [p["lon"] for p in points_data]
    lats = [p["lat"] for p in points_data]
    lon_min, lon_max = min(lons) - 0.1, max(lons) + 0.1
    lat_min, lat_max = min(lats) - 0.1, max(lats) + 0.1

    xi = np.linspace(lon_min, lon_max, 100)
    yi = np.linspace(lat_min, lat_max, 100)
    Xi, Yi = np.meshgrid(xi, yi)

    for ax, metric in zip(axes, metrics):
        errors = [p.get(f"error_{metric}", np.nan) for p in points_data]

        # Compute min/max for colorbar (ignoring NaN)
        valid_errors = [e for e in errors if not np.isnan(e)]
        if valid_errors:
            vmin, vmax = min(valid_errors), max(valid_errors)
        else:
            vmin, vmax = 0, 1

        # Scatter
        im = ax.scatter(
            lons,
            lats,
            c=errors,
            cmap="RdYlBu_r",
            s=120,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            vmin=vmin,
            vmax=vmax,
        )

        # Interpolation contours
        Zi = griddata(
            [(p["lon"], p["lat"]) for p in points_data],
            errors,
            (Xi, Yi),
            method="cubic",
        )
        cf = ax.contour(Xi, Yi, Zi, levels=8, colors="black", linewidths=0.5, alpha=0.4, zorder=2)
        ax.clabel(cf, inline=True, fontsize=7, fmt="%.2f")

        # Bounding box
        rect = patches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=1.5,
            edgecolor="green",
            facecolor="none",
            linestyle="--",
            alpha=0.5,
            zorder=1,
        )
        ax.add_patch(rect)

        # Add basemap tiles
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs="EPSG:4326", source=basemap_source, alpha=0.5)
        except ImportError:
            pass

        plt.colorbar(im, ax=ax, label=f"{metric.upper()} {variable} ({'°C' if 'temp' in variable else 'm/s'})")
        ax.set_title(f"{metric.upper()} — {variable}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)

    if title is None:
        title = f"{variable} Forecast Error — {len(points_data)} Grid Points"
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output is None:
        return fig

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_error_folium(
    data_dir: str | Path,
    variable: str,
    *,
    metrics: Sequence[str] | None = None,
    lead_hours: int = 0,
    output: str | Path | None = None,
    zoom_start: int = 10,
    tiles: str = "CartoDB.Voyager",
) -> Path:
    """Create an interactive HTML map with forecast error tooltips.

    Requires: folium.

    Args:
        data_dir: Root data directory containing grid subdirectories.
        variable: Variable name.
        metrics: Error metrics to show in tooltips.
        lead_hours: Forecast lead hour.
        output: Output HTML file path.
        zoom_start: Initial zoom level.
        tiles: Tile layer name.

    Returns:
        Path to the saved HTML file.
    """
    import folium

    if metrics is None:
        metrics = ["mae"]

    points_data = _load_grid_data(data_dir, variable, metrics[0], lead_hours)
    if not points_data:
        raise ValueError(f"No grid data found in {data_dir}")

    center_lat = sum(p["lat"] for p in points_data) / len(points_data)
    center_lon = sum(p["lon"] for p in points_data) / len(points_data)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=tiles)

    for p in points_data:
        popup_parts = [f'<b>{p["grid_id"]}</b>', f'Lat: {p["lat"]:.4f}', f'Lon: {p["lon"]:.4f}']
        for metric in metrics:
            val = p.get(f"error_{metric}", None)
            if val is not None and not np.isnan(val):
                unit = "°C" if "temp" in variable else "m/s"
                popup_parts.append(f"{metric.upper()} {variable}: {val:.2f} {unit}")
        popup = "<br>".join(popup_parts)

        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=7,
            color="steelblue",
            fill=True,
            fill_color="steelblue",
            fill_opacity=0.7,
            popup=popup,
        ).add_to(m)

    if output is None:
        output = Path(f"error_map_{variable}.html")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    return output
