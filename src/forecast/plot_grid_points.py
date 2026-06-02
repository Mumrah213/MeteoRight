"""Visualization utilities for grid point finder.

Provides:
- Static map plots with basemap tiles (contextily)
- Folium interactive HTML maps
- Multi-location comparison plots

Usage
-----
>>> from forecast.plot_grid_points import plot_grid_map, plot_grid_comparison
>>> plot_grid_map(55.605, 12.574, radius_km=50, output="/tmp/copenhagen_grid.png")
>>> plot_grid_comparison([("Copenhagen", 55.605, 12.574, 50)], output="/tmp/comparison.png")
>>> plot_grid_folium(55.605, 12.574, radius_km=50, output="/tmp/copenhagen_grid.html")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np

# Grid point data type
GridPointTuple = Tuple[float, float, str, float]  # lat, lon, model, distance_km


def _get_bounds(
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> tuple[float, float, float, float]:
    """Compute the square bounding box in degrees.

    Args:
        center_lat: Center latitude.
        center_lon: Center longitude.
        radius_km: Half-side of the square in km.

    Returns:
        (min_lat, max_lat, min_lon, max_lon).
    """
    km_per_deg_lat = 6371.0088 * math.pi / 180.0
    lat_deg = radius_km / km_per_deg_lat
    cos_lat = abs(math.cos(math.radians(center_lat)))
    if cos_lat < 1e-6:
        lon_deg = 180.0
    else:
        lon_deg = radius_km / (km_per_deg_lat * cos_lat)

    return (
        center_lat - lat_deg,
        center_lat + lat_deg,
        center_lon - lon_deg,
        center_lon + lon_deg,
    )


def plot_grid_map(
    center_lat: float,
    center_lon: float,
    radius_km: float = 50.0,
    *,
    output: str | Path | None = None,
    title: str | None = None,
    basemap_source: str = "CartoDB.Voyager",
) -> Path:
    """Plot grid points on a map with basemap tiles.

    Requires: contextily, matplotlib.

    Args:
        center_lat: Center latitude.
        center_lon: Center longitude.
        radius_km: Half-side of square region in km.
        output: Output file path (PNG). If None, returns figure.
        title: Plot title. Defaults to location name.
        basemap_source: Contextily basemap source string.

    Returns:
        Path to the saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from .grid_points import find_grid_points

    points = find_grid_points(center_lat, center_lon, radius_km)
    ifs = [p for p in points if p.model == "ifs"]
    era5 = [p for p in points if p.model == "era5"]

    min_lat, max_lat, min_lon, max_lon = _get_bounds(center_lat, center_lon, radius_km)

    fig, ax = plt.subplots(figsize=(12, 10))

    # IFS points (circles)
    ax.scatter(
        [p.longitude for p in ifs],
        [p.latitude for p in ifs],
        c="steelblue",
        s=100,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="IFS",
    )
    # ERA5 points (squares)
    ax.scatter(
        [p.longitude for p in era5],
        [p.latitude for p in era5],
        c="tomato",
        s=100,
        alpha=0.7,
        marker="s",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="ERA5",
    )

    # Bounding box
    rect = patches.Rectangle(
        (min_lon, min_lat),
        max_lon - min_lon,
        max_lat - min_lat,
        linewidth=2,
        edgecolor="green",
        facecolor="none",
        linestyle="--",
        zorder=4,
        label=f"Square (\u00b1{radius_km} km)",
    )
    ax.add_patch(rect)

    # Center
    ax.plot(
        center_lon,
        center_lat,
        "k*",
        markersize=20,
        zorder=5,
        label="Center",
    )

    # Basemap
    try:
        import contextily as ctx

        ctx.add_basemap(ax, crs="EPSG:4326", source=basemap_source, alpha=0.6)
    except ImportError:
        ax.set_facecolor("#e8f4f8")

    unique = len(set((p.latitude, p.longitude) for p in points)) // 2
    if title is None:
        title = f"ECMWF IFS / ERA5 Grid \u2014 ({center_lat:.3f}, {center_lon:.3f})"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()

    if output is None:
        return fig

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_grid_comparison(
    locations: Sequence[Tuple[str, float, float, float]],
    *,
    output: str | Path | None = None,
    radius_km: float = 50.0,
) -> Path:
    """Plot grid points for multiple locations side by side.

    Args:
        locations: List of (name, lat, lon, radius_km) tuples.
        output: Output file path (PNG).
        radius_km: Default radius if not specified per location.

    Returns:
        Path to the saved image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from .grid_points import find_grid_points

    n = len(locations)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ["steelblue", "tomato"]

    for ax, (name, lat, lon, r) in zip(axes, locations):
        r = r or radius_km
        points = find_grid_points(lat, lon, r)
        ifs = [p for p in points if p.model == "ifs"]
        era5 = [p for p in points if p.model == "era5"]

        min_lat, max_lat, min_lon, max_lon = _get_bounds(lat, lon, r)

        ax.scatter(
            [p.longitude for p in ifs],
            [p.latitude for p in ifs],
            c=colors[0],
            s=80,
            alpha=0.6,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.scatter(
            [p.longitude for p in era5],
            [p.latitude for p in era5],
            c=colors[1],
            s=80,
            alpha=0.6,
            marker="s",
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

        rect = patches.Rectangle(
            (min_lon, min_lat),
            max_lon - min_lon,
            max_lat - min_lat,
            linewidth=1.5,
            edgecolor="green",
            facecolor="none",
            linestyle="--",
            alpha=0.8,
        )
        ax.add_patch(rect)
        ax.plot(lon, lat, "k*", markersize=12, zorder=4)

        unique = len(set((p.latitude, p.longitude) for p in points)) // 2
        ax.set_title(f"{name}\n{unique} grid positions\n\u00b1{r} km", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (\u00b0)")
        ax.set_ylabel("Latitude (\u00b0)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect(
            (max_lat - min_lat) / (max_lon - min_lon) * np.cos(np.radians(lat))
        )

    plt.suptitle("ECMWF IFS / ERA5 0.25\u00b0 Grid Points in Square Regions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output is None:
        return fig

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_grid_folium(
    center_lat: float,
    center_lon: float,
    radius_km: float = 50.0,
    *,
    output: str | Path | None = None,
    zoom_start: int = 10,
    tiles: str = "CartoDB.Voyager",
) -> Path:
    """Create an interactive HTML map with folium.

    Requires: folium.

    Args:
        center_lat: Center latitude.
        center_lon: Center longitude.
        radius_km: Half-side of square region in km.
        output: Output HTML file path.
        zoom_start: Initial zoom level.
        tiles: Tile layer name.

    Returns:
        Path to the saved HTML file.
    """
    import folium

    from .grid_points import find_grid_points

    points = find_grid_points(center_lat, center_lon, radius_km)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=tiles,
    )

    # Bounding box
    min_lat, max_lat, min_lon, max_lon = _get_bounds(center_lat, center_lon, radius_km)

    # Square polygon
    coords = [
        [min_lat, min_lon],
        [max_lat, min_lon],
        [max_lat, max_lon],
        [min_lat, max_lon],
        [min_lat, min_lon],
    ]
    folium.Polygon(
        locations=coords,
        color="green",
        weight=3,
        fill=False,
        popup=f"Square region (\u00b1{radius_km} km)",
    ).add_to(m)

    # IFS points (circles)
    for p in points:
        if p.model == "ifs":
            folium.CircleMarker(
                location=[p.latitude, p.longitude],
                radius=6,
                color="steelblue",
                fill=True,
                fill_color="steelblue",
                fill_opacity=0.7,
                weight=1,
                popup=f"IFS ({p.latitude:.3f}, {p.longitude:.3f})\nDist: {p.distance_km:.1f} km",
            ).add_to(m)

    # ERA5 points (squares)
    for p in points:
        if p.model == "era5":
            folium.RegularPolygonMarker(
                location=[p.latitude, p.longitude],
                number_of_sides=4,
                rotation=0,
                radius=6,
                color="tomato",
                fill=True,
                fill_color="tomato",
                fill_opacity=0.7,
                weight=1,
                popup=f"ERA5 ({p.latitude:.3f}, {p.longitude:.3f})\nDist: {p.distance_km:.1f} km",
            ).add_to(m)

    # Center marker
    folium.Marker(
        location=[center_lat, center_lon],
        icon=folium.Icon(color="black", icon="star", prefix="glyphicon"),
        popup=f"Center ({center_lat}, {center_lon})",
    ).add_to(m)

    if output is None:
        output = Path(f"grid_map_{center_lat}_{center_lon}.html")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output))
    return output
