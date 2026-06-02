"""Utility to find model grid points within a square region.

Given a center location (lat/lon) and a square bounding box (±N km),
returns all native model grid points (ECMWF IFS / ERA5) within that region.

Model Grid Background
---------------------
ECMWF IFS and ERA5 both use a ~0.25° latitude × ~0.25° longitude grid,
with origin at (-90°, -180°). Grid points are at:
  lat = -90 + i * 0.25   for i = 0, 1, 2, ...
  lon = -180 + j * 0.25  for j = 0, 1, 2, ...

At ~55°N:
  1° latitude  ≈ 111 km
  1° longitude ≈ 64 km (varies with cos(lat))

For ±50 km around Copenhagen this yields roughly 3×16 points per model.

Usage
-----
>>> from forecast.grid_points import find_grid_points
>>> points = find_grid_points(55.605, 12.574, radius_km=50)
>>> len(points)
48  # 24 IFS + 24 ERA5

>>> points[0]
{'latitude': 55.375, 'longitude': 10.875, 'model': 'ifs', 'distance_km': 47.2}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Model grid constants
# ---------------------------------------------------------------------------

# Grid spacing in degrees (same for IFS and ERA5)
GRID_SPACING_DEG = 0.25

# Grid origin in degrees (WGS84)
GRID_ORIGIN_LAT = -90.0
GRID_ORIGIN_LON = -180.0

# Earth radius for haversine (WGS84 mean radius)
EARTH_RADIUS_KM = 6371.0088

# Average km per degree at equator
KM_PER_DEG_LAT = EARTH_RADIUS_KM * math.pi / 180.0  # ~111.195 km/deg

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridPoint:
    """A single model grid point with metadata."""

    latitude: float
    longitude: float
    model: Literal["ifs", "era5"]
    distance_km: float


# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------


def km_to_degrees(
    center_lat: float,
    km_lat: float,
    km_lon: float,
) -> tuple[float, float]:
    """Convert distances in km to degrees at a given latitude.

    Latitude spacing is nearly constant (~111 km/deg everywhere).
    Longitude spacing widens toward the poles (scaled by cos(lat)).
    At the poles, longitude spacing is capped to avoid division by zero.

    Args:
        center_lat: Center latitude in degrees (WGS84).
        km_lat: Distance in km along the north-south axis.
        km_lon: Distance in km along the east-west axis.

    Returns:
        Tuple of (lat_deg, lon_deg) offsets.
    """
    lat_deg = km_lat / KM_PER_DEG_LAT
    # Longitude degrees widen as you move away from the equator.
    # At/near poles, cos(lat) → 0, so cap to a large value (full hemisphere).
    cos_lat = abs(math.cos(math.radians(center_lat)))
    if cos_lat < 1e-6:
        # At the pole, any longitude is the same point — use full 180° range
        lon_deg = 180.0
    else:
        lon_deg = km_lon / (KM_PER_DEG_LAT * cos_lat)
    return lat_deg, lon_deg


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in km.

    Args:
        lat1, lon1: First point.
        lat2, lon2: Second point.

    Returns:
        Distance in km.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Grid point generation
# ---------------------------------------------------------------------------


def _generate_grid_points(
    center_lat: float,
    center_lon: float,
    radius_km: float,
) -> list[GridPoint]:
    """Generate grid points within a square region centered on the given location.

    Args:
        center_lat: Center latitude in degrees.
        center_lon: Center longitude in degrees.
        radius_km: Half-side of the square region in km (total width = 2 * radius_km).

    Returns:
        List of GridPoint objects within the square, sorted by distance.
    """
    # Convert km radius to degrees
    lat_deg, lon_deg = km_to_degrees(center_lat, radius_km, radius_km)

    # Compute bounding box in degrees
    min_lat = center_lat - lat_deg
    max_lat = center_lat + lat_deg
    min_lon = center_lon - lon_deg
    max_lon = center_lon + lon_deg

    # Generate grid points at 0.25° intervals
    # Grid origin is at (-90, -180), so valid grid points are:
    #   lat = -90 + i * 0.25
    #   lon = -180 + j * 0.25
    # We find the first grid index >= min for each axis.
    lat_start_idx = math.ceil((min_lat - GRID_ORIGIN_LAT) / GRID_SPACING_DEG)
    lon_start_idx = math.ceil((min_lon - GRID_ORIGIN_LON) / GRID_SPACING_DEG)
    lat_end_idx = math.floor((max_lat - GRID_ORIGIN_LAT) / GRID_SPACING_DEG)
    lon_end_idx = math.floor((max_lon - GRID_ORIGIN_LON) / GRID_SPACING_DEG)

    points: list[GridPoint] = []
    for i in range(lat_start_idx, lat_end_idx + 1):
        for j in range(lon_start_idx, lon_end_idx + 1):
            lat = GRID_ORIGIN_LAT + i * GRID_SPACING_DEG
            lon = GRID_ORIGIN_LON + j * GRID_SPACING_DEG

            # Clamp to valid range
            lat = max(-90.0, min(90.0, lat))
            lon = ((lon + 180.0) % 360.0) - 180.0

            dist = haversine_km(center_lat, center_lon, lat, lon)

            points.append(GridPoint(
                latitude=round(lat, 3),
                longitude=round(lon, 3),
                model="ifs",
                distance_km=round(dist, 1),
            ))
            points.append(GridPoint(
                latitude=round(lat, 3),
                longitude=round(lon, 3),
                model="era5",
                distance_km=round(dist, 1),
            ))

    # Sort by distance from center
    points.sort(key=lambda p: p.distance_km)
    return points


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_grid_points(
    center_lat: float,
    center_lon: float,
    radius_km: float = 50.0,
    models: tuple[str, ...] | None = None,
) -> list[GridPoint]:
    """Find all model grid points within a square region.

    Given a center location and a square bounding box (±radius_km from center
    in each direction), returns all ECMWF IFS and ERA5 grid points that fall
    within that region.

    Both IFS and ERA5 share the same 0.25° grid, so the same lat/lon positions
    apply to both models. Each point is returned twice — once tagged "ifs" and
    once tagged "era5".

    Args:
        center_lat: Center latitude in degrees (WGS84).
        center_lon: Center longitude in degrees (WGS84).
        radius_km: Half-side of the square region in km.
            Total region width = 2 * radius_km.
        models: If specified, filter to only these models.
            Valid values: "ifs", "era5". If None, returns both.

    Returns:
        List of GridPoint objects sorted by distance from center.

    Raises:
        ValueError: If center_lat is out of [-90, 90] or center_lon out of [-180, 180].
        ValueError: If radius_km is not positive.
        ValueError: If models contains invalid model names.

    Example:
        >>> points = find_grid_points(55.605, 12.574, radius_km=50)
        >>> len(points)
        48  # 24 IFS + 24 ERA5

        >>> ifs_only = find_grid_points(55.605, 12.574, radius_km=50, models=("ifs",))
        >>> len(ifs_only)
        24
    """
    # Validate inputs
    if not (-90 <= center_lat <= 90):
        raise ValueError(f"center_lat must be in [-90, 90], got {center_lat}")
    if not (-180 <= center_lon <= 180):
        raise ValueError(f"center_lon must be in [-180, 180], got {center_lon}")
    if radius_km <= 0:
        raise ValueError(f"radius_km must be positive, got {radius_km}")

    valid_models = {"ifs", "era5"}
    if models is not None:
        invalid = set(models) - valid_models
        if invalid:
            raise ValueError(f"Invalid model(s): {invalid}. Valid: {valid_models}")

    # Generate all grid points
    all_points = _generate_grid_points(center_lat, center_lon, radius_km)

    # Filter by model if specified
    if models is not None:
        all_points = [p for p in all_points if p.model in models]

    return all_points


def summarize_grid_points(
    center_lat: float,
    center_lon: float,
    radius_km: float = 50.0,
) -> dict:
    """Return a human-readable summary of grid points in a region.

    Args:
        center_lat: Center latitude.
        center_lon: Center longitude.
        radius_km: Half-side of square region.

    Returns:
        Dictionary with summary statistics.
    """
    points = find_grid_points(center_lat, center_lon, radius_km)

    ifs_points = [p for p in points if p.model == "ifs"]
    era5_points = [p for p in points if p.model == "era5"]

    # Compute unique lat/lon positions (same for both models)
    unique_positions = set((p.latitude, p.longitude) for p in points)

    return {
        "center": {"lat": center_lat, "lon": center_lon},
        "radius_km": radius_km,
        "total_points": len(points),
        "ifs_points": len(ifs_points),
        "era5_points": len(era5_points),
        "unique_positions": len(unique_positions),
        "min_distance_km": min(p.distance_km for p in points) if points else None,
        "max_distance_km": max(p.distance_km for p in points) if points else None,
        "lat_range": {
            "min": min(p.latitude for p in points) if points else None,
            "max": max(p.latitude for p in points) if points else None,
        },
        "lon_range": {
            "min": min(p.longitude for p in points) if points else None,
            "max": max(p.longitude for p in points) if points else None,
        },
    }
