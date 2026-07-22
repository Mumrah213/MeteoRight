"""Interpolate forecasts from model grid points onto observation locations.

Forecast models are evaluated on their own grid; observations sit wherever a
station or analysis point happens to be. Those two locations rarely coincide,
so a native comparison charges the model for a spatial mismatch it never had a
chance to get right. Interpolating the forecast field to the observation point
first removes that component of the error.

This is a *verification-stage* operation. It changes the forecast value that a
verification row carries, before any error is computed — not the error field,
and not the plot. Interpolating an error surface for display (as a contour
plot does) is a different, purely cosmetic thing.

Three methods, in decreasing order of how much they assume:

``bilinear``
    Weight the four grid points of the enclosing cell by their overlap with
    the target. Needs a regular grid; the natural choice when one exists.
``idw``
    Inverse-distance weighting over the ``k`` nearest points. Works on any
    scattered layout, so it is the fallback when the grid is irregular.
``nearest``
    Take the closest grid point. This is the baseline — it is what a native
    comparison already does, so it is useful mainly as a control.

Every method here operates on a *single model's* grid. Never build a
neighbourhood from several models at once: their grids differ (IFS 0.25°,
ICON 0.125°, ICON-EU 0.0625°), and mixing them silently blends model biases
into what is supposed to be a spatial correction. Interpolate each model to
the observation points independently, then compare.

Directional variables (wind direction) are circular: averaging 350° and 10°
must give 0°, not 180°. Pass ``circular=True`` for those, and the weighting is
applied to the unit vectors instead of the raw degrees.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

Method = Literal["nearest", "bilinear", "idw"]

# Grid spacings are degrees; below this two coordinates are the same point.
COORD_TOLERANCE_DEG = 1e-6

# Open-Meteo returns coordinates rounded to the grid cell it served, with a
# little jitter (observed: <500 m). Snapping tolerates that when inferring a
# regular grid.
DEFAULT_SNAP_TOLERANCE_DEG = 0.02

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _circular_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean of angles in degrees, via unit vectors."""
    radians = np.radians(values)
    x = float(np.sum(weights * np.cos(radians)))
    y = float(np.sum(weights * np.sin(radians)))
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return float("nan")
    return float(np.degrees(math.atan2(y, x)) % 360.0)


def _combine(values: np.ndarray, weights: np.ndarray, circular: bool) -> float:
    """Apply weights to values, respecting circularity. Weights must sum to 1."""
    if circular:
        return _circular_mean(values, weights)
    return float(np.sum(weights * values))


def cluster_coordinates(
    coords: np.ndarray,
    *,
    tolerance: float = DEFAULT_SNAP_TOLERANCE_DEG,
) -> np.ndarray:
    """Collapse near-identical coordinates into their cluster centres.

    Open-Meteo returns the coordinate of the grid cell it served, and repeated
    requests for the same cell can differ by a few hundred metres. Those are
    one grid line, not several, so gaps must be measured between cluster
    centres — taking the raw minimum gap would read the jitter itself as the
    grid spacing.
    """
    unique = np.unique(np.asarray(coords, dtype=float))
    if unique.size == 0:
        return unique

    centres: list[float] = []
    group = [unique[0]]
    for value in unique[1:]:
        if value - group[-1] <= tolerance:
            group.append(value)
        else:
            centres.append(float(np.mean(group)))
            group = [value]
    centres.append(float(np.mean(group)))
    return np.array(centres)


def infer_grid_spacing(
    coords: np.ndarray,
    *,
    tolerance: float = DEFAULT_SNAP_TOLERANCE_DEG,
) -> float | None:
    """Infer a regular spacing from one coordinate axis.

    Returns the spacing, or None when the axis has fewer than two distinct
    grid lines or the gaps are not consistent with a single regular step.
    """
    centres = cluster_coordinates(coords, tolerance=tolerance)
    if centres.size < 2:
        return None

    gaps = np.diff(centres)
    gaps = gaps[gaps > tolerance]
    if gaps.size == 0:
        return None

    spacing = float(np.min(gaps))
    # Every gap must be a near-integer multiple of the smallest one.
    multiples = gaps / spacing
    if not np.allclose(multiples, np.round(multiples), atol=0.1):
        return None
    return spacing


def snap_to_grid(values: np.ndarray, spacing: float) -> np.ndarray:
    """Round coordinates onto a regular grid of the given spacing.

    Note this assumes the grid is anchored at zero, which is true for the
    standard global grids (0.25 deg lines fall on multiples of 0.25). Prefer
    :func:`snap_to_lines` when the grid lines are known, since it needs no
    such assumption.
    """
    return np.round(np.asarray(values, dtype=float) / spacing) * spacing


def snap_to_lines(values: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Snap each coordinate to the nearest known grid line.

    This is the safe counterpart to :func:`snap_to_grid`: it derives the grid
    from the data rather than assuming an anchor, so a grid offset from zero
    (or one whose spacing is only approximately regular) still resolves to
    consistent grid points.
    """
    values = np.asarray(values, dtype=float)
    lines = np.asarray(lines, dtype=float)
    if lines.size == 0:
        return values
    nearest = np.abs(values[:, None] - lines[None, :]).argmin(axis=1)
    return lines[nearest]


def interpolate_nearest(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_values: np.ndarray,
    target_lat: float,
    target_lon: float,
    *,
    circular: bool = False,
) -> float:
    """Value at the closest grid point. The native-comparison baseline."""
    del circular  # a single value needs no combination
    distances = np.array(
        [
            haversine_km(target_lat, target_lon, la, lo)
            for la, lo in zip(grid_lats, grid_lons, strict=True)
        ]
    )
    return float(grid_values[int(np.argmin(distances))])


def interpolate_idw(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_values: np.ndarray,
    target_lat: float,
    target_lon: float,
    *,
    power: float = 2.0,
    k: int = 4,
    circular: bool = False,
) -> float:
    """Inverse-distance weighted value from the ``k`` nearest grid points.

    Falls back to the exact value when the target coincides with a grid point,
    which keeps the interpolation exact at the nodes.
    """
    distances = np.array(
        [
            haversine_km(target_lat, target_lon, la, lo)
            for la, lo in zip(grid_lats, grid_lons, strict=True)
        ]
    )

    exact = np.flatnonzero(distances < 1e-9)
    if exact.size:
        return float(grid_values[exact[0]])

    order = np.argsort(distances)[: max(1, min(k, distances.size))]
    weights = 1.0 / np.power(distances[order], power)
    weights = weights / weights.sum()
    return _combine(grid_values[order], weights, circular)


def interpolate_bilinear(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_values: np.ndarray,
    target_lat: float,
    target_lon: float,
    *,
    spacing_lat: float | None = None,
    spacing_lon: float | None = None,
    circular: bool = False,
) -> float:
    """Bilinear value from the four grid points enclosing the target.

    Requires a regular grid. When the enclosing cell is incomplete — the target
    is outside the grid, or a corner is missing — this falls back to IDW rather
    than extrapolating, since extrapolating a weather field off the edge of its
    own grid is not defensible.
    """
    lats = np.asarray(grid_lats, dtype=float)
    lons = np.asarray(grid_lons, dtype=float)
    values = np.asarray(grid_values, dtype=float)

    if spacing_lat is None:
        spacing_lat = infer_grid_spacing(lats)
    if spacing_lon is None:
        spacing_lon = infer_grid_spacing(lons)
    if spacing_lat is None or spacing_lon is None:
        return interpolate_idw(lats, lons, values, target_lat, target_lon, circular=circular)

    snapped_lats = snap_to_grid(lats, spacing_lat)
    snapped_lons = snap_to_grid(lons, spacing_lon)

    lat0 = math.floor(target_lat / spacing_lat) * spacing_lat
    lon0 = math.floor(target_lon / spacing_lon) * spacing_lon
    lat1, lon1 = lat0 + spacing_lat, lon0 + spacing_lon

    corners: list[float] = []
    for corner_lat in (lat0, lat1):
        for corner_lon in (lon0, lon1):
            match = np.flatnonzero(
                (np.abs(snapped_lats - corner_lat) < spacing_lat / 2)
                & (np.abs(snapped_lons - corner_lon) < spacing_lon / 2)
            )
            if match.size == 0:
                # Incomplete cell — IDW rather than extrapolate.
                return interpolate_idw(
                    lats, lons, values, target_lat, target_lon, circular=circular
                )
            corners.append(float(values[match[0]]))

    # Fractional position within the cell.
    u = (target_lon - lon0) / spacing_lon
    v = (target_lat - lat0) / spacing_lat
    u = min(max(u, 0.0), 1.0)
    v = min(max(v, 0.0), 1.0)

    weights = np.array(
        [(1 - v) * (1 - u), (1 - v) * u, v * (1 - u), v * u],
        dtype=float,
    )
    return _combine(np.array(corners, dtype=float), weights, circular)


def interpolate(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_values: np.ndarray,
    target_lat: float,
    target_lon: float,
    *,
    method: Method = "bilinear",
    circular: bool = False,
    power: float = 2.0,
    k: int = 4,
) -> float:
    """Interpolate a single model's grid values onto one target location.

    Args:
        grid_lats: Latitudes of the grid points, one per value.
        grid_lons: Longitudes of the grid points, one per value.
        grid_values: Forecast values at those grid points.
        target_lat: Latitude of the observation location.
        target_lon: Longitude of the observation location.
        method: "nearest", "bilinear" or "idw".
        circular: True for directional variables measured in degrees.
        power: IDW distance exponent.
        k: Number of neighbours IDW considers.

    Returns:
        The interpolated value, or NaN if no finite grid value is available.

    Raises:
        ValueError: If the inputs disagree in length or the method is unknown.
    """
    lats = np.asarray(grid_lats, dtype=float)
    lons = np.asarray(grid_lons, dtype=float)
    values = np.asarray(grid_values, dtype=float)

    if not (lats.size == lons.size == values.size):
        raise ValueError(
            f"grid arrays must be the same length: got {lats.size}, {lons.size}, {values.size}"
        )
    if lats.size == 0:
        raise ValueError("no grid points supplied")

    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    lats, lons, values = lats[finite], lons[finite], values[finite]

    if method == "nearest":
        return interpolate_nearest(lats, lons, values, target_lat, target_lon, circular=circular)
    if method == "idw":
        return interpolate_idw(
            lats, lons, values, target_lat, target_lon, power=power, k=k, circular=circular
        )
    if method == "bilinear":
        return interpolate_bilinear(lats, lons, values, target_lat, target_lon, circular=circular)
    raise ValueError(f"unknown interpolation method: {method!r}")
