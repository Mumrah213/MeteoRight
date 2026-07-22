"""Apply forecast-to-observation interpolation across whole DataFrames.

`interpolate.py` holds the per-point maths. This module applies it to the
forecast/observation tables the pipeline actually carries, vectorised over
time so a year of hourly data finishes in seconds rather than minutes.

The operation replaces each forecast value with an estimate of that forecast
*at the observation location*, leaving provenance columns untouched. Errors
are computed afterwards, from the interpolated value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .interpolate import cluster_coordinates, haversine_km, snap_to_lines

Method = Literal["nearest", "bilinear", "idw"]

# Variables measured in degrees, where 350 and 10 are 20 degrees apart.
CIRCULAR_VARIABLES = frozenset({"wind_direction_10m", "wind_direction_100m", "wind_direction_80m"})


@dataclass
class InterpolationReport:
    """What the interpolation actually did, for the caller to report."""

    method: str
    grid_points: int
    target_points: int
    mean_offset_km: float
    max_offset_km: float
    variables: tuple[str, ...]


def _target_location(observations: pd.DataFrame) -> tuple[float, float] | None:
    """The observation location, if the frame has exactly one."""
    if "latitude" not in observations.columns or "longitude" not in observations.columns:
        return None
    points = observations[["latitude", "longitude"]].drop_duplicates()
    if len(points) != 1:
        return None
    return float(points.iloc[0]["latitude"]), float(points.iloc[0]["longitude"])


def interpolate_forecasts_to_observations(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    variables: list[str],
    *,
    method: Method = "bilinear",
) -> tuple[pd.DataFrame, InterpolationReport]:
    """Replace forecast values with estimates at the observation location.

    Each model is interpolated on its own grid — never across models, whose
    grid spacings differ — and the result keeps one row per
    (issue time, target time, model) with the forecast columns overwritten.

    Args:
        forecasts: Forecast rows carrying latitude/longitude per grid point.
        observations: Observation rows; their location is the target.
        variables: Variable names to interpolate.
        method: "nearest", "bilinear" or "idw".

    Returns:
        (interpolated forecasts, report). When the inputs cannot support
        interpolation — no coordinates, a single grid point, no single
        observation location — the forecasts are returned unchanged and the
        report says so via ``grid_points``.
    """
    target = _target_location(observations)
    present = [v for v in variables if v in forecasts.columns]

    if target is None or not present or "latitude" not in forecasts.columns:
        return forecasts, InterpolationReport(method, 0, 0, 0.0, 0.0, tuple(present))

    target_lat, target_lon = target
    grid = forecasts[["latitude", "longitude"]].drop_duplicates()
    offsets = [
        haversine_km(target_lat, target_lon, row.latitude, row.longitude)
        for row in grid.itertuples()
    ]
    report = InterpolationReport(
        method=method,
        grid_points=len(grid),
        target_points=1,
        mean_offset_km=float(np.mean(offsets)) if offsets else 0.0,
        max_offset_km=float(np.max(offsets)) if offsets else 0.0,
        variables=tuple(present),
    )
    if len(grid) < 4 and method == "bilinear":
        # Not enough points to define a cell; leave the caller to report it.
        return forecasts, report

    out_frames = []
    for _model, model_rows in forecasts.groupby("model", sort=False):
        out_frames.append(
            _interpolate_one_model(model_rows, target_lat, target_lon, present, method)
        )

    interpolated = pd.concat(out_frames, ignore_index=True)
    return interpolated, report


def _interpolate_one_model(
    rows: pd.DataFrame,
    target_lat: float,
    target_lon: float,
    variables: list[str],
    method: Method,
) -> pd.DataFrame:
    """Interpolate a single model's grid onto the target, vectorised over time."""
    keys = ["forecast_issue_time", "forecast_target_time", "lead_hours", "model"]
    keys = [k for k in keys if k in rows.columns]

    # One representative row per (issue, target) to carry provenance forward.
    result = rows.drop_duplicates(subset=keys).copy().reset_index(drop=True)
    result["latitude"] = target_lat
    result["longitude"] = target_lon

    # Derive the grid lines from the data rather than assuming a spacing
    # anchored at zero: repeated requests for one cell come back with a few
    # hundred metres of jitter, and each distinct line must collapse to a
    # single grid coordinate or the cube is built wrong.
    lat_lines = cluster_coordinates(rows["latitude"].to_numpy())
    lon_lines = cluster_coordinates(rows["longitude"].to_numpy())
    use_bilinear = method == "bilinear" and lat_lines.size >= 2 and lon_lines.size >= 2

    grid_lat = snap_to_lines(rows["latitude"].to_numpy(), lat_lines)
    grid_lon = snap_to_lines(rows["longitude"].to_numpy(), lon_lines)

    frame = rows.assign(_glat=grid_lat, _glon=grid_lon)
    index_key = "forecast_target_time" if "forecast_target_time" in frame.columns else keys[0]

    for variable in variables:
        pivot = frame.pivot_table(
            index=index_key, columns=["_glat", "_glon"], values=variable, aggfunc="first"
        )
        if pivot.empty:
            continue

        lats = np.array(sorted({c[0] for c in pivot.columns}))
        lons = np.array(sorted({c[1] for c in pivot.columns}))
        cube = np.full((len(pivot), len(lats), len(lons)), np.nan)
        lat_index = {v: i for i, v in enumerate(lats)}
        lon_index = {v: i for i, v in enumerate(lons)}
        for column in pivot.columns:
            cube[:, lat_index[column[0]], lon_index[column[1]]] = pivot[column].to_numpy()

        circular = variable in CIRCULAR_VARIABLES
        values = _sample_cube(
            cube, lats, lons, target_lat, target_lon, method if use_bilinear else "idw", circular
        )
        mapped = pd.Series(values, index=pivot.index)
        result[variable] = result[index_key].map(mapped).to_numpy()

    return result.drop(columns=[c for c in ("_glat", "_glon") if c in result.columns])


def _sample_cube(
    cube: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    target_lat: float,
    target_lon: float,
    method: str,
    circular: bool,
) -> np.ndarray:
    """Sample a [time, lat, lon] cube at one location, for every time step."""
    if method == "nearest" or len(lats) < 2 or len(lons) < 2:
        i = int(np.argmin(np.abs(lats - target_lat)))
        j = int(np.argmin(np.abs(lons - target_lon)))
        return cube[:, i, j]

    if method == "idw":
        lat_mesh, lon_mesh = np.meshgrid(lats, lons, indexing="ij")
        distances = np.array(
            [
                haversine_km(target_lat, target_lon, la, lo)
                for la, lo in zip(lat_mesh.ravel(), lon_mesh.ravel(), strict=True)
            ]
        )
        flat = cube.reshape(cube.shape[0], -1)
        order = np.argsort(distances)[:4]
        weights = 1.0 / np.maximum(np.power(distances[order], 2.0), 1e-12)
        weights = weights / weights.sum()
        return _weighted(flat[:, order], weights, circular)

    # Bilinear: locate the enclosing cell, then weight its four corners.
    i0 = int(np.clip(np.searchsorted(lats, target_lat) - 1, 0, len(lats) - 2))
    j0 = int(np.clip(np.searchsorted(lons, target_lon) - 1, 0, len(lons) - 2))
    span_lat = lats[i0 + 1] - lats[i0]
    span_lon = lons[j0 + 1] - lons[j0]
    v = np.clip((target_lat - lats[i0]) / span_lat, 0.0, 1.0) if span_lat else 0.0
    u = np.clip((target_lon - lons[j0]) / span_lon, 0.0, 1.0) if span_lon else 0.0

    corners = np.column_stack(
        [cube[:, i0, j0], cube[:, i0, j0 + 1], cube[:, i0 + 1, j0], cube[:, i0 + 1, j0 + 1]]
    )
    weights = np.array([(1 - v) * (1 - u), (1 - v) * u, v * (1 - u), v * u])
    return _weighted(corners, weights, circular)


def _weighted(values: np.ndarray, weights: np.ndarray, circular: bool) -> np.ndarray:
    """Weighted combination across the last axis, renormalised over finite entries."""
    finite = np.isfinite(values)
    weight_matrix = np.broadcast_to(weights, values.shape) * finite
    totals = weight_matrix.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(totals[:, None] > 0, weight_matrix / totals[:, None], np.nan)
        safe = np.where(finite, values, 0.0)
        if circular:
            radians = np.radians(safe)
            x = np.nansum(normalised * np.cos(radians), axis=1)
            y = np.nansum(normalised * np.sin(radians), axis=1)
            result = np.degrees(np.arctan2(y, x)) % 360.0
        else:
            result = np.nansum(normalised * safe, axis=1)

    return np.where(totals > 0, result, np.nan)
