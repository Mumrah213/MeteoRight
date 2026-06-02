"""Location invariants for forecast/observation verification joins."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

import pandas as pd


DEFAULT_MAX_LOCATION_DISTANCE_KM = 50.0


@dataclass(frozen=True)
class LocationAudit:
    """Summary of forecast/observation location compatibility."""

    forecast_locations: int
    observation_locations: int
    max_nearest_distance_km: float
    has_location_id: bool


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points."""
    radius_km = 6371.0088
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)
    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    return 2 * radius_km * asin(sqrt(a))


def _unique_locations(df: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {"latitude", "longitude"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} data missing required location columns: {missing}")

    locs = (
        df[["latitude", "longitude"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if locs.empty:
        raise ValueError(f"{label} data has no non-null latitude/longitude values")
    return locs


def _check_location_ids(forecasts: pd.DataFrame, observations: pd.DataFrame) -> bool:
    if "location_id" not in forecasts.columns and "location_id" not in observations.columns:
        return False
    if "location_id" not in forecasts.columns or "location_id" not in observations.columns:
        raise ValueError("location_id must be present on both forecasts and observations")

    forecast_ids = set(forecasts["location_id"].dropna().unique())
    observation_ids = set(observations["location_id"].dropna().unique())
    if not forecast_ids:
        raise ValueError("Forecast data has location_id column but no non-null values")
    if not observation_ids:
        raise ValueError("Observation data has location_id column but no non-null values")
    missing = forecast_ids - observation_ids
    if missing:
        raise ValueError(f"Forecast location_id values missing from observations: {sorted(missing)}")
    return True


def _validate_location_id_distances(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    max_distance_km: float,
) -> float:
    max_nearest = 0.0
    too_far = []
    for location_id, fcst_group in forecasts.dropna(subset=["location_id"]).groupby("location_id"):
        obs_group = observations[observations["location_id"] == location_id]
        if obs_group.empty:
            continue
        obs_locs = _unique_locations(obs_group, f"Observation location_id={location_id}")
        fcst_locs = _unique_locations(fcst_group, f"Forecast location_id={location_id}")
        for _, fcst_loc in fcst_locs.iterrows():
            nearest = min(
                haversine_km(
                    float(fcst_loc["latitude"]),
                    float(fcst_loc["longitude"]),
                    float(obs_loc["latitude"]),
                    float(obs_loc["longitude"]),
                )
                for _, obs_loc in obs_locs.iterrows()
            )
            max_nearest = max(max_nearest, nearest)
            if nearest > max_distance_km:
                too_far.append((location_id, float(fcst_loc["latitude"]), float(fcst_loc["longitude"]), nearest))

    if too_far:
        preview = [
            {
                "location_id": location_id,
                "latitude": lat,
                "longitude": lon,
                "nearest_observation_km": round(distance, 1),
            }
            for location_id, lat, lon, distance in too_far[:5]
        ]
        raise ValueError(
            "Refusing to align forecast and observation data from different locations "
            f"within matching location_id values: max_allowed_distance_km={max_distance_km}, "
            f"forecast_locations={preview}"
        )
    return max_nearest


def validate_location_compatibility(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    max_distance_km: float = DEFAULT_MAX_LOCATION_DISTANCE_KM,
    allow_multi_location_without_id: bool = False,
) -> LocationAudit:
    """Fail closed when forecast and observation locations are incompatible.

    Single-location datasets may use nearby, non-identical grid points. Multi-
    location datasets must use `location_id` unless the caller explicitly opts
    into handling multiple locations with exact join keys.
    """
    has_location_id = _check_location_ids(forecasts, observations)
    forecast_locs = _unique_locations(forecasts, "Forecast")
    observation_locs = _unique_locations(observations, "Observation")

    if not has_location_id and not allow_multi_location_without_id:
        if len(forecast_locs) > 1 or len(observation_locs) > 1:
            forecast_preview = forecast_locs.head(5).to_dict("records")
            observation_preview = observation_locs.head(5).to_dict("records")
            raise ValueError(
                "Refusing location-ambiguous verification join: multiple forecast or "
                "observation locations are present but no shared location_id column exists; "
                f"forecast_locations={forecast_preview}, observation_locations={observation_preview}"
            )

    if has_location_id:
        max_nearest = _validate_location_id_distances(forecasts, observations, max_distance_km)
        return LocationAudit(
            forecast_locations=len(forecast_locs),
            observation_locations=len(observation_locs),
            max_nearest_distance_km=max_nearest,
            has_location_id=True,
        )

    too_far: list[tuple[float, float, float]] = []
    max_nearest = 0.0
    for _, fcst_loc in forecast_locs.iterrows():
        nearest = min(
            haversine_km(
                float(fcst_loc["latitude"]),
                float(fcst_loc["longitude"]),
                float(obs_loc["latitude"]),
                float(obs_loc["longitude"]),
            )
            for _, obs_loc in observation_locs.iterrows()
        )
        max_nearest = max(max_nearest, nearest)
        if nearest > max_distance_km:
            too_far.append((float(fcst_loc["latitude"]), float(fcst_loc["longitude"]), nearest))

    if too_far:
        fcst_preview = [
            {"latitude": lat, "longitude": lon, "nearest_observation_km": round(distance, 1)}
            for lat, lon, distance in too_far[:5]
        ]
        obs_preview = observation_locs.head(5).to_dict("records")
        raise ValueError(
            "Refusing to align forecast and observation data from different locations: "
            f"max_allowed_distance_km={max_distance_km}, "
            f"forecast_locations={fcst_preview}, observation_locations={obs_preview}"
        )

    return LocationAudit(
        forecast_locations=len(forecast_locs),
        observation_locations=len(observation_locs),
        max_nearest_distance_km=max_nearest,
        has_location_id=has_location_id,
    )
