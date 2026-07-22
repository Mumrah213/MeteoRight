"""Load the bundled samples and build native/interpolated error tables.

Every showcase figure that compares native-grid verification against
interpolated verification needs the same table: one row per
(site, target time, lead), carrying both forecast variants and the
observation. Building it once here keeps the figures consistent with each
other and with `meteoright verify --interpolate`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
GRID_SAMPLE = REPO_ROOT / "examples/copenhagen_grid_sample"
VERIFICATION_SAMPLE = REPO_ROOT / "examples/copenhagen_sample/verification/verification.parquet"

CIRCULAR = {"wind_direction_10m"}


def circular_error(forecast: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Signed angular difference in [-180, 180)."""
    return (forecast - observed + 180.0) % 360.0 - 180.0


def error_of(forecast: np.ndarray, observed: np.ndarray, variable: str) -> np.ndarray:
    """Forecast minus observation, respecting circular variables."""
    if variable in CIRCULAR:
        return circular_error(forecast, observed)
    return forecast - observed


def load_grid_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bundled grid forecasts and their offset observations."""
    forecasts = pd.read_parquet(GRID_SAMPLE / "forecasts.parquet")
    observations = pd.read_parquet(GRID_SAMPLE / "observations.parquet")
    return forecasts, observations


def build_error_table(variables: list[str]) -> pd.DataFrame:
    """Native and interpolated errors for every site, time and lead.

    Returns one row per (site, target time) with columns
    ``{variable}_native_error`` and ``{variable}_interp_error``, plus
    ``site_distance_km`` — the grid-point-to-observation offset that drives
    how much interpolation can help.
    """
    from verification.interpolate import haversine_km
    from verification.interpolate_frames import interpolate_forecasts_to_observations

    forecasts, observations = load_grid_sample()
    grid = forecasts[["latitude", "longitude"]].drop_duplicates()

    frames = []
    for (site_lat, site_lon), site in observations.groupby(["latitude", "longitude"]):
        nearest = min(
            grid.itertuples(),
            key=lambda row: haversine_km(site_lat, site_lon, row.latitude, row.longitude),
        )
        distance = haversine_km(site_lat, site_lon, nearest.latitude, nearest.longitude)

        interpolated, _ = interpolate_forecasts_to_observations(
            forecasts, site, variables, method="bilinear"
        )
        native = forecasts[
            (forecasts.latitude == nearest.latitude) & (forecasts.longitude == nearest.longitude)
        ]

        # Rename explicitly rather than leaning on merge suffixes, which only
        # apply to columns that actually collide.
        native_part = native[["forecast_target_time", "lead_hours", *variables]].rename(
            columns={v: f"{v}_native" for v in variables}
        )
        interp_part = interpolated[["forecast_target_time", *variables]].rename(
            columns={v: f"{v}_interp" for v in variables}
        )
        observed_part = site[["observation_time", *variables]].rename(
            columns={v: f"{v}_obs" for v in variables}
        )

        merged = native_part.merge(
            observed_part, left_on="forecast_target_time", right_on="observation_time"
        ).merge(interp_part, on="forecast_target_time")

        record = pd.DataFrame(
            {
                "forecast_target_time": merged["forecast_target_time"],
                "lead_hours": merged["lead_hours"],
                "site_lat": site_lat,
                "site_lon": site_lon,
                "site_distance_km": distance,
            }
        )
        for variable in variables:
            native_values = merged[f"{variable}_native"].to_numpy(dtype=float)
            interp_values = merged[f"{variable}_interp"].to_numpy(dtype=float)
            observed = merged[f"{variable}_obs"].to_numpy(dtype=float)
            record[f"{variable}_native_error"] = error_of(native_values, observed, variable)
            record[f"{variable}_interp_error"] = error_of(interp_values, observed, variable)
        frames.append(record)

    table = pd.concat(frames, ignore_index=True)
    table["month"] = pd.to_datetime(table["forecast_target_time"]).dt.month
    return table
