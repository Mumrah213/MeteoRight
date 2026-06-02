"""Normalize raw API JSON responses into pandas DataFrames.

All timestamps are converted to UTC. Null values are preserved as None.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataError(Exception):
    """Raised when normalized data fails validation."""


def validate_array_lengths(hourly_data: dict) -> None:
    """Validate that all arrays in the hourly response have the same length as the time array.

    Args:
        hourly_data: The 'hourly' object from an Open-Meteo API response.

    Raises:
        DataError: If arrays have mismatched lengths.
    """
    time_len = len(hourly_data.get("time", []))
    if time_len == 0:
        return

    for key, values in hourly_data.items():
        if key == "time":
            continue
        if len(values) != time_len:
            raise DataError(
                f"Array length mismatch: 'time' has {time_len} entries, "
                f"'{key}' has {len(values)} entries"
            )


def validate_timestamps(times: list[str]) -> None:
    """Validate that timestamps are valid and non-empty.

    Args:
        times: List of ISO 8601 timestamp strings.

    Raises:
        DataError: If timestamps are empty or unparseable.
    """
    if not times:
        raise DataError("Empty timestamp array")

    parsed = []
    for i, t in enumerate(times):
        try:
            # API returns ISO 8601 without timezone — treat as UTC
            dt = datetime.fromisoformat(t).replace(tzinfo=UTC)
            parsed.append(dt)
        except ValueError as e:
            raise DataError(f"Invalid timestamp at index {i}: '{t}' — {e}") from None

    # Check for duplicates within a single run
    if len(parsed) != len(set(parsed)):
        raise DataError(
            f"Duplicate timestamps found in response ({len(parsed)} total, {len(set(parsed))} unique)"
        )


def _parse_timestamps(times: list[str]) -> pd.Series:
    """Parse ISO 8601 timestamps to UTC datetime64."""
    return pd.to_datetime(times, utc=True)


def _melt_variables(
    df: pd.DataFrame,
    hourly_data: dict,
    variables: tuple[str, ...],
) -> pd.DataFrame:
    """Add variable columns from hourly data to the DataFrame.

    Args:
        df: DataFrame with timestamp column(s) already set.
        hourly_data: The 'hourly' object from the API response.
        variables: Which variables to include.

    Returns:
        DataFrame with variable columns added.
    """
    for var in variables:
        if var in hourly_data:
            df[var] = hourly_data[var]
        else:
            logger.warning("Variable '%s' not found in API response", var)
            df[var] = None
    return df


def _add_location_provenance(
    df: pd.DataFrame,
    *,
    location_id: str | None = None,
    location_name: str | None = None,
    requested_latitude: float | None = None,
    requested_longitude: float | None = None,
    location_distance_km: float | None = None,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Attach requested-location metadata to normalized rows."""
    if location_id is not None:
        df["location_id"] = location_id
    if location_name is not None:
        df["location_name"] = location_name
    if requested_latitude is not None:
        df["requested_latitude"] = requested_latitude
    if requested_longitude is not None:
        df["requested_longitude"] = requested_longitude
    if location_distance_km is not None:
        df["location_distance_km"] = location_distance_km
    if extra:
        for key, value in extra.items():
            df[key] = value
    return df


def json_to_observation_df(
    response: dict,
    variables: tuple[str, ...],
    *,
    location_id: str | None = None,
    location_name: str | None = None,
    requested_latitude: float | None = None,
    requested_longitude: float | None = None,
    location_distance_km: float | None = None,
    location_extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Convert an Archive API response to an observation DataFrame.

    Args:
        response: Full JSON response from the Archive API.
        variables: Tuple of variable names to extract.

    Returns:
        DataFrame with columns: observation_time, latitude, longitude, elevation, plus variables.

    Raises:
        DataError: If the response is malformed.
    """
    hourly = response.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise DataError("Response missing 'hourly.time' array")

    validate_array_lengths(hourly)
    times = hourly["time"]
    validate_timestamps(times)

    df = pd.DataFrame(
        {
            "observation_time": _parse_timestamps(times),
            "latitude": response.get("latitude"),
            "longitude": response.get("longitude"),
            "elevation": response.get("elevation"),
        }
    )

    df = _add_location_provenance(
        df,
        location_id=location_id,
        location_name=location_name,
        requested_latitude=requested_latitude,
        requested_longitude=requested_longitude,
        location_distance_km=location_distance_km,
        extra=location_extra,
    )

    return _melt_variables(df, hourly, variables)


def json_to_forecast_df(
    response: dict,
    issue_time: datetime,
    model: str,
    variables: tuple[str, ...],
    *,
    location_id: str | None = None,
    location_name: str | None = None,
    requested_latitude: float | None = None,
    requested_longitude: float | None = None,
    location_distance_km: float | None = None,
    location_extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Convert a Single Runs API response to a forecast DataFrame.

    Args:
        response: Full JSON response from the Single Runs API.
        issue_time: The model initialization time (UTC).
        model: Model name (e.g. "ecmwf_ifs_hres").
        variables: Tuple of variable names to extract.

    Returns:
        DataFrame with columns: forecast_issue_time, forecast_target_time, lead_hours,
        model, latitude, longitude, elevation, plus variables.

    Raises:
        DataError: If the response is malformed.
    """
    hourly = response.get("hourly", {})
    if not hourly or "time" not in hourly:
        raise DataError("Response missing 'hourly.time' array")

    validate_array_lengths(hourly)
    times = hourly["time"]
    validate_timestamps(times)

    target_times = _parse_timestamps(times)
    issue_dt = pd.Timestamp(issue_time)

    lead_hours = ((target_times - issue_dt) / pd.Timedelta(hours=1)).astype("int32")

    df = pd.DataFrame(
        {
            "forecast_issue_time": issue_dt,
            "forecast_target_time": target_times,
            "lead_hours": lead_hours,
            "model": model,
            "latitude": response.get("latitude"),
            "longitude": response.get("longitude"),
            "elevation": response.get("elevation"),
        }
    )

    df = _add_location_provenance(
        df,
        location_id=location_id,
        location_name=location_name,
        requested_latitude=requested_latitude,
        requested_longitude=requested_longitude,
        location_distance_km=location_distance_km,
        extra=location_extra,
    )

    return _melt_variables(df, hourly, variables)
