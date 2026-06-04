"""Timestamp alignment: LEFT JOIN forecasts to observations.

Joins on forecast_target_time == observation_time with location matching.
All forecast rows survive — unmatched rows get NaN in observed columns.
Both forecast_target_time and observation_time are preserved.
"""

import logging

import pandas as pd

from .location import validate_location_compatibility
from .schema import forecast_var, observed_var

logger = logging.getLogger(__name__)


def _validate_location_match(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[float, float]:
    """Ensure forecast and observation locations match.

    Args:
        forecasts: Forecast DataFrame.
        observations: Observation DataFrame.

    Returns:
        (latitude, longitude) pair that both datasets share.

    Raises:
        ValueError: If locations differ between forecast and observation sets.
    """
    audit = validate_location_compatibility(
        forecasts,
        observations,
        allow_multi_location_without_id=True,
    )
    loc = (float(forecasts["latitude"].iloc[0]), float(forecasts["longitude"].iloc[0]))
    logger.info(
        "  Location audit: %d forecast location(s), %d observation location(s), "
        "max nearest distance %.1f km",
        audit.forecast_locations,
        audit.observation_locations,
        audit.max_nearest_distance_km,
    )
    return loc


def align_forecasts_with_observations(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    variables: tuple[str, ...],
) -> pd.DataFrame:
    """LEFT JOIN forecasts to observations on target_time == observation_time.

    Invariant: (forecast_issue_time, forecast_target_time, model) is preserved
    for every forecast row. ALL forecast rows survive — rows with no matching
    observation get NaN in observed_* columns.

    Both forecast_target_time and observation_time are preserved as separate
    columns. For matched rows they are identical; for unmatched forecasts,
    observation_time is NaN.

    Args:
        forecasts: Forecast DataFrame from load_forecasts().
        observations: Observation DataFrame from load_observations().
        variables: Tuple of variable names to align (e.g. ("temperature_2m",)).

    Returns:
        DataFrame with columns in this order:
          forecast_issue_time, forecast_target_time, observation_time,
          lead_hours, model, latitude, longitude, elevation,
          forecast_{var} for each var,
          observed_{var} for each var

    Raises:
        ValueError: If locations don't match between forecast and observation sets.
    """
    # Validate locations match
    _validate_location_match(forecasts, observations)

    # Select observation columns (raw names) then rename
    obs_select = ["observation_time", "latitude", "longitude", "elevation"]
    if "location_id" in observations.columns:
        obs_select.append("location_id")
    obs_select += list(variables)
    obs_rename = {v: observed_var(v) for v in variables}
    obs_df = observations[obs_select].rename(columns=obs_rename)

    # Select forecast columns (raw names) then rename
    f_cols = [
        "forecast_issue_time",
        "forecast_target_time",
        "lead_hours",
        "model",
        "latitude",
        "longitude",
        "elevation",
    ]
    if "location_id" in forecasts.columns:
        f_cols.append("location_id")
    f_cols += list(variables)

    f_rename = {v: forecast_var(v) for v in variables}
    f_df = forecasts[f_cols].rename(columns=f_rename)

    left_keys = ["forecast_target_time"]
    right_keys = ["observation_time"]
    if "location_id" in f_df.columns and "location_id" in obs_df.columns:
        left_keys.append("location_id")
        right_keys.append("location_id")
    elif (
        forecasts[["latitude", "longitude"]].drop_duplicates().shape[0] > 1
        or observations[["latitude", "longitude"]].drop_duplicates().shape[0] > 1
    ):
        left_keys += ["latitude", "longitude", "elevation"]
        right_keys += ["latitude", "longitude", "elevation"]

    merged = pd.merge(f_df, obs_df, left_on=left_keys, right_on=right_keys, how="left")

    # Normalize merge suffixes back to the canonical forecast-location columns.
    for col in ("latitude", "longitude", "elevation", "location_id"):
        left_col = f"{col}_x"
        right_col = f"{col}_y"
        if left_col in merged.columns:
            merged[col] = merged[left_col]
            merged = merged.drop(columns=[left_col])
        elif col not in merged.columns:
            merged[col] = pd.NA
        if right_col in merged.columns:
            merged = merged.drop(columns=[right_col])

    # Ensure all observation columns are nullable (for NaN handling)
    for var in variables:
        col = observed_var(var)
        if col not in merged.columns:
            merged[col] = pd.NA
        else:
            # Convert to float with NA support if not already
            if merged[col].dtype != "float64":
                merged[col] = merged[col].astype("float64")

    logger.info(
        "Aligned: %d forecasts → %d rows (all forecasts preserved, %d have matching observations)",
        len(forecasts),
        len(merged),
        merged["observation_time"].notna().sum(),
    )

    return merged
