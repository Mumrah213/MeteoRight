"""Tests for location safety invariants in verification joins."""

import pandas as pd
import pytest

from verification.location import validate_location_compatibility
from src.analysis.alignment import align


def test_accepts_nearby_single_grid_points():
    forecasts = pd.DataFrame({"latitude": [55.6785], "longitude": [12.570435]})
    observations = pd.DataFrame({"latitude": [55.641476], "longitude": [12.596349]})

    audit = validate_location_compatibility(forecasts, observations)

    assert audit.forecast_locations == 1
    assert audit.observation_locations == 1
    assert audit.max_nearest_distance_km < 10


def test_rejects_different_cities():
    forecasts = pd.DataFrame({"latitude": [51.5], "longitude": [-0.125]})
    observations = pd.DataFrame({"latitude": [55.641476], "longitude": [12.596349]})

    with pytest.raises(ValueError, match="different locations"):
        validate_location_compatibility(forecasts, observations)


def test_rejects_missing_location_columns():
    forecasts = pd.DataFrame({"forecast_target_time": pd.to_datetime(["2024-01-01"])})
    observations = pd.DataFrame({"latitude": [55.6], "longitude": [12.6]})

    with pytest.raises(ValueError, match="missing required location columns"):
        validate_location_compatibility(forecasts, observations)


def test_rejects_multi_location_without_location_id():
    forecasts = pd.DataFrame(
        {
            "latitude": [55.6, 59.3],
            "longitude": [12.6, 18.1],
        }
    )
    observations = pd.DataFrame(
        {
            "latitude": [55.6, 59.3],
            "longitude": [12.6, 18.1],
        }
    )

    with pytest.raises(ValueError, match="location-ambiguous"):
        validate_location_compatibility(forecasts, observations)


def test_accepts_multi_location_with_location_id():
    forecasts = pd.DataFrame(
        {
            "location_id": ["copenhagen", "stockholm"],
            "latitude": [55.6, 59.3],
            "longitude": [12.6, 18.1],
        }
    )
    observations = pd.DataFrame(
        {
            "location_id": ["copenhagen", "stockholm"],
            "latitude": [55.6, 59.3],
            "longitude": [12.6, 18.1],
        }
    )

    audit = validate_location_compatibility(forecasts, observations)

    assert audit.has_location_id is True
    assert audit.forecast_locations == 2


def test_analysis_alignment_rejects_wrong_location_time_match():
    forecasts = pd.DataFrame(
        {
            "forecast_issue_time": pd.to_datetime(["2024-01-01T00:00Z"]),
            "forecast_target_time": pd.to_datetime(["2024-01-02T00:00Z"]),
            "lead_hours": [24],
            "model": ["icon_eu"],
            "latitude": [51.5],
            "longitude": [-0.125],
            "variable": ["temperature_2m"],
            "value": [15.0],
        }
    )
    observations = pd.DataFrame(
        {
            "observation_time": pd.to_datetime(["2024-01-02T00:00Z"]),
            "latitude": [55.641476],
            "longitude": [12.596349],
            "variable": ["temperature_2m"],
            "value": [13.0],
        }
    )

    with pytest.raises(ValueError, match="different locations"):
        align(forecasts, observations)
