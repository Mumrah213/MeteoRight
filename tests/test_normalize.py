"""Tests for normalization — essential conversion tests only."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from downloader.normalize import (
    DataError,
    json_to_forecast_df,
    json_to_observation_df,
)


def test_basic_observation_conversion():
    """Basic observation JSON → DataFrame conversion."""
    response = {
        "latitude": 55.605,
        "longitude": 13.003,
        "elevation": 35.0,
        "hourly": {
            "time": ["2024-03-01T00:00", "2024-03-01T01:00"],
            "temperature_2m": [3.5, 3.2],
            "precipitation": [0.0, 0.1],
        },
    }
    df = json_to_observation_df(response, ("temperature_2m", "precipitation"))
    assert len(df) == 2
    assert df["observation_time"].iloc[0].tz is not None
    assert df["latitude"].iloc[0] == 55.605
    assert df["temperature_2m"].iloc[0] == 3.5


def test_basic_forecast_conversion():
    """Basic forecast JSON → DataFrame conversion."""
    response = {
        "latitude": 55.605,
        "longitude": 13.003,
        "elevation": 35.0,
        "hourly": {
            "time": ["2024-03-01T00:00", "2024-03-01T06:00", "2024-03-01T12:00"],
            "temperature_2m": [4.0, 5.5, 6.0],
            "precipitation": [0.0, 0.2, 0.0],
        },
    }
    issue_time = datetime(2024, 3, 1, 0, 0, tzinfo=UTC)
    df = json_to_forecast_df(
        response, issue_time, "ecmwf_ifs_hres", ("temperature_2m", "precipitation")
    )

    assert len(df) == 3
    assert df["forecast_issue_time"].iloc[0] == pd.Timestamp("2024-03-01 00:00:00+00:00")
    assert df["lead_hours"].iloc[0] == 0
    assert df["lead_hours"].iloc[1] == 6
    assert df["lead_hours"].iloc[2] == 12
    assert df["model"].iloc[0] == "ecmwf_ifs_hres"


def test_missing_hourly_raises():
    """Missing hourly data raises DataError."""
    response = {"latitude": 55.0, "longitude": 13.0, "elevation": 0.0}
    with pytest.raises(DataError, match="missing"):
        json_to_observation_df(response, ("temperature_2m",))
