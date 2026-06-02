from __future__ import annotations

import pandas as pd

from src.historical.api import previous_run_variable_names
from src.historical.download_planner import (
    DownloadTask,
    LocationSpec,
    _validate_response_location,
    build_download_plan,
)
from src.historical.normalize import json_to_previous_runs_forecast_df
from src.historical.storage import write_forecasts, write_observations


def test_previous_run_variable_names():
    assert previous_run_variable_names(("temperature_2m",), (0, 1, 7)) == (
        "temperature_2m",
        "temperature_2m_previous_day1",
        "temperature_2m_previous_day7",
    )


def test_previous_runs_normalizer_maps_fixed_leads_to_forecast_schema():
    response = {
        "latitude": 55.6,
        "longitude": 12.6,
        "elevation": 10,
        "hourly": {
            "time": ["2025-01-10T00:00", "2025-01-10T01:00"],
            "temperature_2m_previous_day1": [2.0, 2.5],
            "temperature_2m_previous_day3": [1.0, 1.5],
        },
    }

    df = json_to_previous_runs_forecast_df(
        response,
        model="icon_eu",
        variables=("temperature_2m",),
        lead_days=(1, 3),
    )

    assert len(df) == 4
    assert set(df["lead_hours"]) == {24, 72}
    day1 = df[df["lead_hours"] == 24].iloc[0]
    assert day1["forecast_target_time"] == pd.Timestamp("2025-01-10T00:00:00Z")
    assert day1["forecast_issue_time"] == pd.Timestamp("2025-01-09T00:00:00Z")
    assert day1["temperature_2m"] == 2.0
    assert day1["model"] == "icon_eu"


def test_previous_runs_normalizer_preserves_location_provenance():
    response = {
        "latitude": 55.7,
        "longitude": 12.6,
        "elevation": 10,
        "hourly": {
            "time": ["2025-01-10T00:00"],
            "temperature_2m_previous_day1": [2.0],
        },
    }

    df = json_to_previous_runs_forecast_df(
        response,
        model="icon_eu",
        variables=("temperature_2m",),
        lead_days=(1,),
        location_id="copenhagen",
        location_name="Copenhagen",
        requested_latitude=55.6761,
        requested_longitude=12.5683,
        location_distance_km=3.3,
    )

    row = df.iloc[0]
    assert row["location_id"] == "copenhagen"
    assert row["location_name"] == "Copenhagen"
    assert row["requested_latitude"] == 55.6761
    assert row["requested_longitude"] == 12.5683
    assert row["latitude"] == 55.7
    assert row["longitude"] == 12.6
    assert row["location_distance_km"] == 3.3


def test_download_location_validation_rejects_remote_grid_point():
    task = DownloadTask(
        task_id="prev:copenhagen:icon_eu:2025-01-01:2025-01-31:1",
        kind="previous_runs",
        location=LocationSpec("Copenhagen", 55.6761, 12.5683, max_grid_distance_km=50.0),
        start_date="2025-01-01",
        end_date="2025-01-31",
        variables=("temperature_2m",),
        model="icon_eu",
        lead_days=(1,),
    )

    response = {"latitude": 51.5, "longitude": -0.125}

    import pytest

    with pytest.raises(ValueError, match="grid point too far"):
        _validate_response_location(response, task)


def test_build_download_plan_estimates_chunked_requests():
    plan = build_download_plan(
        locations=[
            LocationSpec("Copenhagen", 55.6, 12.6),
            LocationSpec("Stockholm", 59.3, 18.1),
        ],
        start_date="2025-01-01",
        end_date="2025-02-28",
        variables=("temperature_2m", "precipitation"),
        models=("icon_eu", "gfs"),
        kinds=("observations", "previous_runs"),
        lead_days=(1, 2, 3),
        chunk_months=1,
    )

    # 2 locations * 2 monthly chunks * (1 observation + 2 model requests)
    assert plan.request_count == 12
    frame = plan.to_frame()
    assert set(frame["kind"]) == {"observations", "previous_runs"}
    assert frame["task_id"].is_unique


def test_storage_dedup_keeps_multiple_locations(tmp_path):
    obs = pd.DataFrame(
        {
            "observation_time": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:00Z"]),
            "latitude": [55.6, 59.3],
            "longitude": [12.6, 18.1],
            "elevation": [10, 20],
            "temperature_2m": [2.0, -1.0],
        }
    )
    write_observations(obs, tmp_path)
    write_observations(obs, tmp_path)
    stored_obs = pd.read_parquet(tmp_path / "observations/year=2025/month=01/data.parquet")
    assert len(stored_obs) == 2
    assert set(stored_obs["latitude"]) == {55.6, 59.3}

    fcst = pd.DataFrame(
        {
            "forecast_issue_time": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:00Z"]),
            "forecast_target_time": pd.to_datetime(["2025-01-02T00:00Z", "2025-01-02T00:00Z"]),
            "lead_hours": [24, 24],
            "model": ["icon_eu", "icon_eu"],
            "latitude": [55.6, 59.3],
            "longitude": [12.6, 18.1],
            "elevation": [10, 20],
            "temperature_2m": [3.0, 0.0],
        }
    )
    write_forecasts(fcst, tmp_path)
    write_forecasts(fcst, tmp_path)
    stored_fcst = pd.read_parquet(
        tmp_path / "forecasts/model=icon_eu/year=2025/month=01/data.parquet"
    )
    assert len(stored_fcst) == 2
    assert set(stored_fcst["latitude"]) == {55.6, 59.3}


def test_storage_dedup_uses_location_id_before_grid_coordinates(tmp_path):
    obs = pd.DataFrame(
        {
            "observation_time": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:00Z"]),
            "location_id": ["copenhagen", "stockholm"],
            "latitude": [55.5, 55.5],
            "longitude": [12.5, 12.5],
            "elevation": [10, 10],
            "temperature_2m": [2.0, -1.0],
        }
    )
    write_observations(obs, tmp_path)
    write_observations(obs, tmp_path)
    stored_obs = pd.read_parquet(tmp_path / "observations/year=2025/month=01/data.parquet")
    assert len(stored_obs) == 2
    assert set(stored_obs["location_id"]) == {"copenhagen", "stockholm"}

    fcst = pd.DataFrame(
        {
            "forecast_issue_time": pd.to_datetime(["2025-01-01T00:00Z", "2025-01-01T00:00Z"]),
            "forecast_target_time": pd.to_datetime(["2025-01-02T00:00Z", "2025-01-02T00:00Z"]),
            "lead_hours": [24, 24],
            "model": ["icon_eu", "icon_eu"],
            "location_id": ["copenhagen", "stockholm"],
            "latitude": [55.5, 55.5],
            "longitude": [12.5, 12.5],
            "elevation": [10, 10],
            "temperature_2m": [3.0, 0.0],
        }
    )
    write_forecasts(fcst, tmp_path)
    write_forecasts(fcst, tmp_path)
    stored_fcst = pd.read_parquet(
        tmp_path / "forecasts/model=icon_eu/year=2025/month=01/data.parquet"
    )
    assert len(stored_fcst) == 2
    assert set(stored_fcst["location_id"]) == {"copenhagen", "stockholm"}
