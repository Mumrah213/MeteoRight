"""Integration tests that call the real Open-Meteo API.

Run with: METEORIGHT_RUN_NETWORK_TESTS=1 python -m pytest tests/test_integration.py
"""

import os
import warnings
from datetime import UTC, datetime

import pandas as pd
import pytest

from downloader.api import fetch_observations, fetch_single_run
from downloader.normalize import json_to_forecast_df, json_to_observation_df
from downloader.storage import list_downloaded, write_forecasts, write_observations

RUN_NETWORK_TESTS = os.environ.get("METEORIGHT_RUN_NETWORK_TESTS", "0") == "1"


@pytest.mark.skipif(not RUN_NETWORK_TESTS, reason="live Open-Meteo integration test")
class TestRealObservationDownload:
    """Test downloading real observation data."""

    def test_fetch_observations(self):
        response = fetch_observations(
            lat=55.605,
            lon=13.003,
            start_date="2024-03-01",
            end_date="2024-03-01",
            variables=("temperature_2m", "precipitation"),
        )
        assert "hourly" in response
        assert "time" in response["hourly"]
        assert len(response["hourly"]["time"]) == 24  # one day

    def test_full_pipeline(self, tmp_path):
        """Download → normalize → store → read back."""
        response = fetch_observations(
            lat=55.605,
            lon=13.003,
            start_date="2024-03-01",
            end_date="2024-03-01",
            variables=("temperature_2m", "precipitation"),
        )
        df = json_to_observation_df(response, ("temperature_2m", "precipitation"))
        assert len(df) == 24

        output_dir = tmp_path / "test_obs"
        rows_written = write_observations(df, output_dir, append=False)
        assert rows_written == 24

        # Read back
        downloaded = list_downloaded(output_dir, "observations")
        assert len(downloaded) == 1
        assert downloaded[0]["row_count"] == 24

        read_df = pd.read_parquet(downloaded[0]["path"])
        assert len(read_df) == 24
        assert "temperature_2m" in read_df.columns


@pytest.mark.skipif(not RUN_NETWORK_TESTS, reason="live Open-Meteo integration test")
class TestRealForecastDownload:
    """Test downloading real forecast data."""

    def test_fetch_single_run(self):
        response = fetch_single_run(
            lat=55.605,
            lon=13.003,
            run_time=datetime(2024, 3, 14, 0, 0, tzinfo=UTC),
            variables=("temperature_2m", "precipitation"),
        )
        assert "hourly" in response
        assert "time" in response["hourly"]

    def test_full_pipeline(self, tmp_path):
        """Download multiple runs → normalize → store → read back."""
        runs = [
            datetime(2024, 3, 14, 0, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 6, 0, tzinfo=UTC),
        ]
        dfs = []
        for run_time in runs:
            response = fetch_single_run(
                lat=55.605,
                lon=13.003,
                run_time=run_time,
                variables=("temperature_2m", "precipitation"),
            )
            df = json_to_forecast_df(
                response, run_time, "ecmwf_ifs_hres", ("temperature_2m", "precipitation")
            )
            dfs.append(df)

        # Ensure consistent columns before concat
        all_columns = sorted(set().union(*(set(df.columns) for df in dfs)))
        dfs = [df.reindex(columns=all_columns, fill_value=float("nan")) for df in dfs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            combined = pd.concat(dfs, ignore_index=True)
        assert len(combined) > 0

        output_dir = tmp_path / "test_fcst"
        rows_written = write_forecasts(combined, output_dir, append=False)
        assert rows_written > 0

        # Verify multiple issue times preserved for same target
        downloaded = list_downloaded(output_dir, "forecasts")
        assert len(downloaded) >= 1

        read_df = pd.read_parquet(downloaded[0]["path"])
        # Should have different issue times
        assert read_df["forecast_issue_time"].nunique() >= 2
