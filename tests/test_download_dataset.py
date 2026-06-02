"""Download and verify a specific, well-known dataset.

This test downloads ECMWF IFS HRES forecasts and ERA5 observations for
Copenhagen (55.6761°N, 12.5683°E) for March 14-15, 2024 — a date with
known weather patterns (early spring transition).

Run with: METEORIGHT_RUN_NETWORK_TESTS=1 python -m pytest tests/test_download_dataset.py
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

# Copenhagen coordinates
COPENHAGEN_LAT = 55.6761
COPENHAGEN_LON = 12.5683


@pytest.mark.skipif(not RUN_NETWORK_TESTS, reason="live Open-Meteo dataset download test")
class TestCopenhagenMarch2024:
    """Download and verify the Copenhagen March 2024 dataset."""

    def test_download_observations(self, tmp_path):
        """Download ERA5 observations for Copenhagen, March 1-7, 2024."""
        response = fetch_observations(
            lat=COPENHAGEN_LAT,
            lon=COPENHAGEN_LON,
            start_date="2024-03-01",
            end_date="2024-03-07",
            variables=("temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover"),
        )

        df = json_to_observation_df(
            response, ("temperature_2m", "precipitation", "wind_speed_10m", "cloud_cover")
        )

        # Verify structure
        assert len(df) == 7 * 24  # 7 days × 24 hours
        assert "temperature_2m" in df.columns
        assert "precipitation" in df.columns
        assert df["observation_time"].min().date() == pd.Timestamp("2024-03-01").date()
        assert df["observation_time"].max().date() == pd.Timestamp("2024-03-07").date()

        # Store
        output_dir = tmp_path / "copenhagen_obs"
        rows = write_observations(df, output_dir)
        assert rows == 7 * 24

        # Verify stored data
        downloaded = list_downloaded(output_dir, "observations")
        assert len(downloaded) == 1
        stored = pd.read_parquet(downloaded[0]["path"])
        assert len(stored) == 7 * 24

    def test_download_forecasts(self, tmp_path):
        """Download ECMWF IFS HRES forecasts for Copenhagen, March 14-15, 2024."""
        runs = [
            datetime(2024, 3, 14, 0, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 6, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 12, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 18, 0, tzinfo=UTC),
        ]

        dfs = []
        for run_time in runs:
            response = fetch_single_run(
                lat=COPENHAGEN_LAT,
                lon=COPENHAGEN_LON,
                run_time=run_time,
                variables=("temperature_2m", "precipitation"),
            )
            df = json_to_forecast_df(
                response, run_time, "ecmwf_ifs_hres", ("temperature_2m", "precipitation")
            )
            dfs.append(df)

        # Ensure consistent columns
        import warnings

        all_columns = sorted(set().union(*(set(df.columns) for df in dfs)))
        dfs = [df.reindex(columns=all_columns, fill_value=float("nan")) for df in dfs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            combined = pd.concat(dfs, ignore_index=True)

        # Verify structure
        assert len(combined) > 0
        assert combined["model"].iloc[0] == "ecmwf_ifs_hres"
        assert combined["forecast_issue_time"].nunique() == 4  # 4 runs

        # Store
        output_dir = tmp_path / "copenhagen_fcst"
        rows = write_forecasts(combined, output_dir)
        assert rows > 0

        # Verify stored data
        downloaded = list_downloaded(output_dir, "forecasts")
        assert len(downloaded) >= 1
        stored = pd.read_parquet(downloaded[0]["path"])
        assert len(stored) == rows
        assert stored["forecast_issue_time"].nunique() == 4

    def test_forecasts_preserve_all_issue_times(self, tmp_path):
        """Verify that multiple issue times for the same target time are preserved."""
        runs = [
            datetime(2024, 3, 14, 0, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 6, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 12, 0, tzinfo=UTC),
        ]

        dfs = []
        for run_time in runs:
            response = fetch_single_run(
                lat=COPENHAGEN_LAT,
                lon=COPENHAGEN_LON,
                run_time=run_time,
                variables=("temperature_2m",),
            )
            df = json_to_forecast_df(response, run_time, "ecmwf_ifs_hres", ("temperature_2m",))
            dfs.append(df)

        all_columns = sorted(set().union(*(set(df.columns) for df in dfs)))
        dfs = [df.reindex(columns=all_columns, fill_value=float("nan")) for df in dfs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            combined = pd.concat(dfs, ignore_index=True)

        output_dir = tmp_path / "copenhagen_issue_times"
        write_forecasts(combined, output_dir)

        # Read back and verify
        downloaded = list_downloaded(output_dir, "forecasts")
        stored = pd.read_parquet(downloaded[0]["path"])

        # Pick a target time that should have multiple forecasts
        target = pd.Timestamp("2024-03-14 12:00:00", tz="UTC")
        mask = stored["forecast_target_time"] == target
        rows_for_target = stored[mask]

        # Should have at least 2 different issue times for this target
        assert rows_for_target["forecast_issue_time"].nunique() >= 2, (
            f"Expected multiple issue times for target {target}, got {rows_for_target['forecast_issue_time'].nunique()}"
        )

    def test_data_ranges_are_reasonable(self, tmp_path):
        """Verify that downloaded data has physically reasonable ranges."""
        # Download observations
        response = fetch_observations(
            lat=COPENHAGEN_LAT,
            lon=COPENHAGEN_LON,
            start_date="2024-03-01",
            end_date="2024-03-03",
            variables=("temperature_2m", "precipitation"),
        )
        df = json_to_observation_df(response, ("temperature_2m", "precipitation"))

        # Temperature should be in a reasonable range for Copenhagen in March
        temps = df["temperature_2m"].dropna()
        assert temps.min() > -30, f"Temperature too low: {temps.min()}"
        assert temps.max() < 50, f"Temperature too high: {temps.max()}"

        # Precipitation should be non-negative
        precip = df["precipitation"].dropna()
        assert (precip >= 0).all(), f"Negative precipitation: {precip.min()}"

    def test_full_forecast_vs_observation_comparison(self, tmp_path):
        """Download both forecasts and observations, merge, and verify comparison is possible."""
        # Download observations for March 14-15
        obs_response = fetch_observations(
            lat=COPENHAGEN_LAT,
            lon=COPENHAGEN_LON,
            start_date="2024-03-14",
            end_date="2024-03-15",
            variables=("temperature_2m", "precipitation"),
        )
        obs_df = json_to_observation_df(obs_response, ("temperature_2m", "precipitation"))

        # Download forecasts for March 14
        runs = [
            datetime(2024, 3, 14, 0, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 6, 0, tzinfo=UTC),
        ]
        fcst_dfs = []
        for run_time in runs:
            response = fetch_single_run(
                lat=COPENHAGEN_LAT,
                lon=COPENHAGEN_LON,
                run_time=run_time,
                variables=("temperature_2m", "precipitation"),
            )
            df = json_to_forecast_df(
                response, run_time, "ecmwf_ifs_hres", ("temperature_2m", "precipitation")
            )
            fcst_dfs.append(df)

        all_columns = sorted(set().union(*(set(df.columns) for df in fcst_dfs)))
        fcst_dfs = [df.reindex(columns=all_columns, fill_value=float("nan")) for df in fcst_dfs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fcst_combined = pd.concat(fcst_dfs, ignore_index=True)

        # Store both
        obs_dir = tmp_path / "comparison_obs"
        fcst_dir = tmp_path / "comparison_fcst"
        write_observations(obs_df, obs_dir)
        write_forecasts(fcst_combined, fcst_dir)

        # Merge on target time
        obs_renamed = obs_df[["observation_time", "temperature_2m"]].rename(
            columns={"observation_time": "forecast_target_time", "temperature_2m": "obs_temp"}
        )
        merged = fcst_combined[
            ["forecast_target_time", "forecast_issue_time", "lead_hours", "temperature_2m"]
        ].merge(obs_renamed, on="forecast_target_time", how="inner")

        # Should have matches
        assert len(merged) > 0, "No matching forecast/observation times"

        # Compute error
        merged["mae_per_hour"] = abs(merged["temperature_2m"] - merged["obs_temp"])
        overall_mae = merged["mae_per_hour"].mean()

        # MAE should be in a reasonable range for a 0-6 hour lead time
        assert 0 < overall_mae < 10, f"Unreasonable MAE: {overall_mae}"
