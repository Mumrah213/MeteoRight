"""Visualization tests: verify data can be plotted without errors.

These tests download data via the API and create plots to verify matplotlib can
consume the data. They do NOT check visual quality.

Run with: METEORIGHT_RUN_NETWORK_TESTS=1 python -m pytest tests/test_visualize.py
"""

import os
import warnings
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for CI
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from downloader.api import fetch_observations, fetch_single_run
from downloader.normalize import json_to_forecast_df, json_to_observation_df

RUN_NETWORK_TESTS = os.environ.get("METEORIGHT_RUN_NETWORK_TESTS", "0") == "1"


@pytest.mark.skipif(not RUN_NETWORK_TESTS, reason="live Open-Meteo visualization test")
class TestVisualization:
    """Tests that verify plotting works with downloaded data."""

    def test_plot_temperature_over_time(self, tmp_path):
        """Plot temperature_2m over time for observations."""
        response = fetch_observations(
            lat=55.605,
            lon=13.003,
            start_date="2024-03-01",
            end_date="2024-03-01",
            variables=("temperature_2m", "precipitation"),
        )
        df = json_to_observation_df(response, ("temperature_2m", "precipitation"))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["observation_time"], df["temperature_2m"], marker=".", markersize=3)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Observation Temperature Over Time")
        fig.savefig(tmp_path / "temp_obs.png", dpi=72)
        plt.close(fig)

        assert (tmp_path / "temp_obs.png").exists()
        assert (tmp_path / "temp_obs.png").stat().st_size > 1000

    def test_plot_forecast_vs_observation(self, tmp_path):
        """Plot forecast and observation temperature on the same chart."""
        # Download observations
        obs_response = fetch_observations(
            lat=55.605,
            lon=13.003,
            start_date="2024-03-01",
            end_date="2024-03-01",
            variables=("temperature_2m", "precipitation"),
        )
        obs_df = json_to_observation_df(obs_response, ("temperature_2m", "precipitation"))

        # Download one forecast run
        run_time = datetime(2024, 3, 1, 0, 0, tzinfo=UTC)
        fcst_response = fetch_single_run(
            lat=55.605,
            lon=13.003,
            run_time=run_time,
            variables=("temperature_2m", "precipitation"),
        )
        fcst_df = json_to_forecast_df(
            fcst_response, run_time, "ecmwf_ifs_hres", ("temperature_2m", "precipitation")
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot observations
        ax.plot(
            obs_df["observation_time"],
            obs_df["temperature_2m"],
            marker=".",
            markersize=4,
            label="Observations",
            color="black",
        )

        # Plot forecast run
        ax.plot(
            fcst_df["forecast_target_time"],
            fcst_df["temperature_2m"],
            marker="x",
            markersize=6,
            label=f"Forecast (issued {run_time.strftime('%m-%d %H:%M')})",
            color="blue",
            alpha=0.7,
        )

        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Forecast vs Observation: Temperature")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(tmp_path / "fcst_vs_obs.png", dpi=72)
        plt.close(fig)

        assert (tmp_path / "fcst_vs_obs.png").exists()
        assert (tmp_path / "fcst_vs_obs.png").stat().st_size > 1000

    def test_plot_multiple_forecast_runs(self, tmp_path):
        """Plot multiple forecast runs for the same target time."""
        runs = [
            datetime(2024, 3, 14, 0, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 6, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 12, 0, tzinfo=UTC),
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

        all_columns = sorted(set().union(*(set(df.columns) for df in dfs)))
        dfs = [df.reindex(columns=all_columns, fill_value=float("nan")) for df in dfs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            combined = pd.concat(dfs, ignore_index=True)

        # Pick a target time with multiple runs
        target = pd.Timestamp("2024-03-14 12:00:00", tz="UTC")
        mask = combined["forecast_target_time"] == target
        runs_for_target = combined[mask]

        if len(runs_for_target) < 2:
            pytest.skip("Not enough forecast runs for this target time")

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["red", "green", "orange", "purple", "brown"]
        for i, (_, run) in enumerate(runs_for_target.iterrows()):
            ax.plot(
                [run["forecast_issue_time"], run["forecast_target_time"]],
                [run["temperature_2m"], run["temperature_2m"]],
                marker="o",
                color=colors[i % len(colors)],
                alpha=0.6,
                label=f"Lead {run['lead_hours']}h",
            )

        ax.set_xlabel("Forecast Issue Time")
        ax.set_ylabel("Predicted Temperature at Target (°C)")
        ax.set_title(f"Multiple Forecasts for {target.strftime('%m-%d %H:%M')} UTC")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(tmp_path / "multiple_forecasts.png", dpi=72)
        plt.close(fig)

        assert (tmp_path / "multiple_forecasts.png").exists()
        assert (tmp_path / "multiple_forecasts.png").stat().st_size > 1000

    def test_plot_temperature_histogram(self, tmp_path):
        """Plot a histogram of temperature distribution."""
        response = fetch_observations(
            lat=55.605,
            lon=13.003,
            start_date="2024-03-01",
            end_date="2024-03-07",
            variables=("temperature_2m",),
        )
        df = json_to_observation_df(response, ("temperature_2m",))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["temperature_2m"].dropna(), bins=15, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Count")
        ax.set_title("Temperature Distribution (7 days)")
        fig.savefig(tmp_path / "temp_histogram.png", dpi=72)
        plt.close(fig)

        assert (tmp_path / "temp_histogram.png").exists()
        assert (tmp_path / "temp_histogram.png").stat().st_size > 1000

    def test_plot_lead_time_error(self, tmp_path):
        """Plot forecast error vs lead time (requires both forecast and obs)."""
        # Download observations for March 14
        obs_response = fetch_observations(
            lat=55.605,
            lon=13.003,
            start_date="2024-03-14",
            end_date="2024-03-14",
            variables=("temperature_2m", "precipitation"),
        )
        obs_df = json_to_observation_df(obs_response, ("temperature_2m", "precipitation"))

        # Download 3 forecast runs
        runs = [
            datetime(2024, 3, 14, 0, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 6, 0, tzinfo=UTC),
            datetime(2024, 3, 14, 12, 0, tzinfo=UTC),
        ]

        fcst_dfs = []
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
            fcst_dfs.append(df)

        all_columns = sorted(set().union(*(set(df.columns) for df in fcst_dfs)))
        fcst_dfs = [df.reindex(columns=all_columns, fill_value=float("nan")) for df in fcst_dfs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fcst_combined = pd.concat(fcst_dfs, ignore_index=True)

        # Merge forecast with observation at matching target times
        obs_renamed = obs_df[["observation_time", "temperature_2m"]].rename(
            columns={"observation_time": "forecast_target_time", "temperature_2m": "obs_temp"}
        )
        merged = fcst_combined[["forecast_target_time", "lead_hours", "temperature_2m"]].merge(
            obs_renamed, on="forecast_target_time", how="inner"
        )

        if merged.empty:
            pytest.skip("No matching forecast/observation times")

        fig, ax = plt.subplots(figsize=(10, 5))
        grouped = (
            merged.groupby("lead_hours")[["temperature_2m", "obs_temp"]]
            .apply(lambda x: abs(x["temperature_2m"] - x["obs_temp"]).mean())
            .reset_index()
        )
        grouped.columns = ["lead_hours", "mae"]

        ax.plot(grouped["lead_hours"], grouped["mae"], marker="o", linewidth=2)
        ax.set_xlabel("Lead Time (hours)")
        ax.set_ylabel("Mean Absolute Error (°C)")
        ax.set_title("Forecast Skill: MAE vs Lead Time")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(tmp_path / "skill_curve.png", dpi=72)
        plt.close(fig)

        assert (tmp_path / "skill_curve.png").exists()
        assert (tmp_path / "skill_curve.png").stat().st_size > 1000
