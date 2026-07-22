#!/usr/bin/env python3
"""Extract a small, committable grid sample from a full local grid run.

The showcase interpolation figure needs forecasts at several grid points plus
observations at the offset locations they are verified against. A full year of
hourly data across 24 grid points is ~64 MB, which does not belong in git, so
this trims it to a few hundred kilobytes while preserving what the figure
measures: the spatial offset, the grid geometry, and enough samples per site
for a stable MAE.

Run once against the local grid run; the output is committed and
`make_interpolation_figure.py` reads it.

Usage:
    python showcase/make_grid_sample.py [--source DIR] [--output DIR]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path.home() / "Desktop/copenhagen_1y_weekly_50km"
DEFAULT_OUTPUT = REPO_ROOT / "examples/copenhagen_grid_sample"

# Enough lead times to show the flat-with-lead behaviour without carrying all 48.
KEEP_LEADS = (0, 6, 12, 24, 36, 47)
VARIABLES = ("wind_speed_10m", "temperature_2m", "wind_gusts_10m", "wind_direction_10m")

FORECAST_COLUMNS = [
    "forecast_issue_time",
    "forecast_target_time",
    "lead_hours",
    "model",
    "latitude",
    "longitude",
    "elevation",
    *VARIABLES,
]
OBSERVATION_COLUMNS = ["observation_time", "latitude", "longitude", "elevation", *VARIABLES]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tables = sorted(glob.glob(str(args.source / "analysis_store/*/tables")))
    if not tables:
        print(f"No analysis_store tables under {args.source}", file=sys.stderr)
        return 1

    forecasts: list[pd.DataFrame] = []
    observations: list[pd.DataFrame] = []

    for table_dir in tables:
        forecast_path = Path(table_dir) / "forecasts_all_models.parquet"
        observation_path = Path(table_dir) / "observations.parquet"
        if not (forecast_path.exists() and observation_path.exists()):
            continue

        fcst = pd.read_parquet(forecast_path)
        fcst = fcst[fcst["lead_hours"].isin(KEEP_LEADS)]
        forecasts.append(fcst[[c for c in FORECAST_COLUMNS if c in fcst.columns]])

        obs = pd.read_parquet(observation_path).drop_duplicates(subset=["observation_time"])
        observations.append(obs[[c for c in OBSERVATION_COLUMNS if c in obs.columns]])

    if not forecasts:
        print("No usable grid points found", file=sys.stderr)
        return 1

    forecast_df = pd.concat(forecasts, ignore_index=True).drop_duplicates(
        subset=["forecast_issue_time", "forecast_target_time", "latitude", "longitude"]
    )
    observation_df = pd.concat(observations, ignore_index=True)

    # Observations are only needed where a forecast target time exists.
    wanted_times = set(forecast_df["forecast_target_time"])
    observation_df = observation_df[observation_df["observation_time"].isin(wanted_times)]
    observation_df = observation_df.drop_duplicates(
        subset=["observation_time", "latitude", "longitude"]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    forecast_out = args.output / "forecasts.parquet"
    observation_out = args.output / "observations.parquet"
    forecast_df.to_parquet(forecast_out, index=False, compression="zstd")
    observation_df.to_parquet(observation_out, index=False, compression="zstd")

    grid_points = forecast_df[["latitude", "longitude"]].drop_duplicates()
    obs_points = observation_df[["latitude", "longitude"]].drop_duplicates()

    print(f"Wrote {forecast_out} ({forecast_out.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(forecast_df)} rows, {len(grid_points)} grid points, leads {sorted(KEEP_LEADS)}")
    print(f"Wrote {observation_out} ({observation_out.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(observation_df)} rows, {len(obs_points)} observation sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
