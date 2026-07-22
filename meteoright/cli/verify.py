"""`meteoright verify` — join forecasts with observations into verification rows."""

from __future__ import annotations

import argparse

import pandas as pd

from ._shared import console


def cmd_verify(args: argparse.Namespace) -> int:
    """Build verification dataset by joining forecasts with observations."""
    from pathlib import Path

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = [v.strip() for v in args.variables.split(",")]

    console.print("[bold]Building verification dataset[/bold]")

    # Load forecasts
    fcst_files = list(data_dir.glob("forecasts/model=*/year=*/month=*/data.parquet"))
    if not fcst_files:
        console.print("[red]No forecast files found[/red]")
        return 1

    forecasts = pd.concat([pd.read_parquet(f) for f in fcst_files], ignore_index=True)
    console.print(f"  Loaded {len(forecasts)} forecast rows from {len(fcst_files)} files")

    # Load observations
    obs_files = list(data_dir.glob("observations/year=*/month=*/data.parquet"))
    if not obs_files:
        console.print("[red]No observation files found[/red]")
        return 1

    observations = pd.concat([pd.read_parquet(f) for f in obs_files], ignore_index=True)
    console.print(f"  Loaded {len(observations)} observation rows from {len(obs_files)} files")

    method = getattr(args, "interpolate", "none")
    if method != "none":
        from verification.interpolate_frames import interpolate_multi_site

        forecasts, report = interpolate_multi_site(
            forecasts, observations, variables, method=method
        )
        if report.grid_points < 4:
            console.print(
                f"[yellow]Only {report.grid_points} forecast grid point(s); "
                "interpolation needs a grid. Falling back to native values.[/yellow]"
            )
        else:
            console.print(
                f"  Interpolated ({method}): {report.grid_points} grid points → "
                f"{report.target_points} observation point(s), "
                f"mean offset {report.mean_offset_km:.2f} km"
            )

    # Multi-location datasets must join on location as well as time, or every
    # site's forecast matches every site's observation and the row count fans
    # out by the number of locations.
    by_location = "location_id" in forecasts.columns and "location_id" in observations.columns

    # Prepare columns
    obs_rename = {v: f"observed_{v}" for v in variables if v in observations.columns}
    obs_keys = ["observation_time"] + (["location_id"] if by_location else [])
    obs_df = observations[obs_keys + list(obs_rename.keys())].copy()
    obs_df = obs_df.rename(columns=obs_rename)

    fcst_cols = [
        "forecast_issue_time",
        "forecast_target_time",
        "lead_hours",
        "model",
        "latitude",
        "longitude",
        "elevation",
    ]
    if by_location:
        fcst_cols.append("location_id")
    fcst_cols += [v for v in variables if v in forecasts.columns]
    fcst_rename = {v: f"forecast_{v}" for v in variables if v in forecasts.columns}
    fcst_df = forecasts[fcst_cols].copy()
    fcst_df = fcst_df.rename(columns=fcst_rename)

    # Join on time (and location, when the dataset covers more than one site)
    left_keys = ["forecast_target_time"] + (["location_id"] if by_location else [])
    right_keys = ["observation_time"] + (["location_id"] if by_location else [])
    merged = pd.merge(fcst_df, obs_df, left_on=left_keys, right_on=right_keys, how="left")

    # Compute errors
    for var in variables:
        fcst_col = f"forecast_{var}"
        obs_col = f"observed_{var}"
        if fcst_col in merged.columns and obs_col in merged.columns:
            merged[f"{var}_error"] = merged[fcst_col] - merged[obs_col]

    # Add time dimensions
    merged["month"] = merged["forecast_target_time"].dt.month
    month_to_season = {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }
    merged["season"] = merged["month"].map(month_to_season)

    matched = merged["observation_time"].notna().sum()
    console.print(f"  Merged: {len(merged)} rows, {matched} with observations")

    # Save
    output_path = output_dir / "verification.parquet"
    merged.to_parquet(output_path)
    console.print(f"[green]Saved:[/green] {output_path}")

    return 0
