"""`meteoright download` — fetch forecasts or observations from Open-Meteo."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ._shared import console, logger


def cmd_download(args: argparse.Namespace) -> int:
    """Download forecasts or observations."""
    import time

    from downloader.api import fetch_observations, fetch_single_run
    from downloader.config import DownloadConfig
    from downloader.constants import DEFAULT_API_SLEEP, MODEL_FORECAST_HOURS, MODEL_RUN_CYCLES
    from downloader.normalize import json_to_forecast_df, json_to_observation_df
    from downloader.storage import (
        write_attribution,
        write_download_log,
        write_forecasts,
        write_observations,
    )

    output_dir = Path(args.output).resolve()
    variables = tuple(v.strip() for v in args.variables.split(","))
    api_key = args.api_key
    archive_url = args.open_meteo_archive_url
    single_runs_url = args.open_meteo_single_runs_url
    if args.open_meteo_base_url:
        base_url = args.open_meteo_base_url.rstrip("/")
        archive_url = archive_url or f"{base_url}/v1/archive"
        single_runs_url = single_runs_url or f"{base_url}/v1/forecast"

    config = DownloadConfig(
        latitude=args.lat,
        longitude=args.lon,
        start_date=args.start,
        end_date=args.end,
        variables=variables,
        model=args.model,
        output_dir=output_dir,
    )

    if args.type == "observations":
        console.print(f"[bold]Downloading observations[/bold]: {args.start} to {args.end}")
        response = fetch_observations(
            config.latitude,
            config.longitude,
            config.start_date,
            config.end_date,
            config.variables,
            config.max_retries,
            config.retry_delay,
            api_key=api_key,
            url=archive_url,
        )
        df = json_to_observation_df(response, config.variables)
        rows = write_observations(df, config.output_dir)
        write_download_log(
            config.output_dir,
            "observations",
            None,
            config.start_date,
            config.end_date,
            rows_downloaded=rows,
        )
        write_attribution(
            config.output_dir,
            "observations",
            "open-meteo-archive",
            latitude=config.latitude,
            longitude=config.longitude,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        console.print(f"[green]Done.[/green] Wrote {rows} observation rows.")
        console.print("[dim]Data attribution: Open-Meteo.com (CC BY 4.0)[/dim]")

    else:  # forecast
        if not args.model:
            console.print("[red]Error: --model required for forecast download[/red]")
            return 1

        console.print(
            f"[bold]Downloading forecasts[/bold]: model={args.model}, {args.start} to {args.end}"
        )

        # Generate run times
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
        cycles = MODEL_RUN_CYCLES.get(args.model, [0, 6, 12, 18])
        forecast_hours = MODEL_FORECAST_HOURS.get(args.model, 240)
        forecast_delta = timedelta(hours=forecast_hours)

        run_times = []
        current = start
        while current < end + forecast_delta:
            for hour in cycles:
                run_time = current.replace(hour=hour, minute=0, second=0)
                if run_time < start - forecast_delta:
                    continue
                if run_time >= end + timedelta(hours=24):
                    continue
                run_times.append(run_time)
            current += timedelta(days=1)

        console.print(f"  Fetching {len(run_times)} forecast runs...")

        all_dfs = []
        skipped = 0
        for i, run_time in enumerate(run_times):
            try:
                response = fetch_single_run(
                    config.latitude,
                    config.longitude,
                    run_time,
                    config.variables,
                    config.max_retries,
                    config.retry_delay,
                    api_key=api_key,
                    url=single_runs_url,
                )
                df = json_to_forecast_df(response, run_time, args.model, config.variables)
                all_dfs.append(df)
                if i < len(run_times) - 1:
                    time.sleep(DEFAULT_API_SLEEP)
            except Exception as e:
                logger.warning("Failed to fetch run %s: %s", run_time.isoformat(), e)
                skipped += 1

        if not all_dfs:
            console.print("[red]No forecast data downloaded.[/red]")
            return 1

        all_columns = sorted(set().union(*(set(df.columns) for df in all_dfs)))
        for df in all_dfs:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = None
        combined = pd.concat(all_dfs, ignore_index=True)[all_columns]
        rows = write_forecasts(combined, config.output_dir)
        write_download_log(
            config.output_dir,
            "forecast",
            args.model,
            config.start_date,
            config.end_date,
            rows_downloaded=rows,
            rows_skipped=skipped,
        )
        write_attribution(
            config.output_dir,
            "forecast",
            "open-meteo-single-runs",
            latitude=config.latitude,
            longitude=config.longitude,
            model=args.model,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        console.print(f"[green]Done.[/green] Wrote {rows} forecast rows ({skipped} runs failed).")
        console.print("[dim]Data attribution: Open-Meteo.com (CC BY 4.0)[/dim]")

    return 0
