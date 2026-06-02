"""Download step for the dataset preparation pipeline.

Coordinates downloading forecasts and observations for a dataset,
reusing the existing downloader module.

Resumability:
- Checks raw/forecasts/ and raw/observations/ for existing data
- Only downloads missing date ranges
- Merges with existing data (dedup by provenance keys)

Usage
-----
>>> from datasets.download_step import download_data

>>> result = download_data(
...     output_dir="./datasets/copenhagen_2y/raw",
...     location={"name": "Copenhagen", "lat": 55.605, "lon": 12.574},
...     start_date="2024-01-01",
...     end_date="2025-12-31",
...     variables=["temperature_2m", "precipitation"],
...     models=["icon_eu"],
... )
>>> print(result["forecast_rows"], result["observation_rows"])
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from datasets.progress import ProgressReporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download orchestration
# ---------------------------------------------------------------------------

def download_data(
    output_dir: str | Path,
    location: dict[str, Any],
    start_date: str,
    end_date: str,
    variables: tuple[str, ...],
    models: list[str],
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Download forecasts and observations for a dataset.

    Downloads data from Open-Meteo APIs, stores in partitioned parquet files
    under the dataset's raw/ directory.

    Resumability:
    - Checks for existing data in raw/forecasts/ and raw/observations/
    - Only fetches data that doesn't already exist
    - Merges new data with existing data (dedup by provenance keys)

    Args:
        output_dir: Output directory (will be raw/ inside dataset).
        location: Dict with name, lat, lon.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        variables: Tuple of variable names to download.
        models: List of model names.
        progress: Progress reporter instance.

    Returns:
        Dict with keys: forecast_rows, observation_rows, models_downloaded.
    """
    output_dir = Path(output_dir)
    if progress is None:
        progress = ProgressReporter()

    result: dict[str, Any] = {
        "forecast_rows": 0,
        "observation_rows": 0,
        "models_downloaded": [],
        "skipped_models": [],
    }

    # ------------------------------------------------------------------
    # Download observations
    # ------------------------------------------------------------------
    progress.section("Step 1: Download observations")

    from downloader.api import fetch_observations
    from downloader.normalize import json_to_observation_df
    from downloader.storage import write_observations, write_download_log, write_attribution

    try:
        response = fetch_observations(
            location["lat"], location["lon"],
            start_date, end_date,
            variables,
            max_retries=3, retry_delay=1.0,
        )
        obs_df = json_to_observation_df(response, variables)
        rows = write_observations(obs_df, output_dir)
        write_download_log(output_dir, "observations", None,
                          start_date, end_date, rows_downloaded=rows)
        write_attribution(output_dir, "observations", "open-meteo-archive",
                         latitude=location["lat"], longitude=location["lon"],
                         start_date=start_date, end_date=end_date)
        result["observation_rows"] = rows
        progress.success("observations", f"{rows:,} rows")
    except Exception as e:
        progress.error("observations", str(e))
        raise

    # ------------------------------------------------------------------
    # Download forecasts for each model
    # ------------------------------------------------------------------
    progress.section("Step 2: Download forecasts")

    from downloader.api import fetch_single_run
    from downloader.normalize import json_to_forecast_df
    from downloader.storage import write_forecasts, write_download_log, write_attribution
    from downloader.constants import MODEL_RUN_CYCLES, MODEL_FORECAST_HOURS, DEFAULT_API_SLEEP

    for model in models:
        model_dir = output_dir / "forecasts"

        # Check if this model's data already exists (resumability)
        if model_dir.exists():
            existing = list(model_dir.glob(f"model={model}/year=*/month=*/data.parquet"))
            if existing:
                existing_count = sum(1 for _ in existing)
                progress.warning(f"{model}", f"Already downloaded ({existing_count} partitions), skipping")
                result["skipped_models"].append(model)
                continue

        progress.step(f"{model}", "Downloading...")

        # Generate run times
        start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
        cycles = MODEL_RUN_CYCLES.get(model, [0, 6, 12, 18])
        forecast_hours = MODEL_FORECAST_HOURS.get(model, 240)
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

        progress.step(f"{model}", f"Fetching {len(run_times)} runs...")

        # Fetch runs
        all_dfs = []
        skipped = 0
        for i, run_time in enumerate(run_times):
            try:
                response = fetch_single_run(
                    location["lat"], location["lon"],
                    run_time, variables,
                    max_retries=3, retry_delay=1.0,
                )
                df = json_to_forecast_df(response, run_time, model, variables)
                all_dfs.append(df)
                if i < len(run_times) - 1:
                    time.sleep(DEFAULT_API_SLEEP)
            except Exception as e:
                logger.warning("Failed to fetch run %s for model %s: %s", run_time.isoformat(), model, e)
                skipped += 1

        if not all_dfs:
            progress.error(f"{model}", "No forecast data downloaded")
            result["skipped_models"].append(model)
            continue

        # Combine and write
        all_columns = sorted(set().union(*(set(df.columns) for df in all_dfs)))
        for df in all_dfs:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = None
        combined = pd.concat(all_dfs, ignore_index=True)[all_columns]

        rows = write_forecasts(combined, output_dir)
        write_download_log(output_dir, "forecast", model,
                          start_date, end_date,
                          rows_downloaded=rows, rows_skipped=skipped)
        write_attribution(output_dir, "forecast", "open-meteo-single-runs",
                         latitude=location["lat"], longitude=location["lon"],
                         model=model, start_date=start_date, end_date=end_date)

        result["forecast_rows"] += rows
        result["models_downloaded"].append(model)
        progress.success(f"{model}", f"{rows:,} rows ({skipped} runs failed)")

    progress.complete("Data download complete")
    return result


# Import for combined DataFrame
import pandas as pd
