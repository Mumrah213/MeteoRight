"""Verification dataset construction for the dataset pipeline.

Builds a canonical verification dataset by joining forecasts with
observations, reusing existing alignment and validation logic.

This step creates:
- verification/verification.parquet — the canonical joined dataset
- Preserves all provenance (issue_time, target_time, model)
- Adds error columns (forecast - observed)
- Adds derived columns (month, season)

Usage
-----
>>> from datasets.build_verification import build_verification

>>> result = build_verification(
...     data_dir="./datasets/copenhagen_2y/raw",
...     verification_dir="./datasets/copenhagen_2y/verification",
...     variables=["temperature_2m", "precipitation"],
... )
>>> print(result["verification_rows"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.progress import ProgressReporter
from verification.location import validate_location_compatibility

logger = logging.getLogger(__name__)

# Season mapping — shared with metrics/advanced_verification
SEASON_MAP: dict[int, str] = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def build_verification(
    data_dir: str | Path,
    verification_dir: str | Path,
    variables: tuple[str, ...],
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Build a canonical verification dataset from raw downloads.

    Joins forecasts with observations on target_time == observation_time,
    preserves provenance, computes error columns, and adds derived time
    dimensions.

    Args:
        data_dir: Raw data directory containing forecasts/ and observations/.
        verification_dir: Output directory for verification dataset.
        variables: Tuple of variable names to include.
        progress: Progress reporter instance.

    Returns:
        Dict with key 'verification_rows' — number of rows in output.
    """
    data_dir = Path(data_dir)
    verification_dir = Path(verification_dir)
    verification_dir.mkdir(parents=True, exist_ok=True)

    if progress is None:
        progress = ProgressReporter()

    # ------------------------------------------------------------------
    # Load forecasts
    # ------------------------------------------------------------------
    progress.section("Step 3: Building verification dataset")

    from downloader.storage import list_downloaded

    fcst_files = list_downloaded(data_dir, "forecasts")
    if not fcst_files:
        raise RuntimeError("No forecast data found. Run download step first.")

    progress.step("forecasts", f"{len(fcst_files)} partition(s) found")

    forecasts = pd.concat(
        [pd.read_parquet(entry["path"]) for entry in fcst_files],
        ignore_index=True,
    )
    forecast_rows = len(forecasts)
    progress.step("forecasts", f"{forecast_rows:,} rows loaded")

    # ------------------------------------------------------------------
    # Load observations
    # ------------------------------------------------------------------
    obs_files = list_downloaded(data_dir, "observations")
    if not obs_files:
        raise RuntimeError("No observation data found. Run download step first.")

    progress.step("observations", f"{len(obs_files)} partition(s) found")

    observations = pd.concat(
        [pd.read_parquet(entry["path"]) for entry in obs_files],
        ignore_index=True,
    )
    obs_rows = len(observations)
    progress.step("observations", f"{obs_rows:,} rows loaded")

    audit = validate_location_compatibility(forecasts, observations)
    progress.step(
        "locations",
        (
            f"{audit.forecast_locations} forecast location(s), "
            f"{audit.observation_locations} observation location(s), "
            f"max nearest distance {audit.max_nearest_distance_km:.1f} km"
        ),
    )

    # ------------------------------------------------------------------
    # Join forecasts with observations
    # ------------------------------------------------------------------
    progress.step("joining", "LEFT JOIN on target_time == observation_time...")

    obs_rename = {v: f"observed_{v}" for v in variables if v in observations.columns}
    obs_cols = ["observation_time"] + list(obs_rename.keys())
    if "location_id" in observations.columns and "location_id" in forecasts.columns:
        obs_cols.append("location_id")
    obs_df = observations[obs_cols].copy()
    obs_df = obs_df.rename(columns=obs_rename)

    fcst_cols = ["forecast_issue_time", "forecast_target_time", "lead_hours", "model",
                 "latitude", "longitude", "elevation"]
    if "location_id" in forecasts.columns:
        fcst_cols.append("location_id")
    fcst_cols += [v for v in variables if v in forecasts.columns]
    fcst_rename = {v: f"forecast_{v}" for v in variables if v in forecasts.columns}
    fcst_df = forecasts[fcst_cols].copy()
    fcst_df = fcst_df.rename(columns=fcst_rename)

    left_keys = ["forecast_target_time"]
    right_keys = ["observation_time"]
    if "location_id" in fcst_df.columns and "location_id" in obs_df.columns:
        left_keys.append("location_id")
        right_keys.append("location_id")

    merged = pd.merge(fcst_df, obs_df, left_on=left_keys, right_on=right_keys, how="left")

    # ------------------------------------------------------------------
    # Compute error columns
    # ------------------------------------------------------------------
    progress.step("errors", "Computing error columns...")

    for var in variables:
        fcst_col = f"forecast_{var}"
        obs_col = f"observed_{var}"
        if fcst_col in merged.columns and obs_col in merged.columns:
            merged[f"{var}_error"] = merged[fcst_col] - merged[obs_col]

    # ------------------------------------------------------------------
    # Add derived time dimensions
    # ------------------------------------------------------------------
    merged["month"] = merged["forecast_target_time"].dt.month
    merged["season"] = merged["month"].map(SEASON_MAP).fillna("UNKNOWN")

    matched = merged["observation_time"].notna().sum()
    progress.step("merged", f"{len(merged):,} rows, {matched:,} with observations")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    progress.step("saving", "Writing verification parquet...")
    output_path = verification_dir / "verification.parquet"
    merged.to_parquet(output_path, engine="pyarrow", index=False)

    progress.success("verification", f"{len(merged):,} rows → {output_path}")
    progress.complete("Verification dataset built")

    return {"verification_rows": len(merged)}
