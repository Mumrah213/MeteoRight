"""Precompute event datasets for the dataset pipeline.

Builds event-enriched verification datasets for each built-in event type:
- frost: Forecast and observed temperature ≤ 0°C
- heavy_precip: Forecast or observed precipitation ≥ 5 mm
- hot_day: Forecast or observed temperature ≥ 30°C
- no_rain: Precipitation < 0.1 mm

Outputs:
- events/frost/forecast_events.parquet, obs_events.parquet, verification_events.parquet
- events/heavy_precip/forecast_events.parquet, obs_events.parquet, verification_events.parquet
- etc.

Each event dataset includes:
- All original verification columns
- An is_event boolean column
- Event timing (first_event_date, event_duration)

Usage
-----
>>> from datasets.build_events import build_events

>>> result = build_events(
...     verification_path="./datasets/copenhagen_2y/verification/verification.parquet",
...     events_dir="./datasets/copenhagen_2y/events",
... )
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.config import EVENT_DEFINITIONS, BUILT_IN_EVENTS
from datasets.progress import ProgressReporter

logger = logging.getLogger(__name__)


def build_events(
    verification_path: str | Path,
    events_dir: str | Path,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Precompute event datasets from a verification dataset.

    For each built-in event type, identifies forecast and observation events,
    then creates joined verification event datasets.

    Args:
        verification_path: Path to verification parquet file.
        events_dir: Output directory for event datasets.
        progress: Progress reporter instance.

    Returns:
        Dict with keys: event_datasets (list of names), event_rows (total rows).
    """
    if progress is None:
        progress = ProgressReporter()

    # ------------------------------------------------------------------
    # Load verification data
    # ------------------------------------------------------------------
    progress.section("Step 5: Precomputing events")

    progress.step("loading", f"Reading verification from {Path(verification_path).name}...")
    verification_df = pd.read_parquet(verification_path)
    progress.step("loaded", f"{len(verification_df):,} rows")

    result: dict[str, Any] = {"event_datasets": [], "event_rows": 0}
    created_files: list[str] = []

    # ------------------------------------------------------------------
    # Process each event type
    # ------------------------------------------------------------------
    for event_name in BUILT_IN_EVENTS:
        definition = EVENT_DEFINITIONS[event_name]
        var = definition["variable"]
        threshold = definition["threshold"]
        operator = definition["operator"]

        progress.section(f"Event: {event_name}")

        # Ensure output directory
        event_dir = events_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # Forecast events
        # ------------------------------------------------------------------
        fcst_col = f"forecast_{var}"
        if fcst_col not in verification_df.columns:
            progress.warning(event_name, f"Variable {var} not in dataset (missing forecast_{var})")
            continue

        fcst_df = _apply_event_threshold(
            verification_df[verification_df[fcst_col].notna()].copy(),
            fcst_col, var, "forecast", threshold, operator,
        )
        if not fcst_df.empty:
            fcst_path = event_dir / "forecast_events.parquet"
            fcst_df.to_parquet(fcst_path, engine="pyarrow", index=False)
            created_files.append(f"{event_name}/forecast_events.parquet")
            progress.success(f"forecast events", f"{len(fcst_df):,} events")
        else:
            progress.warning(f"forecast events", "No events found")

        # ------------------------------------------------------------------
        # Observation events
        # ------------------------------------------------------------------
        obs_col = f"observed_{var}"
        if obs_col not in verification_df.columns:
            progress.warning(event_name, f"Variable {var} not in dataset (missing observed_{var})")
            continue

        obs_df = _apply_event_threshold(
            verification_df[verification_df[obs_col].notna()].copy(),
            obs_col, var, "observed", threshold, operator,
        )
        if not obs_df.empty:
            obs_path = event_dir / "obs_events.parquet"
            obs_df.to_parquet(obs_path, engine="pyarrow", index=False)
            created_files.append(f"{event_name}/obs_events.parquet")
            progress.success(f"obs events", f"{len(obs_df):,} events")

        # ------------------------------------------------------------------
        # Verification events (both forecast and observation events)
        # ------------------------------------------------------------------
        fcst_event_col = f"is_{var}_event_fcst"
        obs_event_col = f"is_{var}_event_obs"
        if fcst_event_col in verification_df.columns and obs_event_col in verification_df.columns:
            ver_df = verification_df[
                (verification_df[fcst_event_col]) | (verification_df[obs_event_col])
            ].copy()
            ver_path = event_dir / "verification_events.parquet"
            ver_df.to_parquet(ver_path, engine="pyarrow", index=False)
            created_files.append(f"{event_name}/verification_events.parquet")
            progress.success(f"verification events", f"{len(ver_df):,} events")

        result["event_datasets"].append(event_name)
        result["event_rows"] += len(fcst_df) + len(obs_df)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    progress.complete(f"Events precomputed: {len(created_files)} files")

    return {
        "event_datasets": result["event_datasets"],
        "event_rows": result["event_rows"],
        "created_files": created_files,
    }


def _apply_event_threshold(
    df: pd.DataFrame,
    value_col: str,
    variable_name: str,
    source: str,
    threshold: float,
    operator: str,
) -> pd.DataFrame:
    """Apply an event threshold to create is_event columns.

    Args:
        df: DataFrame to process.
        value_col: Column name containing the values to threshold.
        variable_name: Variable name (for column naming).
        source: 'forecast' or 'observed'.
        threshold: Threshold value.
        operator: Comparison operator ('gt', 'ge', 'lt', 'le').

    Returns:
        DataFrame with event columns added (is_event, event_timing).
    """
    df = df.copy()
    df[f"event_{variable_name}_{source}"] = threshold

    # Apply the operator
    if operator == "gt":
        df[f"is_{variable_name}_event_{source}"] = df[value_col] > threshold
    elif operator == "ge":
        df[f"is_{variable_name}_event_{source}"] = df[value_col] >= threshold
    elif operator == "lt":
        df[f"is_{variable_name}_event_{source}"] = df[value_col] < threshold
    elif operator == "le":
        df[f"is_{variable_name}_event_{source}"] = df[value_col] <= threshold
    else:
        logger.warning("Unknown event operator: %s", operator)

    # Add event timing columns
    if "forecast_target_time" in df.columns:
        target_time = pd.to_datetime(df["forecast_target_time"], utc=True)
        df["event_date"] = target_time.dt.date

    return df
