"""Parquet storage for weather data.

Handles partitioned writes, append mode, and download logging.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _dedup_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return dedup columns that exist in the DataFrame."""
    return [col for col in candidates if col in df.columns]


def _location_identity_columns(df: pd.DataFrame) -> list[str]:
    """Prefer stable location IDs, falling back to returned grid coordinates."""
    if "location_id" in df.columns and df["location_id"].notna().any():
        return ["location_id"]
    return _dedup_columns(df, ["latitude", "longitude"])


def _ensure_dirs(output_dir: Path) -> None:
    """Create output directories if they don't exist."""
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)


def write_attribution(
    output_dir: Path,
    data_type: str,
    source: str,
    latitude: float | None = None,
    longitude: float | None = None,
    model: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Write a JSON attribution metadata file for downloaded data.

    Open-Meteo data is CC BY 4.0 — this file records provenance so
    downstream tools can propagate the attribution requirement.

    Args:
        output_dir: Root output directory.
        data_type: "forecast" or "observations".
        source: API source name.
        latitude: Requested latitude.
        longitude: Requested longitude.
        model: Model name (for forecasts).
        start_date: Start date string.
        end_date: End date string.
    """
    _ensure_dirs(output_dir)
    path = output_dir / "metadata" / "attribution.json"

    record: dict[str, Any] = {
        "license": "CC BY 4.0",
        "source": source,
        "url": "https://open-meteo.com/",
        "data_type": data_type,
        "attribution": (
            "Weather data provided by Open-Meteo.com "
            "under CC BY 4.0 license "
            "(https://open-meteo.com/en/license)"
        ),
    }
    if latitude is not None:
        record["latitude"] = latitude
    if longitude is not None:
        record["longitude"] = longitude
    if model is not None:
        record["model"] = model
    if start_date is not None:
        record["start_date"] = start_date
    if end_date is not None:
        record["end_date"] = end_date

    # Merge with existing attribution if present (append data_type info)
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                existing: dict[str, Any] = json.load(f)
            # Keep the more complete record
            existing.update(record)
            record = existing
        except (json.JSONDecodeError, ValueError):
            pass

    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    logger.info("Wrote attribution metadata: %s", path)


def read_attribution(output_dir: Path) -> dict[str, Any] | None:
    """Read the attribution metadata file, or return None if missing.

    Args:
        output_dir: Root output directory.

    Returns:
        Attribution dict, or None if not found.
    """
    path = output_dir / "metadata" / "attribution.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_forecasts(
    df: pd.DataFrame,
    output_dir: Path,
    append: bool = True,
) -> int:
    """Write forecast data to partitioned parquet files.

    Partitions: model/year/month

    Args:
        df: DataFrame with ForecastRecord schema.
        output_dir: Root output directory.
        append: If True, merge with existing data before writing.

    Returns:
        Number of rows written.
    """
    _ensure_dirs(output_dir)

    if df.empty:
        logger.info("No forecast data to write")
        return 0

    # Add partition columns
    df = df.copy()
    df["year"] = df["forecast_target_time"].dt.year
    df["month"] = df["forecast_target_time"].dt.month

    written = 0
    for (model, year, month), group in df.groupby(["model", "year", "month"]):
        # Ensure group has the right timestamp columns
        for col in ["forecast_issue_time", "forecast_target_time"]:
            if not pd.api.types.is_datetime64_any_dtype(group[col]):
                group[col] = pd.to_datetime(group[col], utc=True)

        # Append mode: merge with existing
        if append:
            existing = _read_partition(output_dir, "forecasts", model, year, month)
            if existing is not None:
                existing_cols = set(existing.columns)
                dedup_cols = _dedup_columns(
                    group,
                    ["forecast_issue_time", "forecast_target_time", "model"],
                ) + _location_identity_columns(group)
                existing = existing[existing_cols.intersection(set(group.columns))]
                group = pd.concat([existing, group]).drop_duplicates(subset=dedup_cols, keep="last")
                logger.info(
                    "Merged %d existing + %d new rows, %d after dedup (model=%s)",
                    len(existing),
                    len(df),
                    len(group),
                    model,
                )

        # Write parquet
        path = (
            output_dir
            / "forecasts"
            / f"model={model}"
            / f"year={year}"
            / f"month={month:02d}"
            / "data.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(group, preserve_index=False)
        pq.write_table(table, str(path), row_group_size=10000)

        written += len(group)
        logger.info("Wrote %d forecast rows → %s", len(group), path)

    return written


def write_observations(
    df: pd.DataFrame,
    output_dir: Path,
    append: bool = True,
) -> int:
    """Write observation data to partitioned parquet files.

    Partitions: year/month

    Args:
        df: DataFrame with ObservationRecord schema.
        output_dir: Root output directory.
        append: If True, merge with existing data before writing.

    Returns:
        Number of rows written.
    """
    _ensure_dirs(output_dir)

    if df.empty:
        logger.info("No observation data to write")
        return 0

    df = df.copy()
    df["year"] = df["observation_time"].dt.year
    df["month"] = df["observation_time"].dt.month

    written = 0
    for (year, month), group in df.groupby(["year", "month"]):
        if not pd.api.types.is_datetime64_any_dtype(group["observation_time"]):
            group["observation_time"] = pd.to_datetime(group["observation_time"], utc=True)

        if append:
            existing = _read_partition(output_dir, "observations", None, year, month)
            if existing is not None:
                set(existing.columns)
                dedup_cols = _dedup_columns(group, ["observation_time"]) + _location_identity_columns(group)
                group = pd.concat([existing, group]).drop_duplicates(
                    subset=dedup_cols, keep="last"
                )
                logger.info(
                    "Merged %d existing + %d new rows, %d after dedup (year=%d, month=%d)",
                    len(existing),
                    len(group) - len(existing),
                    len(group),
                    year,
                    month,
                )

        path = output_dir / "observations" / f"year={year}" / f"month={month:02d}" / "data.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(group, preserve_index=False)
        pq.write_table(table, str(path), row_group_size=10000)

        written += len(group)
        logger.info("Wrote %d observation rows → %s", len(group), path)

    return written


def _read_partition(
    output_dir: Path,
    dataset_type: str,
    model: str | None,
    year: int,
    month: int,
) -> pd.DataFrame | None:
    """Read an existing partition file, or None if it doesn't exist."""
    if model:
        path = (
            output_dir
            / dataset_type
            / f"model={model}"
            / f"year={year}"
            / f"month={month:02d}"
            / "data.parquet"
        )
    else:
        path = output_dir / dataset_type / f"year={year}" / f"month={month:02d}" / "data.parquet"

    if not path.exists():
        return None

    logger.debug("Reading existing partition: %s", path)
    return pd.read_parquet(path)


def write_download_log(
    output_dir: Path,
    data_type: str,
    model: str | None,
    start_date: str,
    end_date: str,
    rows_downloaded: int,
    rows_skipped: int = 0,
) -> None:
    """Append a download entry to the JSONL log file.

    Args:
        output_dir: Root output directory.
        data_type: "forecast" or "observations".
        model: Model name (None for observations).
        start_date: Start date string.
        end_date: End date string.
        rows_downloaded: Number of rows successfully downloaded.
        rows_skipped: Number of rows skipped (e.g., already existed).
    """
    _ensure_dirs(output_dir)
    log_path = output_dir / "metadata" / "download_log.jsonl"

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "type": data_type,
        "model": model,
        "start_date": start_date,
        "end_date": end_date,
        "rows_downloaded": rows_downloaded,
        "rows_skipped": rows_skipped,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Logged download: %s", entry)


def list_downloaded(
    output_dir: Path,
    data_type: str,
    model: str | None = None,
) -> list[dict]:
    """Scan the data directory and return what's been downloaded.

    Args:
        output_dir: Root output directory.
        data_type: "forecast" or "observations".
        model: Model name (for forecasts).

    Returns:
        List of dicts with keys: model, year, month, path, row_count.
    """
    results = []
    base = output_dir / data_type

    if not base.exists():
        return results

    for partition_dir in base.rglob("data.parquet"):
        rel = partition_dir.relative_to(base)
        parts = rel.parts

        if data_type == "forecasts":
            # model=X/year=Y/month=M/data.parquet
            m = parts[0].split("=")[1] if "=" in parts[0] else None
            y = int(parts[1].split("=")[1])
            mo = int(parts[2].split("=")[1])
            if model and m != model:
                continue
            entry = {"model": m, "year": y, "month": mo, "path": str(partition_dir)}
        else:
            # year=Y/month=M/data.parquet
            y = int(parts[0].split("=")[1])
            mo = int(parts[1].split("=")[1])
            entry = {"model": None, "year": y, "month": mo, "path": str(partition_dir)}

        try:
            entry["row_count"] = len(pd.read_parquet(partition_dir))
        except Exception:
            entry["row_count"] = -1

        results.append(entry)

    return results
