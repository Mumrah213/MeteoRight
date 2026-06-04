"""Manifest generation and management for curated research datasets.

Every prepared dataset has a JSON manifest that describes:
- What it contains (location, time range, variables, models)
- How it was generated (pipeline steps, versions, timestamps)
- Where its data came from (provenance, license)
- Statistics about the dataset (row counts, event counts)

Usage
-----
>>> from datasets.manifest import build_manifest, write_manifest, validate_manifest

>>> manifest = build_manifest(
...     dataset_name="copenhagen_2y",
...     location={"name": "Copenhagen", "lat": 55.605, "lon": 12.574},
...     start_date="2024-01-01",
...     end_date="2025-12-31",
...     variables=["temperature_2m", "precipitation"],
...     models=["icon_eu"],
... )
>>> write_manifest(manifest, "./datasets/copenhagen_2y/manifests/manifest.json")
>>> validate_manifest(manifest)  # returns list of issues
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schema import (
    LOC_LATITUDE,
    LOC_LONGITUDE,
    LOC_NAME,
    LOC_TIMEZONE,
    MANIFEST_FIELD_DATASET_NAME,
    MANIFEST_FIELD_GENERATED_AT,
    MANIFEST_FIELD_LOCATION,
    MANIFEST_FIELD_MODELS,
    MANIFEST_FIELD_PIPELINE,
    MANIFEST_FIELD_PIPELINE_VERSION,
    MANIFEST_FIELD_PROVENANCE,
    MANIFEST_FIELD_STATISTICS,
    MANIFEST_FIELD_TIME_RANGE,
    MANIFEST_FIELD_VARIABLES,
    MANIFEST_FIELD_VERSION,
    PIPELINE_END_TIME,
    PIPELINE_START_TIME,
    PIPELINE_STEPS_COMPLETED,
    PROV_FORECAST_SOURCE,
    PROV_LICENSE,
    PROV_OBSERVATION_SOURCE,
    STAT_EVENT_DATASETS,
    STAT_FORECAST_ROWS,
    STAT_METRIC_GROUPS,
    STAT_OBSERVATION_ROWS,
    STAT_VERIFICATION_ROWS,
    TR_END_DATE,
    TR_START_DATE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

def build_manifest(
    dataset_name: str,
    location: dict[str, Any],
    start_date: str,
    end_date: str,
    variables: list[str],
    models: list[str],
    version: str = "1.0.0",
    pipeline_version: str = "2.0.0",
    pipeline_start_time: str | None = None,
) -> dict[str, Any]:
    """Build a dataset manifest dictionary.

    Args:
        dataset_name: Unique dataset name (e.g., "copenhagen_2y").
        location: Dict with keys: name, lat, lon, timezone (optional).
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        variables: List of variable names included in the dataset.
        models: List of model names included in the dataset.
        version: Dataset version string.
        pipeline_version: Pipeline version string.
        pipeline_start_time: ISO 8601 UTC timestamp when pipeline started.

    Returns:
        Manifest dictionary ready to be serialized to JSON.

    Example:
        >>> manifest = build_manifest(
        ...     dataset_name="copenhagen_2y",
        ...     location={"name": "Copenhagen", "lat": 55.605, "lon": 12.574},
        ...     start_date="2024-01-01",
        ...     end_date="2025-12-31",
        ...     variables=["temperature_2m"],
        ...     models=["icon_eu"],
        ... )
    """
    now = datetime.now(timezone.utc)

    manifest: dict[str, Any] = {
        MANIFEST_FIELD_DATASET_NAME: dataset_name,
        MANIFEST_FIELD_VERSION: version,
        MANIFEST_FIELD_PIPELINE_VERSION: pipeline_version,
        MANIFEST_FIELD_LOCATION: {
            LOC_NAME: location.get("name", ""),
            LOC_LATITUDE: location["lat"],
            LOC_LONGITUDE: location["lon"],
            LOC_TIMEZONE: location.get("timezone", "UTC"),
        },
        MANIFEST_FIELD_TIME_RANGE: {
            TR_START_DATE: start_date,
            TR_END_DATE: end_date,
        },
        MANIFEST_FIELD_VARIABLES: sorted(variables),
        MANIFEST_FIELD_MODELS: sorted(models),
        MANIFEST_FIELD_GENERATED_AT: now.isoformat(),
        MANIFEST_FIELD_STATISTICS: {
            STAT_FORECAST_ROWS: 0,
            STAT_OBSERVATION_ROWS: 0,
            STAT_VERIFICATION_ROWS: 0,
            STAT_EVENT_DATASETS: [],
            STAT_METRIC_GROUPS: [],
        },
        MANIFEST_FIELD_PROVENANCE: {
            PROV_FORECAST_SOURCE: "open-meteo-single-runs",
            PROV_OBSERVATION_SOURCE: "open-meteo-archive",
            PROV_LICENSE: "CC BY 4.0",
        },
        MANIFEST_FIELD_PIPELINE: {
            PIPELINE_STEPS_COMPLETED: [],
        },
    }

    if pipeline_start_time:
        manifest[MANIFEST_FIELD_PIPELINE][PIPELINE_START_TIME] = pipeline_start_time

    return manifest


def update_manifest_statistics(
    manifest: dict[str, Any],
    forecast_rows: int | None = None,
    observation_rows: int | None = None,
    verification_rows: int | None = None,
    event_datasets: list[str] | None = None,
    metric_groups: list[str] | None = None,
) -> dict[str, Any]:
    """Update statistics in an existing manifest.

    Args:
        manifest: Existing manifest dictionary.
        forecast_rows: Number of forecast rows (replaces current value).
        observation_rows: Number of observation rows.
        verification_rows: Number of verification rows.
        event_datasets: List of event dataset names.
        metric_groups: List of metric group names.

    Returns:
        Updated manifest dictionary.
    """
    stats = manifest.setdefault(MANIFEST_FIELD_STATISTICS, {})

    if forecast_rows is not None:
        stats[STAT_FORECAST_ROWS] = forecast_rows
    if observation_rows is not None:
        stats[STAT_OBSERVATION_ROWS] = observation_rows
    if verification_rows is not None:
        stats[STAT_VERIFICATION_ROWS] = verification_rows
    if event_datasets is not None:
        stats[STAT_EVENT_DATASETS] = sorted(event_datasets)
    if metric_groups is not None:
        stats[STAT_METRIC_GROUPS] = sorted(metric_groups)

    return manifest


def update_pipeline_status(
    manifest: dict[str, Any],
    step: str,
    end_time: str | None = None,
) -> dict[str, Any]:
    """Record a pipeline step completion in the manifest.

    Args:
        manifest: Existing manifest dictionary.
        step: Step name (e.g., "download", "verify", "metrics").
        end_time: ISO 8601 UTC timestamp for step completion.

    Returns:
        Updated manifest dictionary.
    """
    pipeline = manifest.setdefault(MANIFEST_FIELD_PIPELINE, {})
    steps = pipeline.setdefault(PIPELINE_STEPS_COMPLETED, [])

    if step not in steps:
        steps.append(step)

    if end_time is not None:
        pipeline[PIPELINE_END_TIME] = end_time

    return manifest


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------

def write_manifest(
    manifest: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write manifest dictionary to JSON file.

    Creates parent directories if needed.

    Args:
        manifest: Manifest dictionary.
        output_path: Output file path (.json).

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("Wrote manifest: %s", output_path)
    return output_path


def write_manifest_to_dataset(
    manifest: dict[str, Any],
    dataset_dir: str | Path,
) -> Path:
    """Write manifest to the manifests/ subdirectory of a dataset.

    Args:
        manifest: Manifest dictionary.
        dataset_dir: Root dataset directory.

    Returns:
        Path to the written manifest file.
    """
    dataset_dir = Path(dataset_dir)
    output_path = dataset_dir / "manifests" / "manifest.json"
    return write_manifest(manifest, output_path)


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------

def read_manifest(
    dataset_dir: str | Path,
) -> dict[str, Any] | None:
    """Read manifest from a dataset directory.

    Args:
        dataset_dir: Root dataset directory.

    Returns:
        Manifest dictionary, or None if not found.
    """
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifests" / "manifest.json"

    if not manifest_path.exists():
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------

class ManifestIssue:
    """Represents a validation issue in a manifest."""

    def __init__(
        self,
        check: str,
        message: str,
        severity: str = "error",
    ):
        self.check = check
        self.message = message
        self.severity = severity

    def __repr__(self) -> str:
        return (
            f"ManifestIssue(check={self.check!r}, message={self.message!r}, "
            f"severity={self.severity!r})"
        )


def validate_manifest(
    manifest: dict[str, Any],
) -> list[ManifestIssue]:
    """Validate a manifest dictionary.

    Checks:
    - All required top-level fields are present
    - Location subfields are valid
    - Time range dates are present
    - Statistics counts are non-negative
    - Variables and models are non-empty lists

    Args:
        manifest: Manifest dictionary to validate.

    Returns:
        List of ManifestIssue objects (empty if valid).
    """
    issues: list[ManifestIssue] = []

    # Required top-level fields
    for field in [
        MANIFEST_FIELD_DATASET_NAME,
        MANIFEST_FIELD_VERSION,
        MANIFEST_FIELD_PIPELINE_VERSION,
        MANIFEST_FIELD_LOCATION,
        MANIFEST_FIELD_TIME_RANGE,
        MANIFEST_FIELD_VARIABLES,
        MANIFEST_FIELD_MODELS,
        MANIFEST_FIELD_GENERATED_AT,
        MANIFEST_FIELD_STATISTICS,
        MANIFEST_FIELD_PROVENANCE,
    ]:
        if field not in manifest:
            issues.append(ManifestIssue(
                "missing_field",
                f"Required field missing: {field}",
                "error",
            ))

    # Validate location subfields
    location = manifest.get(MANIFEST_FIELD_LOCATION, {})
    for field in [LOC_NAME, LOC_LATITUDE, LOC_LONGITUDE]:
        if field not in location:
            issues.append(ManifestIssue(
                "missing_location_field",
                f"Location missing required field: {field}",
                "error",
            ))

    if LOC_LATITUDE in location:
        try:
            lat = float(location[LOC_LATITUDE])
            if not (-90 <= lat <= 90):
                issues.append(ManifestIssue(
                    "invalid_latitude",
                    f"Latitude {lat} is out of range [-90, 90]",
                    "error",
                ))
        except (ValueError, TypeError):
            issues.append(ManifestIssue(
                "invalid_latitude",
                f"Latitude is not a number: {location[LOC_LATITUDE]}",
                "error",
            ))

    if LOC_LONGITUDE in location:
        try:
            lon = float(location[LOC_LONGITUDE])
            if not (-180 <= lon <= 180):
                issues.append(ManifestIssue(
                    "invalid_longitude",
                    f"Longitude {lon} is out of range [-180, 180]",
                    "error",
                ))
        except (ValueError, TypeError):
            issues.append(ManifestIssue(
                "invalid_longitude",
                f"Longitude is not a number: {location[LOC_LONGITUDE]}",
                "error",
            ))

    # Validate statistics numeric fields
    stats = manifest.get(MANIFEST_FIELD_STATISTICS, {})
    for field in [STAT_FORECAST_ROWS, STAT_OBSERVATION_ROWS, STAT_VERIFICATION_ROWS]:
        if field in stats:
            val = stats[field]
            if not isinstance(val, (int, float)) or val < 0:
                issues.append(ManifestIssue(
                    "invalid_statistics_value",
                    f"Statistics field '{field}' must be a non-negative number, got {val}",
                    "error",
                ))

    # Validate lists are non-empty
    for field in [MANIFEST_FIELD_VARIABLES, MANIFEST_FIELD_MODELS]:
        if field in manifest:
            val = manifest[field]
            if not isinstance(val, list) or len(val) == 0:
                issues.append(ManifestIssue(
                    "empty_list",
                    f"Field '{field}' must be a non-empty list",
                    "error",
                ))

    logger.info("Manifest validation: %d issue(s)", len(issues))
    return issues
