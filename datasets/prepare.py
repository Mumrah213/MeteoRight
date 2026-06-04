"""Main pipeline orchestration for curated research dataset preparation.

This is the entry point for building a complete curated research dataset.
It coordinates all pipeline steps: download, verification, metrics, events,
aggregations, report, and manifest writing.

Usage (CLI)
-----------
>>> from datasets.prepare import run_pipeline

>>> dataset_dir = run_pipeline(
...     location={"name": "Copenhagen", "lat": 55.605, "lon": 12.574},
...     start_date="2024-01-01",
...     end_date="2025-12-31",
...     variables=["temperature_2m", "precipitation"],
...     models=["icon_eu"],
...     build_metrics=True,
...     build_events=True,
...     output_dir="./datasets",
... )
>>> print(dataset_dir)
./datasets/copenhagen_2y

Usage (resumable)
-----------------
>>> # First run: downloads and builds verification
>>> run_pipeline(..., build_metrics=False, build_events=False)
>>> # Second run: resumes, builds remaining steps
>>> run_pipeline(..., build_metrics=True, build_events=True)

Directory structure created
--------------------------
    datasets/
    └── {dataset_name}/
        ├── raw/
        │   ├── forecasts/          # Downloaded forecast parquet
        │   └── observations/       # Downloaded observation parquet
        ├── verification/           # Built verification dataset
        ├── metrics/                # Precomputed metrics
        ├── events/                 # Precomputed events
        ├── aggregates/             # Lead-time, seasonal summaries
        ├── manifests/              # JSON manifest(s)
        ├── metadata/               # Attribution, provenance
        └── reports/                # Summary report (markdown)
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_METRIC_GROUPS,
    DEFAULT_MODELS,
    EVENT_DEFINITIONS,
    generate_dataset_name,
)
from .manifest import (
    build_manifest,
    update_manifest_statistics,
    update_pipeline_status,
    write_manifest_to_dataset,
)
from .progress import ProgressReporter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    location: dict[str, Any],
    start_date: str,
    end_date: str,
    variables: tuple[str, ...] | None = None,
    models: list[str] | None = None,
    build_metrics: bool = True,
    build_events: bool = True,
    build_aggregations: bool = True,
    output_dir: str | Path = "./datasets",
    progress: ProgressReporter | None = None,
) -> str:
    """Execute the full dataset preparation pipeline.

    Args:
        location: Dict with keys: name, lat, lon.
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        variables: Tuple of variable names. If None, uses all variables.
        models: List of model names. If None, uses default for location.
        build_metrics: If True, precompute metrics.
        build_events: If True, precompute events.
        build_aggregations: If True, compute lead-time degradation.
        output_dir: Root output directory.
        progress: Progress reporter instance.

    Returns:
        Absolute path to the created dataset directory.
    """
    if progress is None:
        progress = ProgressReporter()

    # ------------------------------------------------------------------
    # Resolve defaults
    # ------------------------------------------------------------------
    if variables is None:
        variables = tuple([
            "temperature_2m", "precipitation", "wind_speed_10m",
            "pressure_msl", "relative_humidity_2m",
        ])

    if models is None:
        models = list(DEFAULT_MODELS.get(
            location.get("name", "").lower(),
            ["icon_eu"],
        ))

    # ------------------------------------------------------------------
    # Setup directories
    # ------------------------------------------------------------------
    dataset_name = generate_dataset_name(
        location["name"], start_date, end_date,
    )
    dataset_dir = Path(output_dir) / dataset_name

    if dataset_dir.exists():
        if _has_partial_data(dataset_dir):
            progress.warning("Existing dataset", f"'{dataset_name}' already exists, will resume")
        else:
            progress.info(f"Removing existing empty directory: {dataset_name}")
            shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (dataset_dir / "raw" / "forecasts").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "raw" / "observations").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "verification").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "events").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "aggregates").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "reports").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialize manifest
    # ------------------------------------------------------------------
    pipeline_start = datetime.now(timezone.utc).isoformat()
    manifest = build_manifest(
        dataset_name=dataset_name,
        location=location,
        start_date=start_date,
        end_date=end_date,
        variables=list(variables),
        models=models,
        pipeline_version="2.0.0",
        pipeline_start_time=pipeline_start,
    )

    # ------------------------------------------------------------------
    # Step 1: Download
    # ------------------------------------------------------------------
    from .resume import CheckpointTracker, STEP_DOWNLOAD

    tracker = CheckpointTracker(dataset_dir)
    data_dir = dataset_dir / "raw"

    if tracker.skip_step(STEP_DOWNLOAD):
        progress.warning("download", "Already complete, skipping")
    else:
        from .download_step import download_data
        result = download_data(
            output_dir=data_dir,
            location=location,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            models=models,
            progress=progress,
        )
        manifest = update_manifest_statistics(
            manifest,
            forecast_rows=result.get("forecast_rows", 0),
            observation_rows=result.get("observation_rows", 0),
        )
        manifest = update_pipeline_status(manifest, "download")
        tracker.mark_complete(STEP_DOWNLOAD)

    # ------------------------------------------------------------------
    # Step 2: Verification
    # ------------------------------------------------------------------
    from .resume import STEP_VERIFY

    if tracker.skip_step(STEP_VERIFY):
        progress.warning("verify", "Already complete, skipping")
    else:
        from .build_verification import build_verification
        result = build_verification(
            data_dir=data_dir,
            verification_dir=dataset_dir / "verification",
            variables=variables,
            progress=progress,
        )
        manifest = update_manifest_statistics(
            manifest,
            verification_rows=result.get("verification_rows", 0),
        )
        manifest = update_pipeline_status(manifest, "verify")
        tracker.mark_complete(STEP_VERIFY)

    # ------------------------------------------------------------------
    # Step 3: Metrics
    # ------------------------------------------------------------------
    from .resume import STEP_METRICS

    verification_path = dataset_dir / "verification" / "verification.parquet"
    if build_metrics:
        if tracker.skip_step(STEP_METRICS):
            progress.warning("metrics", "Already complete, skipping")
        else:
            from .build_metrics import build_metrics
            result = build_metrics(
                verification_path=verification_path,
                metrics_dir=dataset_dir / "metrics",
                variables=variables,
                progress=progress,
            )
            manifest = update_manifest_statistics(
                manifest,
                metric_groups=result.get("metric_groups", []),
            )
            manifest = update_pipeline_status(manifest, "metrics")
            tracker.mark_complete(STEP_METRICS)

    # ------------------------------------------------------------------
    # Step 4: Events
    # ------------------------------------------------------------------
    from .resume import STEP_EVENTS

    if build_events:
        if tracker.skip_step(STEP_EVENTS):
            progress.warning("events", "Already complete, skipping")
        else:
            from .build_events import build_events
            result = build_events(
                verification_path=verification_path,
                events_dir=dataset_dir / "events",
                progress=progress,
            )
            manifest = update_manifest_statistics(
                manifest,
                event_datasets=result.get("event_datasets", []),
            )
            manifest = update_pipeline_status(manifest, "events")
            tracker.mark_complete(STEP_EVENTS)

    # ------------------------------------------------------------------
    # Step 5: Aggregations
    # ------------------------------------------------------------------
    from .resume import STEP_AGGREGATIONS

    if build_aggregations:
        if tracker.skip_step(STEP_AGGREGATIONS):
            progress.warning("aggregations", "Already complete, skipping")
        else:
            from .build_aggregations import build_aggregations
            result = build_aggregations(
                verification_path=verification_path,
                aggregations_dir=dataset_dir / "aggregates",
                variables=variables,
                progress=progress,
            )
            manifest = update_pipeline_status(manifest, "aggregations")
            tracker.mark_complete(STEP_AGGREGATIONS)

    # ------------------------------------------------------------------
    # Step 6: Report
    # ------------------------------------------------------------------
    from .resume import STEP_REPORT

    if tracker.skip_step(STEP_REPORT):
        progress.warning("report", "Already complete, skipping")
    else:
        from .report import generate_report
        report = generate_report(
            manifest=manifest,
            data_dir=dataset_dir,
        )
        manifest = update_pipeline_status(manifest, "report")
        tracker.mark_complete(STEP_REPORT)

    # ------------------------------------------------------------------
    # Step 7: Final manifest
    # ------------------------------------------------------------------
    from .resume import STEP_MANIFEST

    if tracker.skip_step(STEP_MANIFEST):
        progress.warning("manifest", "Already complete, skipping")
    else:
        manifest = update_pipeline_status(manifest, "manifest")
        write_manifest_to_dataset(manifest, dataset_dir)
        tracker.mark_complete(STEP_MANIFEST)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    progress.section("Dataset preparation complete")
    progress.summary(
        f"Dataset: {dataset_name}",
        Location=location["name"],
        Coordinates=f"{location['lat']}, {location['lon']}",
        TimeRange=f"{start_date} → {end_date}",
        Variables=", ".join(variables),
        Models=", ".join(models),
        DatasetPath=str(dataset_dir),
        PipelineSteps=", ".join(tracker.get_completed()),
    )

    return str(dataset_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_partial_data(dataset_dir: Path) -> bool:
    """Check if a dataset directory already contains partial data."""
    for sub in ["raw/forecasts", "raw/observations", "verification", "metrics"]:
        p = dataset_dir / sub
        if p.exists():
            return True
    return False
