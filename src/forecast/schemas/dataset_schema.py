"""Schema constants for curated research datasets.

Defines:
- Dataset directory layout constants
- Manifest JSON schema (field names and types)
- Required vs optional fields
- Metadata subdirectory structure

This module provides the canonical definitions that all other dataset
modules reference for consistency.

Directory Layout
----------------
Each prepared dataset is a self-contained directory:

    datasets/
    └── {dataset_name}/
        ├── raw/
        │   ├── forecasts/          # Downloaded forecast parquet (partitioned)
        │   └── observations/       # Downloaded observation parquet (partitioned)
        ├── verification/           # Built verification dataset
        ├── metrics/
        │   ├── by_lead_hours/      # Per-lead-hour metrics
        │   ├── by_season/          # Seasonal metrics
        │   ├── by_month/           # Monthly metrics
        │   └── by_model/           # Per-model metrics
        ├── events/                 # Precomputed event datasets
        ├── aggregates/             # Lead-time degradation, seasonal summaries
        ├── notebooks/              # Starter notebook for each variable
        ├── manifests/              # JSON manifest(s)
        ├── metadata/               # Attribution, provenance, lineage
        └── reports/                # Summary report (markdown)

Manifest Schema
---------------
Every dataset has a ``manifest.json`` in the ``manifests/`` subdirectory:

    {
        "dataset_name": str,
        "version": str,
        "pipeline_version": str,
        "location": { "name": str, "latitude": float, "longitude": float },
        "time_range": { "start_date": str, "end_date": str },
        "variables": list[str],
        "models": list[str],
        "generated_at": str,           # ISO 8601 UTC timestamp
        "statistics": { ... },         # Row counts, event info
        "provenance": { ... },         # Source API info, license
        "pipeline": { ... }            # Steps completed, timing
    }
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Directory layout constants
# ---------------------------------------------------------------------------

DIR_RAW = "raw"
DIR_RAW_FORECASTS = "forecasts"
DIR_RAW_OBSERVATIONS = "observations"

DIR_VERIFICATION = "verification"
DIR_METRICS = "metrics"
DIR_EVENTS = "events"
DIR_AGGREGATES = "aggregates"
DIR_NOTEBOOKS = "notebooks"
DIR_MANIFESTS = "manifests"
DIR_METADATA = "metadata"
DIR_REPORTS = "reports"

# Metric subdirectories
DIR_METRICS_LEAD_HOURS = "by_lead_hours"
DIR_METRICS_SEASON = "by_season"
DIR_METRICS_MONTH = "by_month"
DIR_METRICS_MODEL = "by_model"

# ---------------------------------------------------------------------------
# Manifest field names
# ---------------------------------------------------------------------------

MANIFEST_FIELD_DATASET_NAME = "dataset_name"
MANIFEST_FIELD_VERSION = "version"
MANIFEST_FIELD_PIPELINE_VERSION = "pipeline_version"
MANIFEST_FIELD_LOCATION = "location"
MANIFEST_FIELD_TIME_RANGE = "time_range"
MANIFEST_FIELD_VARIABLES = "variables"
MANIFEST_FIELD_MODELS = "models"
MANIFEST_FIELD_GENERATED_AT = "generated_at"
MANIFEST_FIELD_STATISTICS = "statistics"
MANIFEST_FIELD_PROVENANCE = "provenance"
MANIFEST_FIELD_PIPELINE = "pipeline"

# Location subfields
LOC_NAME = "name"
LOC_LATITUDE = "latitude"
LOC_LONGITUDE = "longitude"
LOC_TIMEZONE = "timezone"

# Time range subfields
TR_START_DATE = "start_date"
TR_END_DATE = "end_date"

# Statistics subfields
STAT_FORECAST_ROWS = "forecast_rows"
STAT_OBSERVATION_ROWS = "observation_rows"
STAT_VERIFICATION_ROWS = "verification_rows"
STAT_EVENT_DATASETS = "event_datasets"
STAT_METRIC_GROUPS = "metric_groups"

# Provenance subfields
PROV_FORECAST_SOURCE = "forecast_source"
PROV_OBSERVATION_SOURCE = "observation_source"
PROV_LICENSE = "license"

# Pipeline subfields
PIPELINE_STEPS_COMPLETED = "steps_completed"
PIPELINE_START_TIME = "start_time"
PIPELINE_END_TIME = "end_time"

# ---------------------------------------------------------------------------
# Manifest schema validation
# ---------------------------------------------------------------------------

MANIFEST_REQUIRED_FIELDS = [
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
]

LOCATION_REQUIRED = [LOC_NAME, LOC_LATITUDE, LOC_LONGITUDE]
TIME_RANGE_REQUIRED = [TR_START_DATE, TR_END_DATE]
STATISTICS_REQUIRED = [
    STAT_FORECAST_ROWS,
    STAT_OBSERVATION_ROWS,
    STAT_VERIFICATION_ROWS,
]
PROVENANCE_REQUIRED = [
    PROV_FORECAST_SOURCE,
    PROV_OBSERVATION_SOURCE,
    PROV_LICENSE,
]
