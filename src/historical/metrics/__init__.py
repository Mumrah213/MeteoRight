"""Weather Forecast Verification Metrics & Analysis.

Phase 3 — deterministic forecast verification metrics, aggregation pipelines,
and metric storage/output.

Reads canonical verification parquet datasets (produced by Phase 2),
computes deterministic forecast metrics, aggregates across dimensions,
and outputs derived metric parquet datasets.

Never mutates or overwrites verification datasets.
"""

__version__ = "0.1.0"

# Public API
from .aggregations import aggregate_metrics
from .io import export_csv, read_verification, write_metrics
from .metrics import compute_bias, compute_mae, compute_rmse, compute_std
from .summaries import (
    summary_by_lead_time,
    summary_by_model_and_lead,
    summary_by_month,
    summary_by_season,
)
from .validation import validate_metrics

__all__ = [
    "__version__",
    "aggregate_metrics",
    "compute_bias",
    "compute_mae",
    "compute_rmse",
    "compute_std",
    "export_csv",
    "read_verification",
    "summary_by_lead_time",
    "summary_by_model_and_lead",
    "summary_by_month",
    "summary_by_season",
    "validate_metrics",
    "write_metrics",
]
