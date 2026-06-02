"""Precompute deterministic metrics for the dataset pipeline.

Computes aggregated verification metrics at multiple granularities:
- by_lead_hours: Metrics broken down by forecast lead time
- by_season: Seasonal metrics
- by_month: Monthly metrics
- by_model: Per-model metrics (if multiple models)

Reuses the existing metrics/aggregations module.

Output:
- metrics/by_lead_hours/{var}_{metrics}_by_lead_hours.parquet
- metrics/by_season/{var}_{metrics}_by_season.parquet
- metrics/by_month/{var}_{metrics}_by_month.parquet
- metrics/by_model/{var}_{metrics}_by_model.parquet

Usage
-----
>>> from datasets.build_metrics import build_metrics

>>> result = build_metrics(
...     verification_path="./datasets/copenhagen_2y/verification/verification.parquet",
...     metrics_dir="./datasets/copenhagen_2y/metrics",
...     variables=["temperature_2m", "precipitation"],
... )
>>> print(result["metric_groups"])
['lead_hours', 'season', 'month', 'model']
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets.progress import ProgressReporter

logger = logging.getLogger(__name__)


def build_metrics(
    verification_path: str | Path,
    metrics_dir: str | Path,
    variables: tuple[str, ...],
    metric_names: tuple[str, ...] | None = None,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Precompute deterministic metrics at multiple granularities.

    Args:
        verification_path: Path to verification parquet file.
        metrics_dir: Output directory for metrics.
        variables: Tuple of variable names.
        metric_names: Metric names to compute (default: mae, rmse, bias).
        progress: Progress reporter instance.

    Returns:
        Dict with key 'metric_groups' — list of groupby dimensions used.
    """
    from metrics.io import read_verification, write_metrics
    from metrics.aggregations import aggregate_metrics

    metrics_dir = Path(metrics_dir)

    if progress is None:
        progress = ProgressReporter()

    if metric_names is None:
        metric_names = ("mae", "rmse", "bias")

    # ------------------------------------------------------------------
    # Load verification data
    # ------------------------------------------------------------------
    progress.section("Step 4: Precomputing metrics")

    progress.step("loading", f"Reading verification from {Path(verification_path).name}...")
    verification_df = read_verification(verification_path)
    progress.step("loaded", f"{len(verification_df):,} verification rows")

    # ------------------------------------------------------------------
    # Compute metrics at each granularity
    # ------------------------------------------------------------------
    metric_groups = ("lead_hours", "season", "month", "model")
    created_files: list[str] = []

    for group_by in metric_groups:
        group_dir = metrics_dir / f"by_{group_by}"
        group_dir.mkdir(parents=True, exist_ok=True)

        progress.step(f"metrics by {group_by}", "Computing...")

        metrics_df = aggregate_metrics(
            verification_df,
            groupby=[group_by],
            metrics=list(metric_names),
            variables=list(variables),
        )

        if metrics_df.empty:
            progress.warning(f"by_{group_by}", "No metric rows produced")
            continue

        # Write one file per variable
        for var in variables:
            var_df = metrics_df[metrics_df["variable"] == var]
            if var_df.empty:
                continue
            filename = f"{var}_{'_'.join(metric_names)}_by_{group_by}.parquet"
            path = group_dir / filename
            write_metrics(var_df, path)
            created_files.append(filename)
            progress.step(f"  {group_by}/{var}", f"{len(var_df):,} rows")

        progress.success(f"by_{group_by}", f"{len(metrics_df):,} metric rows")

    progress.complete(f"Metrics precomputed: {len(created_files)} files")

    return {
        "metric_groups": list(metric_groups),
        "created_files": created_files,
    }
