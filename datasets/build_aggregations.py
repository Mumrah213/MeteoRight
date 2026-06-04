"""Compute lead-time degradation and seasonal summaries for the dataset pipeline.

Creates aggregation outputs that show how forecast skill degrades with
lead time, and how performance varies by season. These are standard
diagnostic outputs for research datasets.

Outputs:
- aggregates/lead_time_degradation.parquet — skill score vs lead hours
- aggregates/seasonal_summary.parquet — metrics broken down by season
- aggregates/{var}_by_lead_hours.parquet — per-variable lead time analysis

Usage
-----
>>> from datasets.build_aggregations import build_aggregations

>>> result = build_aggregations(
...     verification_path="./datasets/copenhagen_2y/verification/verification.parquet",
...     aggregations_dir="./datasets/copenhagen_2y/aggregates",
...     variables=["temperature_2m", "precipitation"],
... )
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from datasets.progress import ProgressReporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Aggregation pipeline
# ---------------------------------------------------------------------------

def build_aggregations(
    verification_path: str | Path,
    aggregations_dir: str | Path,
    variables: tuple[str, ...],
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    """Compute lead-time degradation and seasonal summaries.

    Args:
        verification_path: Path to verification parquet file.
        aggregations_dir: Output directory for aggregated data.
        variables: Tuple of variable names.
        progress: Progress reporter instance.

    Returns:
        Dict with keys: created_files (list), total_rows.
    """
    aggregations_dir = Path(aggregations_dir)
    aggregations_dir.mkdir(parents=True, exist_ok=True)

    if progress is None:
        progress = ProgressReporter()

    # ------------------------------------------------------------------
    # Load verification data
    # ------------------------------------------------------------------
    progress.section("Step 6: Computing aggregations")

    progress.step("loading", f"Reading verification from {Path(verification_path).name}...")
    verification_df = pd.read_parquet(verification_path)
    progress.step("loaded", f"{len(verification_df):,} rows")

    result: dict[str, Any] = {"created_files": [], "total_rows": 0}
    created_files: list[str] = []

    # ------------------------------------------------------------------
    # Per-variable lead-time analysis
    # ------------------------------------------------------------------
    progress.step("lead_time", "Computing per-variable lead-time analysis...")

    for var in variables:
        fcst_col = f"forecast_{var}"
        obs_col = f"observed_{var}"

        if fcst_col not in verification_df.columns or obs_col not in verification_df.columns:
            progress.warning(var, "Missing forecast or observed column")
            continue

        # Filter valid pairs and compute RMSE per lead hour
        valid = verification_df[[fcst_col, obs_col, "lead_hours"]].dropna()
        valid = valid[valid["lead_hours"] >= 0]

        if valid.empty:
            progress.warning(var, "No valid data for lead-time analysis")
            continue

        # Compute RMSE per lead hour
        valid["error_sq"] = (valid[fcst_col] - valid[obs_col]) ** 2
        lead_time_agg = valid.groupby("lead_hours").agg(
            count=("lead_hours", "size"),
            rmse=("error_sq", lambda x: (x.mean()) ** 0.5),
        ).reset_index()

        lead_time_agg = lead_time_agg.sort_values("lead_hours")
        lead_time_agg.columns = ["lead_hours", "count", f"{var}_rmse"]
        lead_time_agg.to_parquet(
            aggregations_dir / f"{var}_by_lead_hours.parquet",
            engine="pyarrow", index=False,
        )
        created_files.append(f"{var}_by_lead_hours.parquet")
        progress.step(f"  lead_time/{var}", f"{len(lead_time_agg):,} lead hours")

    # ------------------------------------------------------------------
    # Seasonal summary (per variable)
    # ------------------------------------------------------------------
    progress.step("seasonal", "Computing seasonal summaries...")

    for var in variables:
        fcst_col = f"forecast_{var}"
        obs_col = f"observed_{var}"

        if fcst_col not in verification_df.columns or obs_col not in verification_df.columns:
            continue

        valid = verification_df[[fcst_col, obs_col, "season"]].dropna()
        if valid.empty:
            continue

        valid["error_abs"] = abs(valid[fcst_col] - valid[obs_col])
        seasonal_agg = valid.groupby("season").agg(
            count=("season", "size"),
            mae=("error_abs", "mean"),
            bias=(fcst_col, lambda x: x.mean() - valid[obs_col].mean()),
        ).reset_index()

        seasonal_agg.to_parquet(
            aggregations_dir / f"{var}_by_season.parquet",
            engine="pyarrow", index=False,
        )
        created_files.append(f"{var}_by_season.parquet")
        progress.step(f"  seasonal/{var}", f"{len(seasonal_agg):,} seasons")

    # ------------------------------------------------------------------
    # Combined lead-time degradation summary
    # ------------------------------------------------------------------
    if created_files:
        combined_files = [
            aggregations_dir / f for f in created_files
            if "_by_lead_hours.parquet" in f
        ]
        if combined_files:
            dfs = [pd.read_parquet(f) for f in combined_files]
            if dfs:
                summary = dfs[0]
                for df in dfs[1:]:
                    summary[f"{df.columns[2]}_combined"] = df[df.columns[2]]
                summary.to_parquet(
                    aggregations_dir / "lead_time_degradation.parquet",
                    engine="pyarrow", index=False,
                )
                created_files.append("lead_time_degradation.parquet")
                progress.success("lead_time_summary", f"Combined for {len(dfs)} variables")

    progress.complete(f"Aggregations computed: {len(created_files)} files")

    return {"created_files": created_files, "total_rows": len(verification_df)}
