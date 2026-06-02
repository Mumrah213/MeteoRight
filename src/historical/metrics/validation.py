"""Validation layer for aggregated metrics.

Checks for:
- NaN-heavy groups (>90% missing errors)
- Empty groups (zero sample size with non-zero total)
- Absurd RMSE values (>100x MAE — indicates computation error)
- Negative sample sizes (should never happen)
- Duplicate aggregation keys
- Zero sample size groups

Produces a structured validation report.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from .schemas import (
    METRIC_NAME_COL,
    METRIC_VALUE_COL,
    MISSING_COUNT_COL,
    SAMPLE_SIZE_COL,
    TOTAL_ROWS_COL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Groups with >90% missing errors are flagged
NAN_HEAVY_THRESHOLD = 0.9

# RMSE > 100x MAE is almost certainly a computation error
RMSE_MAE_RATIO_THRESHOLD = 100.0

# Absurd RMSE for temperature (°C) — > 1000°C is impossible
ABSOLUTE_RMSE_THRESHOLD_TEMP = 1000.0

# Absurd RMSE for precipitation (mm) — > 10000mm is impossible
ABSOLUTE_RMSE_THRESHOLD_PRECIP = 10000.0


# ---------------------------------------------------------------------------
# Validation issue dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single validation issue found in the metrics output."""

    severity: Literal["error", "warning", "info"]
    check: str
    message: str
    count: int  # number of rows/records affected


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------


def validate_metrics(
    df: pd.DataFrame,
    variables: list[str] | None = None,
) -> list[ValidationIssue]:
    """Validate aggregated metrics DataFrame.

    Args:
        df: Long-format metrics DataFrame from aggregate_metrics().
        variables: Optional list of variables to check. If None, checks all.

    Returns:
        List of ValidationIssue objects.

    Example:
        >>> issues = validate_metrics(result_df)
        >>> for issue in issues:
        ...     print(f"[{issue.severity}] {issue.check}: {issue.message}")
    """
    if df.empty:
        return [
            ValidationIssue(
                severity="warning",
                check="empty_output",
                message="Metrics DataFrame is empty",
                count=0,
            )
        ]

    checks = [
        _check_negative_sample_size(df),
        _check_duplicate_keys(df),
        _check_nan_heavy_groups(df),
        _check_absurd_rmse(df, variables),
    ]

    issues = []
    for check_issues in checks:
        issues.extend(check_issues)

    # Summary logging
    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    infos = sum(1 for i in issues if i.severity == "info")
    logger.info(
        "Metrics validation: %d errors, %d warnings, %d info",
        errors,
        warnings,
        infos,
    )

    return issues


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_negative_sample_size(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check for negative sample sizes (should never happen)."""
    if SAMPLE_SIZE_COL not in df.columns:
        return [
            ValidationIssue(
                severity="info",
                check="negative_sample_size",
                message="sample_size column not present — skipping check",
                count=0,
            )
        ]

    neg = df[SAMPLE_SIZE_COL] < 0
    count = int(neg.sum())

    if count > 0:
        return [
            ValidationIssue(
                severity="error",
                check="negative_sample_size",
                message=f"{count} rows have negative sample_size",
                count=count,
            )
        ]

    return [
        ValidationIssue(
            severity="info",
            check="negative_sample_size",
            message="No negative sample sizes found",
            count=0,
        )
    ]


def _check_duplicate_keys(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check for duplicate aggregation keys.

    Each (dimension_cols, variable, metric_name) should be unique.
    """
    # Find dimension columns (everything except the fixed columns)
    fixed_cols = {
        METRIC_NAME_COL,
        METRIC_VALUE_COL,
        SAMPLE_SIZE_COL,
        TOTAL_ROWS_COL,
        MISSING_COUNT_COL,
        "variable",
    }
    dim_cols = [c for c in df.columns if c not in fixed_cols]

    if not dim_cols:
        return [
            ValidationIssue(
                severity="info",
                check="duplicate_key",
                message="No dimension columns to check for duplicates",
                count=0,
            )
        ]

    # Check for duplicates within each (dim_cols, variable, metric_name)
    # Include 'variable' in the key — same dimension+metric can appear
    # for different variables
    check_cols = dim_cols + ["variable", METRIC_NAME_COL]
    dupes = df.duplicated(subset=check_cols, keep="first")
    count = int(dupes.sum())

    if count > 0:
        return [
            ValidationIssue(
                severity="error",
                check="duplicate_key",
                message=f"{count} rows have duplicate aggregation keys",
                count=count,
            )
        ]

    return [
        ValidationIssue(
            severity="info",
            check="duplicate_key",
            message="No duplicate aggregation keys found",
            count=0,
        )
    ]


def _check_nan_heavy_groups(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check for groups where >90% of errors are missing.

    These groups have very little data and their metrics may not be
    statistically meaningful.
    """
    if SAMPLE_SIZE_COL not in df.columns or TOTAL_ROWS_COL not in df.columns:
        return [
            ValidationIssue(
                severity="info",
                check="nan_heavy_group",
                message="sample_size/total_rows columns not present — skipping check",
                count=0,
            )
        ]

    total = df[TOTAL_ROWS_COL]
    sample = df[SAMPLE_SIZE_COL]

    # Only check groups that have at least some total data
    has_data = total > 0
    if not has_data.any():
        return [
            ValidationIssue(
                severity="info",
                check="nan_heavy_group",
                message="No groups with data to check",
                count=0,
            )
        ]

    # NaN-heavy: sample < threshold * total
    nan_heavy = sample < NAN_HEAVY_THRESHOLD * total
    count = int(nan_heavy.sum())

    if count > 0:
        return [
            ValidationIssue(
                severity="warning",
                check="nan_heavy_group",
                message=f"{count} groups have >{NAN_HEAVY_THRESHOLD * 100:.0f}% missing errors",
                count=count,
            )
        ]

    return [
        ValidationIssue(
            severity="info",
            check="nan_heavy_group",
            message="No NaN-heavy groups found",
            count=0,
        )
    ]


def _check_absurd_rmse(
    df: pd.DataFrame,
    variables: list[str] | None = None,
) -> list[ValidationIssue]:
    """Check for absurd RMSE values.

    RMSE should never be > 100x MAE (unless all errors are zero, which
    would make both zero). Also checks absolute thresholds.
    """
    if METRIC_VALUE_COL not in df.columns or METRIC_NAME_COL not in df.columns:
        return [
            ValidationIssue(
                severity="info",
                check="absurd_rmse",
                message="metric_value/metric_name columns not present — skipping check",
                count=0,
            )
        ]

    issues: list[ValidationIssue] = []

    # Check RMSE > 100x MAE within same (dimension_cols, variable)
    fixed_cols = {
        METRIC_NAME_COL,
        METRIC_VALUE_COL,
        SAMPLE_SIZE_COL,
        TOTAL_ROWS_COL,
        MISSING_COUNT_COL,
        "variable",
    }
    dim_cols = [c for c in df.columns if c not in fixed_cols]

    if dim_cols:
        # Pivot to compare RMSE vs MAE within groups
        pivot = df.pivot_table(
            index=dim_cols,
            columns=METRIC_NAME_COL,
            values=METRIC_VALUE_COL,
            aggfunc="first",
        )

        if "rmse" in pivot.columns and "mae" in pivot.columns:
            # Avoid division by zero
            mae = pivot["mae"].replace(0, float("nan"))
            ratio = pivot["rmse"] / mae
            absurd = ratio > RMSE_MAE_RATIO_THRESHOLD
            count = int(absurd.sum())

            if count > 0:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        check="absurd_rmse",
                        message=f"{count} groups have RMSE > {RMSE_MAE_RATIO_THRESHOLD}x MAE",
                        count=count,
                    )
                )

    # Check absolute thresholds
    if variables:
        for var in variables:
            threshold = (
                ABSOLUTE_RMSE_THRESHOLD_TEMP
                if "temp" in var.lower()
                else ABSOLUTE_RMSE_THRESHOLD_PRECIP
            )
            rmse_rows = df[(df[METRIC_NAME_COL] == "rmse") & (df["variable"] == var)]
            absurd_abs = rmse_rows[METRIC_VALUE_COL] > threshold
            count = int(absurd_abs.sum())

            if count > 0:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        check="absurd_rmse",
                        message=f"{count} RMSE values for '{var}' exceed {threshold}",
                        count=count,
                    )
                )

    if not issues:
        issues.append(
            ValidationIssue(
                severity="info",
                check="absurd_rmse",
                message="No absurd RMSE values found",
                count=0,
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------


def save_validation_report(
    issues: list[ValidationIssue],
    output_path: str | Path,
) -> Path:
    """Save validation report as JSON.

    Args:
        issues: List of ValidationIssue from validate_metrics().
        output_path: Output file path (.json).

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_issues": len(issues),
        "errors": sum(1 for i in issues if i.severity == "error"),
        "warnings": sum(1 for i in issues if i.severity == "warning"),
        "info": sum(1 for i in issues if i.severity == "info"),
        "issues": [asdict(issue) for issue in issues],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Saved validation report: %s", output_path)
    return output_path
