"""Validation layer for skill outputs.

Checks for:
- Impossible confusion counts (negative values, overflow)
- Inconsistent totals (count sum != total_rows)
- NaN-heavy groups (>50% excluded)
- Empty event buckets (no valid samples)
- Undefined metrics (NaN due to zero denominators)

Produces a structured validation report (mirrors P3 validation.py pattern).

Validation Rules
----------------

| Check                | SEVERITY | Condition                                    |
|---------------------|----------|----------------------------------------------|
| negative_count      | Error    | Any count < 0                                |
| count_overflow      | Error    | Sum of counts > total_rows                   |
| count_underflow     | Error    | Sum of counts + excluded_nan < total_rows    |
| nan_heavy_group     | Warning  | excluded_nan / total_rows > 0.5              |
| empty_event_bucket  | Warning  | All four counts are zero                     |
| undefined_hit_rate  | Info     | hits + misses == 0                           |
| undefined_far       | Info     | false_alarms + correct_negatives == 0        |
| undefined_precision | Info     | hits + false_alarms == 0                     |
| undefined_csi       | Info     | hits + misses + false_alarms == 0            |
| undefined_accuracy  | Info     | sample_size == 0                             |
| undefined_odds      | Info     | misses * false_alarms == 0                   |

Usage
-----
>>> from verification.validation import validate_skill_scores, validate_confusion
>>> confusion_df = compute_confusion_matrix(df, event_name="frost", groupby=["lead_hours"])
>>> issues = validate_confusion(confusion_df)
>>> for issue in issues:
...     print(f"[{issue.severity}] {issue.check}: {issue.message}")
"""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

SEVERITY = Literal["error", "warning", "info"]


@dataclass
class ValidationIssue:
    """A single validation issue found in skill outputs."""

    severity: SEVERITY
    check: str
    message: str
    count: int  # number of rows/records affected


# ---------------------------------------------------------------------------
# Confusion matrix validation
# ---------------------------------------------------------------------------


def validate_confusion(
    df: pd.DataFrame,
) -> list[ValidationIssue]:
    """Validate a confusion matrix DataFrame.

    Args:
        df: Confusion statistics DataFrame from compute_confusion_matrix().

    Returns:
        List of ValidationIssue objects.
    """
    if df.empty:
        return [
            ValidationIssue(
                severity="warning",
                check="empty_output",
                message="Confusion DataFrame is empty",
                count=0,
            )
        ]

    checks = [
        _check_negative_counts(df),
        _check_count_overflow(df),
        _check_nan_heavy_groups(df),
        _check_empty_event_buckets(df),
    ]

    issues: list[ValidationIssue] = []
    for check_issues in checks:
        issues.extend(check_issues)

    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    infos = sum(1 for i in issues if i.severity == "info")
    logger.info(
        "Confusion validation: %d errors, %d warnings, %d info",
        errors,
        warnings,
        infos,
    )

    return issues


# ---------------------------------------------------------------------------
# Skill scores validation
# ---------------------------------------------------------------------------


def validate_skill_scores(
    df: pd.DataFrame,
) -> list[ValidationIssue]:
    """Validate a skill scores DataFrame.

    Checks for undefined metrics (NaN due to zero denominators).

    Args:
        df: Skill scores DataFrame from compute_skill_scores().

    Returns:
        List of ValidationIssue objects.
    """
    if df.empty:
        return [
            ValidationIssue(
                severity="warning",
                check="empty_output",
                message="Skill scores DataFrame is empty",
                count=0,
            )
        ]

    issues: list[ValidationIssue] = []

    # Check for NaN metric values
    nan_mask = df["metric_value"].isna()
    if nan_mask.any():
        int(nan_mask.sum())
        # Group by metric_name to provide more detail
        nan_by_metric = df[nan_mask].groupby("metric_name").size()
        for metric, count in nan_by_metric.items():
            issues.append(
                ValidationIssue(
                    severity="info",
                    check=f"undefined_{metric}",
                    message=f"{count} groups have undefined {metric} (zero denominator)",
                    count=count,
                )
            )

    # Check for NaN-heavy groups
    if "missing_count" in df.columns and "total_rows" in df.columns:
        total = df["total_rows"]
        missing = df["missing_count"]
        nan_heavy = (total > 0) & (missing / total > 0.5)
        count = int(nan_heavy.sum())
        if count > 0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    check="nan_heavy_group",
                    message=f"{count} groups have >50% missing data in skill scores",
                    count=count,
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Individual confusion checks
# ---------------------------------------------------------------------------


def _check_negative_counts(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check for negative counts (should never happen)."""
    count_cols = ["hits", "misses", "false_alarms", "correct_negatives", "excluded_nan"]
    issues: list[ValidationIssue] = []

    for col in count_cols:
        if col not in df.columns:
            continue
        neg = df[col] < 0
        count = int(neg.sum())
        if count > 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    check="negative_count",
                    message=f"{count} rows have negative {col}",
                    count=count,
                )
            )

    if not issues:
        issues.append(
            ValidationIssue(
                severity="info",
                check="negative_count",
                message="No negative counts found",
                count=0,
            )
        )

    return issues


def _check_count_overflow(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check that count sum doesn't exceed total_rows."""
    issues: list[ValidationIssue] = []

    if not all(
        c in df.columns
        for c in [
            "hits",
            "misses",
            "false_alarms",
            "correct_negatives",
            "excluded_nan",
            "total_rows",
        ]
    ):
        issues.append(
            ValidationIssue(
                severity="info",
                check="count_overflow",
                message="Missing columns — skipping overflow check",
                count=0,
            )
        )
        return issues

    counts_sum = (
        df["hits"]
        + df["misses"]
        + df["false_alarms"]
        + df["correct_negatives"]
        + df["excluded_nan"]
    )
    overflow = counts_sum > df["total_rows"]
    count = int(overflow.sum())

    if count > 0:
        issues.append(
            ValidationIssue(
                severity="error",
                check="count_overflow",
                message=f"{count} rows have count sum > total_rows",
                count=count,
            )
        )
    else:
        issues.append(
            ValidationIssue(
                severity="info",
                check="count_overflow",
                message="No count overflow detected",
                count=0,
            )
        )

    return issues


def _check_nan_heavy_groups(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check for groups with >50% NaN exclusions."""
    issues: list[ValidationIssue] = []

    if "excluded_nan" not in df.columns or "total_rows" not in df.columns:
        issues.append(
            ValidationIssue(
                severity="info",
                check="nan_heavy_group",
                message="excluded_nan/total_rows columns not present — skipping",
                count=0,
            )
        )
        return issues

    has_data = df["total_rows"] > 0
    if not has_data.any():
        issues.append(
            ValidationIssue(
                severity="info",
                check="nan_heavy_group",
                message="No groups with data to check",
                count=0,
            )
        )
        return issues

    nan_heavy = has_data & (df["excluded_nan"] / df["total_rows"] > 0.5)
    count = int(nan_heavy.sum())

    if count > 0:
        issues.append(
            ValidationIssue(
                severity="warning",
                check="nan_heavy_group",
                message=f"{count} groups have >50% NaN exclusions",
                count=count,
            )
        )
    else:
        issues.append(
            ValidationIssue(
                severity="info",
                check="nan_heavy_group",
                message="No NaN-heavy groups found",
                count=0,
            )
        )

    return issues


def _check_empty_event_buckets(df: pd.DataFrame) -> list[ValidationIssue]:
    """Check for groups with zero valid samples."""
    issues: list[ValidationIssue] = []

    required = ["hits", "misses", "false_alarms", "correct_negatives"]
    if not all(c in df.columns for c in required):
        issues.append(
            ValidationIssue(
                severity="info",
                check="empty_event_bucket",
                message="Missing count columns — skipping empty bucket check",
                count=0,
            )
        )
        return issues

    # Empty if all four counts are zero (but total_rows > 0, meaning everything was NaN)
    all_zero = df[required].sum(axis=1) == 0
    has_data = df["total_rows"] > 0
    empty = all_zero & has_data
    count = int(empty.sum())

    if count > 0:
        issues.append(
            ValidationIssue(
                severity="warning",
                check="empty_event_bucket",
                message=f"{count} groups have zero valid samples (all NaN)",
                count=count,
            )
        )
    else:
        issues.append(
            ValidationIssue(
                severity="info",
                check="empty_event_bucket",
                message="No empty event buckets found",
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
        issues: List of ValidationIssue objects.
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
        "issues": [
            {
                "severity": issue.severity,
                "check": issue.check,
                "message": issue.message,
                "count": issue.count,
            }
            for issue in issues
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Saved validation report: %s", output_path)
    return output_path
