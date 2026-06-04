"""Validation layer for canonical verification datasets.

Checks for missing observations, duplicates, impossible lead times,
and timestamp gaps. Produces a structured validation report.
"""


import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation issue dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    """A single validation issue found in the verification dataset."""

    severity: Literal["error", "warning", "info"]
    check: str
    message: str
    count: int  # number of rows affected


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class VerificationValidator:
    """Validate a canonical verification DataFrame."""

    # Lead time bounds
    MIN_LEAD_HOURS = 0
    MAX_LEAD_HOURS = 720  # 30 days — anything larger is absurd

    def __init__(self, variables: tuple[str, ...]):
        """Initialize validator.

        Args:
            variables: Tuple of variable names used in the dataset.
        """
        self.variables = variables

    def validate(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Run all validation checks. Return list of issues found.

        Args:
            df: Canonical verification DataFrame.

        Returns:
            List of ValidationIssue objects.
        """
        checks = [
            self._check_provenance(df),
            self._check_missing_observations(df),
            self._check_duplicate_observations(df),
            self._check_lead_times(df),
            self._check_timestamp_gaps(df),
        ]
        issues = []
        for check_issues in checks:
            issues.extend(check_issues)

        # Summary logging
        errors = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")
        infos = sum(1 for i in issues if i.severity == "info")
        logger.info(
            "Validation: %d errors, %d warnings, %d info",
            errors,
            warnings,
            infos,
        )

        return issues

    def _check_provenance(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Verify the core invariant: (issue_time, target_time, model) uniqueness.

        Each row must have a unique provenance triple. With LEFT JOIN, multiple
        rows can share the same (target_time, model) if multiple issue_times
        exist — that's correct and expected.
        """
        provenance_cols = ["forecast_issue_time", "forecast_target_time", "model"]
        dupes = df.duplicated(subset=provenance_cols, keep="first")
        count = int(dupes.sum())

        if count > 0:
            return [
                ValidationIssue(
                    severity="error",
                    check="provenance_uniqueness",
                    message=(
                        f"{count} rows share the same "
                        f"(forecast_issue_time, forecast_target_time, model) tuple. "
                        f"This violates the forecast provenance invariant."
                    ),
                    count=count,
                )
            ]
        return [
            ValidationIssue(
                severity="info",
                check="provenance_uniqueness",
                message="All forecast rows have unique provenance triples",
                count=0,
            )
        ]

    def _check_missing_observations(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Check for forecast rows with no matching observation.

        Expected with LEFT JOIN — this is the primary purpose of this check.
        Reports count and severity based on proportion of missing observations.
        """
        missing = df["observation_time"].isna()
        count = int(missing.sum())
        total = len(df)
        pct = (count / total * 100) if total > 0 else 0.0

        if count == 0:
            return [
                ValidationIssue(
                    severity="info",
                    check="missing_observations",
                    message="All forecasts have matching observations",
                    count=0,
                )
            ]

        # Severity based on proportion
        if pct > 50:
            severity = "error"
        elif pct > 10:
            severity = "warning"
        else:
            severity = "info"

        return [
            ValidationIssue(
                severity=severity,
                check="missing_observations",
                message=(
                    f"{count} of {total} forecasts ({pct:.1f}%) have no matching "
                    f"observation. These rows have NaN in observed_* columns."
                ),
                count=count,
            )
        ]

    def _check_duplicate_observations(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Check for duplicate observation_time within same location.

        Should not happen — observations are unique per time/location.
        Also checks for unexpected multiplicity from the LEFT JOIN.
        """
        # Check observation_time duplicates (among matched rows)
        matched = df[df["observation_time"].notna()]
        obs_dupes = matched.duplicated(
            subset=["observation_time", "latitude", "longitude"], keep=False
        )
        count = int(obs_dupes.sum())

        if count > 0:
            return [
                ValidationIssue(
                    severity="error",
                    check="duplicate_observations",
                    message=(
                        f"{count} rows have duplicate observation_time at the same "
                        f"location. This indicates a data integrity issue."
                    ),
                    count=count,
                )
            ]

        # Check for LEFT JOIN artifacts: multiple forecast rows per observation
        # This is expected (multiple issue times for same target), but flag if
        # the observation count is vastly disproportionate to forecast count
        obs_count = matched["observation_time"].nunique()
        forecast_count = len(df)
        if obs_count > 0 and forecast_count / obs_count > 100:
            return [
                ValidationIssue(
                    severity="warning",
                    check="duplicate_observations",
                    message=(
                        f"High ratio of forecasts ({forecast_count}) to unique "
                        f"observations ({obs_count}): {forecast_count / obs_count:.0f}x. "
                        f"This may indicate many forecast runs per observation time, "
                        f"which is normal for high-frequency models."
                    ),
                    count=forecast_count - obs_count,
                )
            ]

        return [
            ValidationIssue(
                severity="info",
                check="duplicate_observations",
                message="No duplicate observations found",
                count=0,
            )
        ]

    def _check_lead_times(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Check for impossible lead times.

        Flags:
          - lead_hours < MIN_LEAD_HOURS (should never happen)
          - lead_hours > MAX_LEAD_HOURS (30 days — absurdly large)
        """
        too_small = df["lead_hours"] < self.MIN_LEAD_HOURS
        too_large = df["lead_hours"] > self.MAX_LEAD_HOURS

        count_small = int(too_small.sum())
        count_large = int(too_large.sum())
        total = count_small + count_large

        if total == 0:
            return [
                ValidationIssue(
                    severity="info",
                    check="lead_times",
                    message=(
                        f"All lead_hours in [{self.MIN_LEAD_HOURS}, {self.MAX_LEAD_HOURS}] range"
                    ),
                    count=0,
                )
            ]

        parts = []
        if count_small > 0:
            parts.append(f"{count_small} with lead_hours < {self.MIN_LEAD_HOURS}")
        if count_large > 0:
            parts.append(f"{count_large} with lead_hours > {self.MAX_LEAD_HOURS}")

        return [
            ValidationIssue(
                severity="error",
                check="lead_times",
                message=f"Impossible lead times: {'; '.join(parts)}",
                count=total,
            )
        ]

    def _check_timestamp_gaps(self, df: pd.DataFrame) -> list[ValidationIssue]:
        """Check for gaps in observation time series.

        Flags any gap > 1 hour between consecutive observation times
        within each (model, lat, lon) group. Only checks matched rows.
        """
        matched = df[df["observation_time"].notna()].copy()
        if matched.empty:
            return [
                ValidationIssue(
                    severity="info",
                    check="timestamp_gaps",
                    message="No matched observations to check for gaps",
                    count=0,
                )
            ]

        # Sort by location and time
        matched = matched.sort_values(["latitude", "longitude", "forecast_target_time"])

        # Compute time gaps within each location group
        matched["prev_time"] = matched.groupby(["latitude", "longitude"])["observation_time"].shift(
            1
        )

        matched["gap_hours"] = (matched["observation_time"] - matched["prev_time"]) / pd.Timedelta(
            hours=1
        )

        # Gaps > 1 hour (allow small floating point tolerance)
        gaps = matched[matched["gap_hours"] > 1.5]
        count = len(gaps)

        if count == 0:
            return [
                ValidationIssue(
                    severity="info",
                    check="timestamp_gaps",
                    message="No observation time gaps > 1 hour found",
                    count=0,
                )
            ]

        # Count unique gap locations
        gap_locations = gaps.groupby(["latitude", "longitude"]).ngroups
        return [
            ValidationIssue(
                severity="warning",
                check="timestamp_gaps",
                message=(
                    f"{count} timestamp gaps > 1 hour found across "
                    f"{gap_locations} location(s). Observation coverage may be incomplete."
                ),
                count=count,
            )
        ]


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------


def save_validation_report(
    issues: list[ValidationIssue],
    output_path: Path,
) -> None:
    """Save validation report as JSON.

    Output: output_path/validation_report.json

    Args:
        issues: List of ValidationIssue from VerificationValidator.validate().
        output_path: Path to write the JSON report.
    """
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
