"""Phase 2: Post-transformation validation.

After raw response → canonical records, these checks ensure the
transformation was correct and lossless.

Three categories:
  STRUCTURAL  — time arrays consistent, monotonic, correct lead_times
  SEMANTIC    — nulls preserved, variable expansion correct
  CROSS-CHECK — canonical records reconstruct original ForecastRun exactly
"""

from __future__ import annotations

import logging
from datetime import datetime

from ..schemas.validation import (
    SemanticIssue,
    StructuralIssue,
    ValidationResult,
)
from .models import CanonicalForecastRecord, ForecastRun
from .pipeline import _parse_iso_time

logger = logging.getLogger(__name__)


# ── Structural Checks ──────────────────────────────────────────────────────


def check_time_consistency(run: ForecastRun) -> list[StructuralIssue]:
    """Verify all variable arrays match the time array length."""
    issues: list[StructuralIssue] = []
    n = run.time_count

    for var_name, values in run.variables.items():
        if len(values) != n:
            issues.append(
                StructuralIssue(
                    rule="array_length_consistency",
                    field=var_name,
                    message=(
                        f"Variable '{var_name}' has {len(values)} values "
                        f"but time array has {n} entries"
                    ),
                    raw_value={"expected": n, "actual": len(values)},
                )
            )

    return issues


def check_time_monotonic(run: ForecastRun) -> list[StructuralIssue]:
    """Verify times are strictly increasing."""
    issues: list[StructuralIssue] = []
    times = run.times

    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            issues.append(
                StructuralIssue(
                    rule="timestamps_monotonic",
                    field="time",
                    message=(
                        f"Non-monotonic timestamps at index {i}: '{times[i]}' <= '{times[i - 1]}'"
                    ),
                    raw_value={"earlier": times[i - 1], "later": times[i]},
                )
            )

    return issues


def check_lead_time_correctness(
    run: ForecastRun,
    records: list[CanonicalForecastRecord],
) -> list[StructuralIssue]:
    """Verify every record has correct lead_hours."""
    issues: list[StructuralIssue] = []

    for rec in records:
        expected_lead = (rec.valid_time - rec.run_time).total_seconds() / 3600
        if abs(expected_lead - rec.lead_hours) > 1e-6:
            issues.append(
                StructuralIssue(
                    rule="lead_time_correctness",
                    field=rec.variable,
                    message=(
                        f"Lead time mismatch at {rec.valid_time}: "
                        f"computed={expected_lead:.1f}h, stored={rec.lead_hours}h"
                    ),
                    raw_value={"expected": expected_lead, "actual": rec.lead_hours},
                )
            )

    return issues


# ── Semantic Checks ────────────────────────────────────────────────────────


def check_null_preservation(
    run: ForecastRun,
    records: list[CanonicalForecastRecord],
) -> list[SemanticIssue]:
    """Verify no nulls were silently dropped during flattening.

    Uses valid_time matching (not lead_hours index) to find records,
    which is correct for arbitrary run_time offsets.
    """
    issues: list[SemanticIssue] = []

    # Build a lookup from (variable, valid_time) -> record for fast matching
    record_lookup: dict[tuple[str, datetime], CanonicalForecastRecord] = {}
    for rec in records:
        record_lookup[(rec.variable, rec.valid_time)] = rec

    for var_name in run.variable_names:
        values = run.variables[var_name]
        for i, val in enumerate(values):
            if val is None:
                # Match by valid_time, not lead_hours index
                valid_time = _parse_iso_time(run.times[i])
                rec = record_lookup.get((var_name, valid_time))
                if rec is not None and rec.value is not None:
                    issues.append(
                        SemanticIssue(
                            rule="null_preservation",
                            variable=var_name,
                            value=None,
                            message=(
                                f"Null value at index {i} was not preserved — "
                                f"record has value {rec.value}"
                            ),
                        )
                    )

    return issues


def check_variable_expansion(
    run: ForecastRun,
    records: list[CanonicalForecastRecord],
) -> list[SemanticIssue]:
    """Verify every (time, variable) pair produced exactly one record."""
    issues: list[SemanticIssue] = []
    n = run.time_count

    # Build a lookup from datetime -> time_index for efficient matching
    time_to_idx: dict[datetime, int] = {}
    for i, t in enumerate(run.times):
        time_to_idx[_parse_iso_time(t)] = i

    # Count records per (time_index, variable)
    seen: set[tuple[int, str]] = set()
    for rec in records:
        try:
            time_idx = time_to_idx[rec.valid_time]
            key = (time_idx, rec.variable)
            if key in seen:
                issues.append(
                    SemanticIssue(
                        rule="variable_expansion",
                        variable=rec.variable,
                        value=None,
                        message=(
                            f"Duplicate record for variable '{rec.variable}' "
                            f"at time index {time_idx}"
                        ),
                    )
                )
            seen.add(key)
        except KeyError:
            issues.append(
                SemanticIssue(
                    rule="variable_expansion",
                    variable=rec.variable,
                    value=None,
                    message="Record valid_time not found in run times",
                )
            )

    # Check completeness: every (time_idx, var) should have a record
    expected_count = n * len(run.variable_names)
    if len(seen) != expected_count:
        issues.append(
            SemanticIssue(
                rule="variable_expansion",
                variable="all",
                value=None,
                message=(
                    f"Expected {expected_count} records "
                    f"({n} times × {len(run.variable_names)} vars), "
                    f"got {len(seen)}"
                ),
            )
        )

    return issues


def check_unit_propagation(
    run: ForecastRun,
    records: list[CanonicalForecastRecord],
) -> list[SemanticIssue]:
    """Verify units were correctly propagated to all records."""
    issues: list[SemanticIssue] = []

    for rec in records:
        expected_unit = run.units.get(rec.variable)
        if rec.unit != expected_unit:
            issues.append(
                SemanticIssue(
                    rule="unit_propagation",
                    variable=rec.variable,
                    value=None,
                    message=(f"Unit mismatch: expected '{expected_unit}', got '{rec.unit}'"),
                )
            )

    return issues


# ── Cross-Check ────────────────────────────────────────────────────────────


def check_round_trip(
    run: ForecastRun,
    records: list[CanonicalForecastRecord],
) -> list[StructuralIssue | SemanticIssue]:
    """Verify canonical records reconstruct the original ForecastRun exactly.

    This is the most critical check: it ensures no data was lost or
    corrupted during flattening. Verifies:
    - Value reconstruction matches originals
    - lead_hours are consistent with (valid_time - run_time)
    """
    issues: list[StructuralIssue | SemanticIssue] = []
    n = run.time_count

    # Build a lookup from datetime -> index
    time_to_idx: dict[datetime, int] = {}
    for i, t in enumerate(run.times):
        time_to_idx[_parse_iso_time(t)] = i

    # Reconstruct variable arrays from records
    reconstructed: dict[str, list[float | None]] = {var: [None] * n for var in run.variable_names}

    for rec in records:
        try:
            idx = time_to_idx[rec.valid_time]
            reconstructed[rec.variable][idx] = rec.value
        except KeyError:
            issues.append(
                StructuralIssue(
                    rule="round_trip",
                    field=rec.variable,
                    message="Record valid_time not found in run times",
                )
            )

    # Compare reconstructed vs original
    for var_name in run.variable_names:
        orig = run.variables.get(var_name)
        recon = reconstructed.get(var_name)

        if orig is None or recon is None:
            continue

        for i in range(n):
            o = orig[i]
            r = recon[i]
            if o != r:
                # Allow float comparison for non-null values
                if o is not None and r is not None and abs(o - r) < 1e-9:
                    continue
                issues.append(
                    StructuralIssue(
                        rule="round_trip",
                        field=var_name,
                        message=(f"Value mismatch at index {i}: original={o}, reconstructed={r}"),
                        raw_value={"original": o, "reconstructed": r},
                    )
                )

    # Verify lead_hours consistency
    for rec in records:
        expected_lead = (rec.valid_time - rec.run_time).total_seconds() / 3600
        if abs(expected_lead - rec.lead_hours) > 1e-6:
            issues.append(
                StructuralIssue(
                    rule="round_trip_lead_hours",
                    field=rec.variable,
                    message=(
                        f"Lead time mismatch at {rec.valid_time}: "
                        f"computed={expected_lead:.3f}h, stored={rec.lead_hours}h"
                    ),
                    raw_value={"expected": expected_lead, "actual": rec.lead_hours},
                )
            )

    return issues


# ── Full Validation Pipeline ───────────────────────────────────────────────


def validate_transformation(
    run: ForecastRun,
    records: list[CanonicalForecastRecord],
) -> ValidationResult:
    """Run all post-transformation checks on a single run.

    Returns a ValidationResult with all issues found.
    """
    result = ValidationResult(
        valid=True,
        provider=run.provider,
        endpoint="canonical",
        rules_checked=0,
        rules_passed=0,
    )

    # Structural checks
    for check_fn in [check_time_consistency, check_time_monotonic]:
        issues = check_fn(run)
        result.rules_checked += 1
        if not issues:
            result.mark_passed()
        else:
            for issue in issues:
                result.add_issue(issue)

    issues = check_lead_time_correctness(run, records)
    result.rules_checked += 1
    if not issues:
        result.mark_passed()
    else:
        for issue in issues:
            result.add_issue(issue)

    # Semantic checks
    for check_fn in [check_null_preservation, check_variable_expansion, check_unit_propagation]:
        issues = check_fn(run, records)
        result.rules_checked += 1
        if not issues:
            result.mark_passed()
        else:
            for issue in issues:
                result.add_issue(issue)

    # Cross-check
    issues = check_round_trip(run, records)
    result.rules_checked += 1
    if not issues:
        result.mark_passed()
    else:
        for issue in issues:
            result.add_issue(issue)

    result.valid = not result.has_critical_issues

    logger.debug(
        "Validation for %s run: %d rules, %d issues",
        run.provider,
        result.rules_checked,
        len(result.issues),
    )

    return result


def validate_all(
    runs: list[ForecastRun],
    records_map: dict[int, list[CanonicalForecastRecord]],
) -> list[ValidationResult]:
    """Validate multiple runs and their canonical records.

    records_map: index → (run_index, records)
    """
    results: list[ValidationResult] = []

    for i, run in enumerate(runs):
        if i in records_map:
            result = validate_transformation(run, records_map[i])
        else:
            result = ValidationResult(
                valid=False,
                issues=[
                    StructuralIssue(
                        rule="missing_records",
                        field="all",
                        message=f"No canonical records found for run index {i}",
                    )
                ],
                provider=run.provider,
                endpoint="canonical",
                rules_checked=1,
                rules_passed=0,
            )
        results.append(result)

    return results
