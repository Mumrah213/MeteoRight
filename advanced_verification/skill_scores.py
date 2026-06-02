"""Derived skill scores from confusion matrix counts.

Computes scientifically meaningful forecast skill metrics from the four
fundamental confusion matrix counts (hits, misses, false_alarms, correct_negatives).

All formulas are explicitly documented. Each function returns NaN when the
denominator is zero, with structured warnings.

Metric Formulas
---------------

Hit Rate (POD):
    hits / (hits + misses)
    Range: [0, 1], perfect = 1.0
    "Of all events that occurred, how many did we catch?"
    NaN if hits + misses == 0

False Alarm Rate (FAR):
    false_alarms / (false_alarms + correct_negatives)
    Range: [0, 1], perfect = 0.0
    "Of all non-event days, how many did we falsely alarm?"
    NaN if false_alarms + correct_negatives == 0

Precision (PPV):
    hits / (hits + false_alarms)
    Range: [0, 1], perfect = 1.0
    "Of all days we predicted an event, how many actually occurred?"
    NaN if hits + false_alarms == 0

Critical Success Index (CSI, Threat Score):
    hits / (hits + misses + false_alarms)
    Range: [0, 1], perfect = 1.0
    "Of all forecast events + missed events, how many were correct?"
    Ignores correct negatives (not influenced by climate)
    NaN if hits + misses + false_alarms == 0

Miss Rate:
    misses / (hits + misses)
    Range: [0, 1], perfect = 0.0
    Note: Miss Rate = 1 - Hit Rate

Accuracy:
    (hits + correct_negatives) / (hits + misses + false_alarms + correct_negatives)
    Range: [0, 1], perfect = 1.0
    Problematic for rare events (correct negatives dominate)
    NaN if total == 0

Odds Ratio:
    (hits * correct_negatives) / (misses * false_alarms)
    Range: [0, ∞), perfect = ∞
    NaN if misses * false_alarms == 0

Usage
-----
>>> from advanced_verification.skill_scores import compute_skill_scores
>>> confusion_df = compute_confusion_matrix(df, event_name="frost", groupby=["lead_hours"])
>>> skill_df = compute_skill_scores(confusion_df)
>>> print(skill_df[skill_df["metric_name"] == "csi"])
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    ACCURACY_NAME,
    CORRECT_NEGATIVES_COL,
    CSI_NAME,
    EVENT_NAME_COL,
    EXCLUDED_NAN_COL,
    FALSE_ALARM_RATE_NAME,
    FALSE_ALARMS_COL,
    HIT_RATE_NAME,
    HITS_COL,
    LONG_FORMAT_DIMENSION_COLS,
    METRIC_NAME_COL,
    METRIC_VALUE_COL,
    MISS_RATE_NAME,
    MISSES_COL,
    MISSING_COUNT_COL,
    ODDS_RATIO_NAME,
    PRECISION_NAME,
    SAMPLE_SIZE_COL,
    TOTAL_ROWS_COL,
    VARIABLE_COL,
)

logger = logging.getLogger(__name__)

# Registry of skill score functions
# Maps metric name to function that takes confusion counts and returns (value, excluded)


def _compute_hit_rate(hits: int, misses: int, _fa: int = 0, _cn: int = 0) -> tuple[float, int]:
    """Compute Hit Rate (Probability of Detection).

    Formula: hits / (hits + misses)

    Returns:
        (hit_rate, excluded_count)
    """
    total_events = hits + misses
    if total_events == 0:
        return float("nan"), 0
    return hits / total_events, 0


def _compute_false_alarm_rate(
    hits: int, misses: int, false_alarms: int, correct_negatives: int
) -> tuple[float, int]:
    """Compute False Alarm Rate.

    Formula: false_alarms / (false_alarms + correct_negatives)

    Returns:
        (far, excluded_count).
    """
    total_non_events = false_alarms + correct_negatives
    if total_non_events == 0:
        return float("nan"), 0
    return false_alarms / total_non_events, 0


def _compute_precision(
    hits: int, misses: int, false_alarms: int, correct_negatives: int
) -> tuple[float, int]:
    """Compute Precision (Positive Predictive Value).

    Formula: hits / (hits + false_alarms)

    Returns:
        (precision, excluded_count).
    """
    total_predicted = hits + false_alarms
    if total_predicted == 0:
        return float("nan"), 0
    return hits / total_predicted, 0


def _compute_csi(
    hits: int, misses: int, false_alarms: int, correct_negatives: int = 0
) -> tuple[float, int]:
    """Compute Critical Success Index (Threat Score).

    Formula: hits / (hits + misses + false_alarms)

    Returns:
        (csi, excluded_count).
    """
    total = hits + misses + false_alarms
    if total == 0:
        return float("nan"), 0
    return hits / total, 0


def _compute_miss_rate(hits: int, misses: int, _fa: int = 0, _cn: int = 0) -> tuple[float, int]:
    """Compute Miss Rate.

    Formula: misses / (hits + misses)

    Returns:
        (miss_rate, excluded_count).
    """
    total_events = hits + misses
    if total_events == 0:
        return float("nan"), 0
    return misses / total_events, 0


def _compute_accuracy(
    hits: int, misses: int, false_alarms: int, correct_negatives: int
) -> tuple[float, int]:
    """Compute Accuracy.

    Formula: (hits + correct_negatives) / total

    Args:
        hits: Number of hits.
        misses: Number of misses.
        false_alarms: Number of false alarms.
        correct_negatives: Number of correct negatives.

    Returns:
        (accuracy, excluded_count).
    """
    total = hits + misses + false_alarms + correct_negatives
    if total == 0:
        return float("nan"), 0
    return (hits + correct_negatives) / total, 0


def _compute_odds_ratio(
    hits: int, misses: int, false_alarms: int, correct_negatives: int
) -> tuple[float, int]:
    """Compute Odds Ratio.

    Formula: (hits * correct_negatives) / (misses * false_alarms)

    Args:
        hits: Number of hits.
        misses: Number of misses.
        false_alarms: Number of false alarms.
        correct_negatives: Number of correct negatives.

    Returns:
        (odds_ratio, excluded_count).
    """
    denom = misses * false_alarms
    if denom == 0:
        return float("nan"), 0
    return (hits * correct_negatives) / denom, 0


# Registry — maps metric name to computation function
SKILL_SCORE_FUNCTIONS: dict[str, callable] = {
    HIT_RATE_NAME: _compute_hit_rate,
    FALSE_ALARM_RATE_NAME: _compute_false_alarm_rate,
    PRECISION_NAME: _compute_precision,
    CSI_NAME: _compute_csi,
    MISS_RATE_NAME: _compute_miss_rate,
    ACCURACY_NAME: _compute_accuracy,
    ODDS_RATIO_NAME: _compute_odds_ratio,
}


def compute_skill_scores(confusion_df: pd.DataFrame) -> pd.DataFrame:
    """Compute skill scores from a confusion matrix DataFrame.

    Takes a confusion statistics DataFrame (output of compute_confusion_matrix)
    and produces a long-format skill scores DataFrame with one row per
    (event_name, metric_name, groupby_keys).

    Args:
        confusion_df: Confusion statistics DataFrame from compute_confusion_matrix().
            Must contain columns: hits, misses, false_alarms, correct_negatives.

    Returns:
        Long-format DataFrame with columns:
            [dimension_cols..., event_name, variable, metric_name, metric_value,
             sample_size, total_rows, missing_count]

    Example:
        >>> confusion_df = compute_confusion_matrix(df, event_name="frost", groupby=["lead_hours"])
        >>> skill_df = compute_skill_scores(confusion_df)
        >>> print(skill_df[skill_df["metric_name"] == "csi"])
    """
    # Identify dimension columns (everything except confusion counts and fixed columns)
    fixed_cols = {
        "hits",
        "misses",
        "false_alarms",
        "correct_negatives",
        "excluded_nan",
        "sample_size",
        "total_rows",
        "event_name",
        "variable",
        "threshold",
        "operator",
        METRIC_NAME_COL,
        METRIC_VALUE_COL,
        SAMPLE_SIZE_COL,
        TOTAL_ROWS_COL,
        MISSING_COUNT_COL,
        VARIABLE_COL,
    }

    # Get groupby columns from the input (everything except the confusion columns)
    input_cols = set(confusion_df.columns)
    dim_cols = [
        c for c in input_cols if c not in fixed_cols and c != "threshold" and c != "operator"
    ]

    output_rows: list[dict] = []

    for _, row in confusion_df.iterrows():
        # Extract confusion counts
        hits = int(row.get(HITS_COL, 0))
        misses = int(row.get(MISSES_COL, 0))
        false_alarms = int(row.get(FALSE_ALARMS_COL, 0))
        correct_negatives = int(row.get(CORRECT_NEGATIVES_COL, 0))
        excluded_nan = int(row.get(EXCLUDED_NAN_COL, 0))
        sample_size = hits + misses + false_alarms + correct_negatives
        total_rows = int(row.get(TOTAL_ROWS_COL, 0))
        missing_count = excluded_nan

        # Add dimension columns and event metadata
        base = {
            EVENT_NAME_COL: row[EVENT_NAME_COL],
            VARIABLE_COL: row[VARIABLE_COL],
            SAMPLE_SIZE_COL: sample_size,
            TOTAL_ROWS_COL: total_rows,
            MISSING_COUNT_COL: missing_count,
        }
        # Add dimension columns
        for col in dim_cols:
            base[col] = row[col]

        # Compute each skill score
        for metric_name, compute_func in SKILL_SCORE_FUNCTIONS.items():
            value, excluded = compute_func(hits, misses, false_alarms, correct_negatives)

            result_row = {**base, METRIC_NAME_COL: metric_name, METRIC_VALUE_COL: value}

            # If sample_size is 0, mark as NaN-heavy
            if sample_size == 0 and not np.isnan(value):
                result_row[METRIC_VALUE_COL] = float("nan")
                logger.debug(
                    "Skill score '%s' for event '%s': NaN (no valid samples)",
                    metric_name,
                    row[EVENT_NAME_COL],
                )

            output_rows.append(result_row)

    if not output_rows:
        return pd.DataFrame(
            columns=LONG_FORMAT_DIMENSION_COLS
            + [
                EVENT_NAME_COL,
                VARIABLE_COL,
                METRIC_NAME_COL,
                METRIC_VALUE_COL,
                SAMPLE_SIZE_COL,
                TOTAL_ROWS_COL,
                MISSING_COUNT_COL,
            ]
        )

    result = pd.DataFrame(output_rows)

    # Ensure column order
    cols = LONG_FORMAT_DIMENSION_COLS + [
        EVENT_NAME_COL,
        VARIABLE_COL,
        METRIC_NAME_COL,
        METRIC_VALUE_COL,
        SAMPLE_SIZE_COL,
        TOTAL_ROWS_COL,
        MISSING_COUNT_COL,
    ]
    for col in cols:
        if col not in result.columns:
            result[col] = pd.NA
    result = result[cols]

    logger.info(
        "Skill scores computed: %d confusion groups × %d metrics = %d rows",
        len(confusion_df),
        len(SKILL_SCORE_FUNCTIONS),
        len(output_rows),
    )

    return result


# Re-export for downstream modules
__all__ = [
    "SKILL_SCORE_FUNCTIONS",
    "compute_skill_scores",
    "compute_skill_scores_from_path",
]


def compute_skill_scores_from_path(
    verification_path: str | Path,
    event_name: str,
    groupby: list[str] | None = None,
) -> pd.DataFrame:
    """Compute skill scores from a verification parquet file.

    Convenience function that reads verification data, generates events,
    computes confusion matrix, and derives skill scores.

    Args:
        verification_path: Path to verification parquet file or glob pattern.
        event_name: Name of built-in event.
        groupby: List of dimension columns to group by.

    Returns:
        Long-format skill scores DataFrame.
    """
    verification_path = Path(verification_path)

    if verification_path.is_dir():
        parquet_files = list(verification_path.glob("*.parquet"))
    else:
        parquet_files = list(verification_path.parent.glob(verification_path.name))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at: {verification_path}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    from .confusion import compute_confusion_matrix
    from .events import generate_events

    df = generate_events(df, event_name=event_name)
    confusion_df = compute_confusion_matrix(df, event_name=event_name, groupby=groupby)

    return compute_skill_scores(confusion_df)
