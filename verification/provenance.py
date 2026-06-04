"""Forecast provenance integrity checks.

Verifies that the core invariant — (issue_time, target_time, model) —
is preserved through the alignment process. This is assertion, not
transformation.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provenance checks
# ---------------------------------------------------------------------------


def verify_provenance_integrity(df: pd.DataFrame) -> None:
    """Assert that no forecast provenance was lost during alignment.

    Raises ValueError if:
      - Duplicate (forecast_issue_time, forecast_target_time, model) rows exist
      - lead_hours is inconsistent with (target_time - issue_time) beyond 1 hour
      - Any lead_hours < 0

    Args:
        df: Aligned DataFrame from align_forecasts_with_observations().

    Raises:
        ValueError: If provenance invariant is violated.
    """
    # Check 1: uniqueness of provenance triple
    provenance_cols = ["forecast_issue_time", "forecast_target_time", "model"]
    dupes = df.duplicated(subset=provenance_cols, keep=False)
    if dupes.any():
        raise ValueError(
            f"Provenance violation: {dupes.sum()} duplicate "
            f"({provenance_cols[0]}, {provenance_cols[1]}, {provenance_cols[2]}) "
            f"rows found. This should not happen — each forecast run for a "
            f"given target time is unique."
        )

    # Check 2: lead_hours consistency
    expected_lead = (
        (df["forecast_target_time"] - df["forecast_issue_time"]) / pd.Timedelta(hours=1)
    ).astype("int64")
    diff = (df["lead_hours"] - expected_lead).abs()
    inconsistent = diff > 1  # Allow 1 hour rounding tolerance
    if inconsistent.any():
        raise ValueError(
            f"Provenance violation: {inconsistent.sum()} rows have lead_hours "
            f"inconsistent with (target_time - issue_time) by more than 1 hour."
        )

    # Check 3: non-negative lead hours
    negative = df["lead_hours"] < 0
    if negative.any():
        raise ValueError(
            f"Provenance violation: {negative.sum()} rows have negative "
            f"lead_hours. Forecast target_time must be >= issue_time."
        )

    logger.info("  Provenance integrity verified — no violations found")


def compute_lead_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute and validate lead_hours from timestamps.

    This is a sanity check / recomputation, not a new calculation.
    If lead_hours is already present and consistent, it is preserved as-is.
    If missing or inconsistent, it is recomputed.

    Args:
        df: DataFrame with forecast_issue_time and forecast_target_time.

    Returns:
        DataFrame with lead_hours column (int32).
    """
    if "lead_hours" not in df.columns:
        # Recompute from timestamps
        df = df.copy()
        df["lead_hours"] = (
            (df["forecast_target_time"] - df["forecast_issue_time"]) / pd.Timedelta(hours=1)
        ).astype("int32")
        logger.info("  Recomputed lead_hours from timestamps")
    else:
        # Validate existing lead_hours
        expected_lead = (
            (df["forecast_target_time"] - df["forecast_issue_time"]) / pd.Timedelta(hours=1)
        ).astype("int64")
        mismatch = (df["lead_hours"] - expected_lead).abs() > 1
        if mismatch.any():
            logger.warning(
                "  Recomputing lead_hours for %d rows with timestamp mismatch",
                mismatch.sum(),
            )
            df = df.copy()
            df["lead_hours"] = expected_lead.astype("int32")

    return df
