"""Baseline forecast models for relative skill scoring.

Provides simple baseline forecasts against which to compare operational
forecast skill. Baselines are defined using only historical observation data,
never the forecasts being evaluated.

Baseline Types
--------------

Climatology Baseline
    Estimates event probability from historical observation frequency.
    For each group (e.g., month), predicts the event if P(event) > 0.5.

Persistence Baseline
    Uses the previous time step's observed state as the forecast for
    the current time step. Only applicable for consecutive forecast runs.

Design Principle
----------------
Baselines are computed separately from the forecast model's skill scores.
The architecture supports computing both in a single pipeline, but the
baseline computation does not depend on forecast data at all.

Future Extensibility
--------------------
This module is designed to support additional baseline types in the future:
- Seasonal climatology (separate from monthly)
- Trend-adjusted climatology
- Random forecast (baseline for comparison)
- No-skill baseline (always predict mean probability)

Usage
-----
>>> from advanced_verification.baselines import compute_climatology_skill
>>> # Compute climatology baseline skill for comparison
>>> baseline_skill = compute_climatology_skill(verification_df, event_name="frost")
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from .schema import (
    LONG_FORMAT_DIMENSION_COLS,
    SAMPLE_SIZE_COL,
    TOTAL_ROWS_COL,
)

logger = logging.getLogger(__name__)

BaselineType = Literal["climatology", "persistence"]


def compute_climatology_event_probability(
    df: pd.DataFrame,
    event_name: str,
    groupby: list[str] | None = None,
) -> pd.DataFrame:
    """Compute event probability from historical observations.

    Estimates P(event) = count(observed=event) / total_observed for each group.

    This is used to construct a climatology baseline forecast that always
    predicts the event when P(event) > 0.5 and never otherwise.

    Args:
        df: DataFrame with event_observed_{event_name} column.
        event_name: Event name.
        groupby: Groupby dimensions (e.g., ["month", "lead_hours"]).
            If None, computes global probability.

    Returns:
        DataFrame with columns: groupby_dims..., event_name, climatology_prob, sample_size, total_rows.
    """
    obs_col = f"event_observed_{event_name}"

    if obs_col not in df.columns:
        raise ValueError(f"Missing column '{obs_col}'. Run generate_events() first.")

    # Filter to rows with valid observations
    valid = df[obs_col].notna() | df[obs_col].eq(True) | df[obs_col].eq(False)

    if groupby:
        grouped = df[valid].groupby(groupby, sort=True)
        results = []
        for group_key, group_df in grouped:
            key_dict = (
                dict(zip(groupby, group_key, strict=False))
                if isinstance(group_key, tuple)
                else {groupby[0]: group_key}
            )
            n = len(group_df)
            prob = group_df[obs_col].mean() if n > 0 else float("nan")
            row = {
                **key_dict,
                EVENT_NAME_COL: event_name,
                "climatology_prob": prob,
                SAMPLE_SIZE_COL: n,
                TOTAL_ROWS_COL: n,
            }
            results.append(row)
    else:
        n = len(df[valid])
        prob = df[valid][obs_col].mean() if n > 0 else float("nan")
        results = [
            {
                EVENT_NAME_COL: event_name,
                "climatology_prob": prob,
                SAMPLE_SIZE_COL: n,
                TOTAL_ROWS_COL: n,
            }
        ]

    result = pd.DataFrame(results)
    logger.info(
        "Climatology: %s — P(event)=%.3f across %d rows",
        event_name,
        prob if pd.notna(prob) else float("nan"),
        n,
    )
    return result


# Re-export for use in baselines
from .schema import EVENT_NAME_COL


def compute_climatology_skill(
    confusion_df: pd.DataFrame,
    climatology_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute relative skill compared to climatology baseline.

    For each group, compares the forecast model's CSI (or other metric)
    against the climatology baseline's CSI.

    Relative Skill = 1 - (skill_forecast - skill_baseline) / (1 - skill_baseline)

    Where skill is CSI (Critical Success Index).

    Args:
        confusion_df: Confusion matrix DataFrame from compute_confusion_matrix().
        climatology_df: Climatology probabilities from compute_climatology_event_probability().

    Returns:
        DataFrame with relative skill scores for each group.
    """
    # Compute climatology CSI for each group
    baseline_rows = []
    forecast_rows = []

    for _, clim_row in climatology_df.iterrows():
        group_key = tuple(
            clim_row[col] for col in LONG_FORMAT_DIMENSION_COLS if col in climatology_df.columns
        )

        # Find matching confusion row
        match = confusion_df[confusion_df[EVENT_NAME_COL] == clim_row[EVENT_NAME_COL]]
        if group_key:
            for dim_col in LONG_FORMAT_DIMENSION_COLS:
                if dim_col in confusion_df.columns:
                    match = match[match[dim_col] == clim_row.get(dim_col, None)]

        if match.empty:
            continue

        conf = match.iloc[0]
        hits = int(conf.get("hits", 0))
        misses = int(conf.get("misses", 0))
        false_alarms = int(conf.get("false_alarms", 0))
        correct_negatives = int(conf.get("correct_negatives", 0))

        # Forecast CSI
        denom = hits + misses + false_alarms
        forecast_csi = hits / denom if denom > 0 else float("nan")

        # Climatology CSI: P(event) predicts event if P > 0.5
        p_event = clim_row["climatology_prob"]
        if p_event > 0.5:
            # Climatology always predicts event → hits = observed events, false_alarms = non-events
            total_observed = hits + misses  # total actual events
            total_non_events = false_alarms + correct_negatives  # total actual non-events
            clim_hits = int(total_observed * p_event)
            clim_false_alarms = int(total_non_events * p_event)
            clim_misses = total_observed - clim_hits
            clim_csi = (
                clim_hits / (clim_hits + clim_misses + clim_false_alarms)
                if (clim_hits + clim_misses + clim_false_alarms) > 0
                else float("nan")
            )
        else:
            # Climatology never predicts event → all are misses or correct negatives
            clim_csi = 0.0  # No hits at all

        baseline_rows.append(
            {
                "climatology_csi": clim_csi,
                **{
                    dim_col: clim_row.get(dim_col)
                    for dim_col in LONG_FORMAT_DIMENSION_COLS
                    if dim_col in climatology_df.columns
                },
            }
        )
        forecast_rows.append(
            {
                "forecast_csi": forecast_csi,
                **{
                    dim_col: conf.get(dim_col)
                    for dim_col in LONG_FORMAT_DIMENSION_COLS
                    if dim_col in confusion_df.columns
                },
            }
        )

    if baseline_rows and forecast_rows:
        baseline = pd.DataFrame(baseline_rows)
        forecast = pd.DataFrame(forecast_rows)
        # Compute relative skill
        denom = 1.0 - baseline["climatology_csi"].replace(1.0, float("nan"))
        relative_skill = 1.0 - (forecast["forecast_csi"] - baseline["climatology_csi"]) / denom
        forecast["relative_skill"] = relative_skill
        return forecast

    return pd.DataFrame()
