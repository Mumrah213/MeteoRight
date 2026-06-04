"""Lead-time degradation analysis.

Computes quantitative measures of forecast degradation:
- Degradation rate (error increase per lead time)
- Half-skill lead time (when error doubles)
- Asymptotic skill level

Usage
-----
>>> from weather_analysis.analysis.degradation import (
...     compute_degradation_rate,
...     compute_half_skill_lead_time,
... )
>>> rate = compute_degradation_rate(df, "temperature_2m", "mae")
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Degradation rate
# ---------------------------------------------------------------------------


def compute_degradation_rate(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    model: str | None = None,
    lead_range: tuple[int, int] | None = None,
) -> float:
    """Compute the degradation rate (error increase per lead hour).

    Uses linear regression on MAE vs lead_hours to estimate the rate
    at which forecast error increases with lead time.

    Args:
        df: Metrics DataFrame (long format).
        variable: Variable name.
        metric: Metric name (e.g. "mae").
        model: Optional model filter.
        lead_range: Optional (min_lead, max_lead) range for fitting.

    Returns:
        Degradation rate in metric_units per hour.
        Returns NaN if insufficient data.
    """
    data = df[(df["variable"] == variable) & (df["metric_name"] == metric)].copy()

    if model:
        data = data[data["model"] == model]

    if lead_range:
        data = data[(data["lead_hours"] >= lead_range[0]) & (data["lead_hours"] <= lead_range[1])]

    if len(data) < 3:
        logger.warning("compute_degradation_rate: insufficient data (%d points)", len(data))
        return float("nan")

    x = data["lead_hours"].values
    y = data["metric_value"].values

    # Linear regression: y = a + b*x, where b is the degradation rate
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return float("nan")

    # Use numpy polyfit for linear regression
    coeffs = np.polyfit(x_clean, y_clean, 1)
    rate = float(coeffs[0])  # slope = degradation rate

    logger.info(
        "Degradation rate for %s / %s: %.4f per hour",
        variable,
        metric,
        rate,
    )
    return rate


def compute_degradation_summary(
    df: pd.DataFrame,
    variables: list[str],
    metrics: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Compute degradation rates for multiple variables and metrics.

    Args:
        df: Metrics DataFrame.
        variables: List of variable names.
        metrics: List of metric names. Defaults to ["mae"].
        models: List of model names.

    Returns:
        DataFrame with columns: variable, metric, model, degradation_rate.
    """
    if metrics is None:
        metrics = ["mae"]

    rows = []

    data = df[df["variable"].isin(variables)]
    if models:
        data = data[data["model"].isin(models)]

    for variable in variables:
        var_data = data[data["variable"] == variable]

        for metric in metrics:
            metric_data = var_data[var_data["metric_name"] == metric]

            if "model" in metric_data.columns:
                model_list = metric_data["model"].unique()
            else:
                model_list = ["_all_"]

            for model in model_list:
                rate = compute_degradation_rate(
                    metric_data,
                    variable,
                    metric,
                    model=model if len(model_list) > 1 else None,
                )
                rows.append(
                    {
                        "variable": variable,
                        "metric_name": metric,
                        "model": model if len(model_list) > 1 else "",
                        "degradation_rate": rate,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Half-skill lead time
# ---------------------------------------------------------------------------


def compute_half_skill_lead_time(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    model: str | None = None,
) -> float | None:
    """Compute the lead time at which skill degrades by half.

    "Half-skill" is defined as the lead time where the error has
    doubled from its minimum value. This is a practical measure
    of forecast usefulness.

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metric: Metric name.
        model: Optional model filter.

    Returns:
        Lead time in hours where error doubles, or None if not computable.
    """
    data = df[(df["variable"] == variable) & (df["metric_name"] == metric)].copy()

    if model:
        data = data[data["model"] == model]

    if len(data) < 2:
        return None

    data = data.sort_values("lead_hours")
    values = data["metric_value"].values
    lead_hours = data["lead_hours"].values

    min_error = np.nanmin(values)
    if min_error <= 0:
        return None

    threshold = 2.0 * min_error

    # Find first lead time where error >= threshold
    for lh, val in zip(lead_hours, values, strict=False):
        if val >= threshold:
            logger.info(
                "Half-skill lead time for %s / %s: %.0f h (error goes from %.2f to %.2f)",
                variable,
                metric,
                lh,
                min_error,
                val,
            )
            return float(lh)

    logger.info(
        "Half-skill lead time for %s / %s: not reached (max error %.2f < 2× min error %.2f)",
        variable,
        metric,
        np.nanmax(values),
        min_error,
    )
    return None


# ---------------------------------------------------------------------------
# Skill score
# ---------------------------------------------------------------------------


def compute_skill_score(
    df: pd.DataFrame,
    variable: str,
    metric: str,
    model: str,
    reference_model: str,
    lead_hours: int | None = None,
) -> float:
    """Compute relative skill score between two models.

    Skill score = 1 - (model_error / reference_error)

    Positive score = model is better than reference
    Negative score = model is worse than reference

    Args:
        df: Metrics DataFrame.
        variable: Variable name.
        metric: Metric name (lower is better for MAE/RMSE).
        model: Model name to evaluate.
        reference_model: Reference model name.
        lead_hours: Optional lead time filter.

    Returns:
        Skill score (float). NaN if reference model has zero error.
    """
    mask = (df["variable"] == variable) & (df["metric_name"] == metric)

    model_data = df[mask & (df["model"] == model)]
    ref_data = df[mask & (df["model"] == reference_model)]

    if lead_hours is not None:
        model_data = model_data[model_data["lead_hours"] == lead_hours]
        ref_data = ref_data[ref_data["lead_hours"] == lead_hours]

    if model_data.empty or ref_data.empty:
        return float("nan")

    model_error = model_data["metric_value"].mean()
    ref_error = ref_data["metric_value"].mean()

    if ref_error == 0:
        return float("nan")

    skill = 1.0 - (model_error / ref_error)

    logger.info(
        "Skill score: %s vs %s for %s / %s = %.3f",
        model,
        reference_model,
        variable,
        metric,
        skill,
    )
    return float(skill)
