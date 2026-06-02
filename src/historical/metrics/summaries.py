"""Convenience summary functions.

Wraps aggregate_metrics() for the four most common aggregation patterns:
1. By lead_hours
2. By model + lead_hours
3. By month
4. By season

These are thin wrappers — they exist for convenience, not as a replacement
for the general aggregate_metrics() function.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .aggregations import aggregate_metrics
from .io import write_metrics

logger = logging.getLogger(__name__)

# Default metric set
DEFAULT_METRICS = ["mae", "rmse", "bias"]

# Default variables
DEFAULT_VARIABLES = ["temperature_2m", "precipitation"]


def summary_by_lead_time(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    variables: list[str] | None = None,
    output_dir: str | Path | None = None,
    output_filename: str | None = None,
) -> pd.DataFrame:
    """Compute metrics grouped by lead_hours.

    This is the primary aggregation — it answers:
    "How does forecast accuracy change with lead time?"

    Args:
        df: Verification DataFrame.
        metrics: Metric names to compute. Defaults to [mae, rmse, bias].
        variables: Variable names to compute. Defaults to [temperature_2m, precipitation].
        output_dir: If provided, write results to parquet.
        output_filename: Output filename (default: "{variable}_by_lead_time.parquet").

    Returns:
        Long-format metrics DataFrame.
    """
    metrics = metrics or DEFAULT_METRICS
    variables = variables or DEFAULT_VARIABLES

    logger.info("Computing summary by lead_hours")
    result = aggregate_metrics(
        df,
        groupby=["lead_hours"],
        metrics=metrics,
        variables=variables,
    )

    if output_dir is not None:
        for var in variables:
            var_result = result[result["variable"] == var]
            if not var_result.empty:
                fn = output_filename or f"{var}_by_lead_time.parquet"
                out_path = Path(output_dir) / "by_lead_time" / fn
                write_metrics(
                    var_result,
                    out_path,
                    metadata={
                        "groupby": ["lead_hours"],
                        "metrics": metrics,
                        "variable": var,
                    },
                )

    return result


def summary_by_model_and_lead(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    variables: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compute metrics grouped by (model, lead_hours).

    This answers: "Which model performs best at each lead time?"

    Args:
        df: Verification DataFrame.
        metrics: Metric names to compute. Defaults to [mae, rmse, bias].
        variables: Variable names to compute. Defaults to [temperature_2m, precipitation].
        output_dir: If provided, write results to parquet.

    Returns:
        Long-format metrics DataFrame.
    """
    metrics = metrics or DEFAULT_METRICS
    variables = variables or DEFAULT_VARIABLES

    logger.info("Computing summary by model + lead_hours")
    result = aggregate_metrics(
        df,
        groupby=["model", "lead_hours"],
        metrics=metrics,
        variables=variables,
    )

    if output_dir is not None:
        for var in variables:
            var_result = result[result["variable"] == var]
            if not var_result.empty:
                write_metrics(
                    var_result,
                    Path(output_dir) / "by_model_and_lead" / f"{var}_parquet",
                    metadata={
                        "groupby": ["model", "lead_hours"],
                        "metrics": metrics,
                        "variable": var,
                    },
                )

    return result


def summary_by_month(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    variables: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compute metrics grouped by month.

    This answers: "Are forecasts seasonally biased?"

    Args:
        df: Verification DataFrame.
        metrics: Metric names to compute. Defaults to [mae, rmse, bias].
        variables: Variable names to compute. Defaults to [temperature_2m, precipitation].
        output_dir: If provided, write results to parquet.

    Returns:
        Long-format metrics DataFrame.
    """
    metrics = metrics or DEFAULT_METRICS
    variables = variables or DEFAULT_VARIABLES

    logger.info("Computing summary by month")
    result = aggregate_metrics(
        df,
        groupby=["month"],
        metrics=metrics,
        variables=variables,
    )

    if output_dir is not None:
        for var in variables:
            var_result = result[result["variable"] == var]
            if not var_result.empty:
                write_metrics(
                    var_result,
                    Path(output_dir) / "by_month" / f"{var}_parquet",
                    metadata={
                        "groupby": ["month"],
                        "metrics": metrics,
                        "variable": var,
                    },
                )

    return result


def summary_by_season(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    variables: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compute metrics grouped by meteorological season.

    This answers: "Are forecasts seasonally biased?"

    Seasons:
        DJF: December, January, February
        MAM: March, April, May
        JJA: June, July, August
        SON: September, October, November

    Args:
        df: Verification DataFrame.
        metrics: Metric names to compute. Defaults to [mae, rmse, bias].
        variables: Variable names to compute. Defaults to [temperature_2m, precipitation].
        output_dir: If provided, write results to parquet.

    Returns:
        Long-format metrics DataFrame.
    """
    metrics = metrics or DEFAULT_METRICS
    variables = variables or DEFAULT_VARIABLES

    logger.info("Computing summary by season")
    result = aggregate_metrics(
        df,
        groupby=["season"],
        metrics=metrics,
        variables=variables,
    )

    if output_dir is not None:
        for var in variables:
            var_result = result[result["variable"] == var]
            if not var_result.empty:
                write_metrics(
                    var_result,
                    Path(output_dir) / "by_season" / f"{var}_parquet",
                    metadata={
                        "groupby": ["season"],
                        "metrics": metrics,
                        "variable": var,
                    },
                )

    return result
