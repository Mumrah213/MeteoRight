"""Compute basic error columns: forecast - observed.

Only simple subtraction — no aggregated metrics. NaN propagation is automatic.
"""

import logging

import pandas as pd

from .schema import ERROR_SUFFIX, forecast_var, observed_var

logger = logging.getLogger(__name__)


def compute_error_columns(
    df: pd.DataFrame,
    variables: tuple[str, ...],
) -> pd.DataFrame:
    """Compute error = forecast - observed for each variable.

    For each variable in variables, adds:
      {variable}_error = forecast_{variable} - observed_{variable}

    NaN propagation is automatic: if either operand is NaN, error is NaN.
    This includes rows where no observation was found (LEFT JOIN result).

    Args:
        df: Aligned DataFrame with forecast_{var} and observed_{var} columns.
        variables: Tuple of variable names.

    Returns:
        DataFrame with new {var}_error columns appended for each variable.
    """
    df = df.copy()

    for var in variables:
        forecast_col = forecast_var(var)
        observed_col = observed_var(var)
        error_col = f"{var}{ERROR_SUFFIX}"

        if forecast_col not in df.columns or observed_col not in df.columns:
            logger.warning(
                "  Skipping error computation for '%s': columns %s and/or %s not found",
                var,
                forecast_col,
                observed_col,
            )
            continue

        df[error_col] = df[forecast_col] - df[observed_col]
        logger.debug("  Computed %s = %s - %s", error_col, forecast_col, observed_col)

    logger.info("  Computed error columns for %d variables", len(variables))

    return df
