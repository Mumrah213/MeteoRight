"""Deterministic forecast verification metrics.

Pure functions for computing forecast verification metrics from error series.

All functions are:
- NaN-safe: rows with NaN errors are excluded from computation
- Deterministic: same input always produces same output
- Well-documented with explicit formulas
- Unit tested

Metric Formulas
---------------

MAE (Mean Absolute Error):
    mean(|error|)
    = sum(|error_i|) / n

RMSE (Root Mean Squared Error):
    sqrt(mean(error^2))
    = sqrt(sum(error_i^2) / n)

Bias (Mean Signed Error):
    mean(error)
    = sum(error_i) / n
    Positive bias = overprediction
    Negative bias = underprediction

Std (Error Standard Deviation):
    std(error, ddof=1)
    = sqrt(sum((error_i - mean(error))^2) / (n - 1))
    Uses ddof=1 (Bessel-corrected sample standard deviation).
    Requires at least 2 non-NaN values; returns NaN for n < 2.

Missing Data Handling
---------------------
Rows with NaN errors are excluded from metric computation.
The caller (aggregation engine) is responsible for tracking
sample_size, total_rows, and missing_count per bucket.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_mae(errors: pd.Series) -> float:
    """Compute Mean Absolute Error.

    Formula: mean(|error|)

    Args:
        errors: Series of forecast errors (forecast - observed).
            NaN values are excluded from computation.

    Returns:
        MAE as a float. Returns NaN if all values are NaN or series is empty.

    Example:
        >>> import pandas as pd
        >>> errors = pd.Series([1.0, -2.0, 3.0, float('nan')])
        >>> compute_mae(errors)
        2.0
    """
    clean = errors.dropna()
    if clean.empty:
        logger.debug("compute_mae: no non-NaN values, returning NaN")
        return float("nan")
    result = clean.abs().mean()
    logger.debug("compute_mae: n=%d, mae=%.6f", len(clean), result)
    return float(result)


def compute_rmse(errors: pd.Series) -> float:
    """Compute Root Mean Squared Error.

    Formula: sqrt(mean(error^2))

    Args:
        errors: Series of forecast errors (forecast - observed).
            NaN values are excluded from computation.

    Returns:
        RMSE as a float. Returns NaN if all values are NaN or series is empty.

    Example:
        >>> import pandas as pd
        >>> errors = pd.Series([1.0, -2.0, 3.0, float('nan')])
        >>> compute_rmse(errors)
        2.449489742783178
    """
    clean = errors.dropna()
    if clean.empty:
        logger.debug("compute_rmse: no non-NaN values, returning NaN")
        return float("nan")
    result = (clean.pow(2).mean()) ** 0.5
    logger.debug("compute_rmse: n=%d, rmse=%.6f", len(clean), result)
    return float(result)


def compute_bias(errors: pd.Series) -> float:
    """Compute Bias (Mean Signed Error).

    Formula: mean(error)

    Positive bias means forecasts overpredict (forecast > observed).
    Negative bias means forecasts underpredict (forecast < observed).

    Args:
        errors: Series of forecast errors (forecast - observed).
            NaN values are excluded from computation.

    Returns:
        Bias as a float. Returns NaN if all values are NaN or series is empty.

    Example:
        >>> import pandas as pd
        >>> errors = pd.Series([1.0, -2.0, 3.0, float('nan')])
        >>> compute_bias(errors)
        0.6666666666666666
    """
    clean = errors.dropna()
    if clean.empty:
        logger.debug("compute_bias: no non-NaN values, returning NaN")
        return float("nan")
    result = clean.mean()
    logger.debug("compute_bias: n=%d, bias=%.6f", len(clean), result)
    return float(result)


def compute_std(errors: pd.Series) -> float:
    """Compute Error Standard Deviation.

    Formula: std(error, ddof=1) = sqrt(sum((error_i - mean)^2) / (n - 1))

    Uses ddof=1 (Bessel-corrected sample standard deviation).
    Requires at least 2 non-NaN values; returns NaN for n < 2.

    Args:
        errors: Series of forecast errors (forecast - observed).
            NaN values are excluded from computation.

    Returns:
        Standard deviation as a float. Returns NaN if fewer than 2
        non-NaN values exist or all values are NaN.

    Example:
        >>> import pandas as pd
        >>> errors = pd.Series([1.0, -2.0, 3.0, float('nan')])
        >>> compute_std(errors)
        2.0816659994661326
    """
    clean = errors.dropna()
    if len(clean) < 2:
        logger.debug(
            "compute_std: only %d non-NaN values (need >= 2), returning NaN",
            len(clean),
        )
        return float("nan")
    result = clean.std(ddof=1)
    logger.debug("compute_std: n=%d, std=%.6f", len(clean), result)
    return float(result)


# ---------------------------------------------------------------------------
# Metric registry — maps metric names to functions
# ---------------------------------------------------------------------------

METRIC_FUNCTIONS: dict[str, callable] = {
    "mae": compute_mae,
    "rmse": compute_rmse,
    "bias": compute_bias,
    "std": compute_std,
}


def get_metric_function(name: str) -> callable:
    """Get the metric computation function by name.

    Args:
        name: Metric name (e.g. "mae", "rmse", "bias", "std").

    Returns:
        The corresponding metric function.

    Raises:
        ValueError: If the metric name is not recognized.

    Example:
        >>> func = get_metric_function("mae")
        >>> func(pd.Series([1.0, 2.0, 3.0]))
        2.0
    """
    if name not in METRIC_FUNCTIONS:
        raise ValueError(
            f"Unknown metric '{name}'. Available metrics: {', '.join(METRIC_FUNCTIONS.keys())}"
        )
    return METRIC_FUNCTIONS[name]
