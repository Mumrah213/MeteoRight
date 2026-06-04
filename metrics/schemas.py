"""Schema constants and column naming conventions.

Extends the P2 verification schema with metric-specific column names,
long-format output schemas, and utility helpers.

Invariant: canonical verification datasets (P2) are never mutated.
All metrics are derived artifacts stored separately.
"""

# ---------------------------------------------------------------------------
# Metric name constants
# ---------------------------------------------------------------------------

MAE = "mae"
RMSE = "rmse"
BIAS = "bias"
STD = "std"

ALL_METRICS = (MAE, RMSE, BIAS, STD)

# ---------------------------------------------------------------------------
# Aggregation metadata column names
# ---------------------------------------------------------------------------

METRIC_NAME_COL = "metric_name"
METRIC_VALUE_COL = "metric_value"
SAMPLE_SIZE_COL = "sample_size"
TOTAL_ROWS_COL = "total_rows"
MISSING_COUNT_COL = "missing_count"
VARIABLE_COL = "variable"

# ---------------------------------------------------------------------------
# Aggregation dimension column names
# ---------------------------------------------------------------------------

LEAD_HOURS_COL = "lead_hours"
MODEL_COL = "model"
MONTH_COL = "month"
SEASON_COL = "season"

# ---------------------------------------------------------------------------
# Long-format output schema
# ---------------------------------------------------------------------------

# The long-format schema for aggregated metrics.
# Every aggregation output row contains exactly one metric value.
#
# Columns:
#   - Dimension columns (lead_hours, model, month, season) — present if
#     used in the groupby, otherwise omitted.
#   - variable: the variable name (e.g. "temperature_2m")
#   - metric_name: one of mae, rmse, bias, std
#   - metric_value: the computed metric value (float64)
#   - sample_size: count of non-NaN errors used in computation
#   - total_rows: total rows in the bucket (including NaN errors)
#   - missing_count: rows with NaN error (total_rows - sample_size)

LONG_FORMAT_DIMENSION_COLS = [
    LEAD_HOURS_COL,
    MODEL_COL,
    MONTH_COL,
    SEASON_COL,
]

LONG_FORMAT_FIXED_COLS = [
    VARIABLE_COL,
    METRIC_NAME_COL,
    METRIC_VALUE_COL,
    SAMPLE_SIZE_COL,
    TOTAL_ROWS_COL,
    MISSING_COUNT_COL,
]


def long_format_columns(dimensions: list[str] | None = None) -> list[str]:
    """Return the full column order for a long-format aggregation output.

    Args:
        dimensions: List of dimension column names to include
            (e.g. ["lead_hours", "model"]). If None, no dimension columns
            are included (useful for single-group outputs).

    Returns:
        Ordered list of column names: [dimensions..., variable, metric_name,
        metric_value, sample_size, total_rows, missing_count].
    """
    if dimensions is None:
        dimensions = []
    # Filter to valid dimension columns
    valid_dims = [d for d in dimensions if d in LONG_FORMAT_DIMENSION_COLS]
    return valid_dims + LONG_FORMAT_FIXED_COLS


# ---------------------------------------------------------------------------
# Season mapping (meteorological)
# ---------------------------------------------------------------------------

SEASON_MAP: dict[int, str] = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

# ---------------------------------------------------------------------------
# Validation issue types
# ---------------------------------------------------------------------------

VALIDATION_CHECKS = [
    "nan_heavy_group",
    "empty_group",
    "absurd_rmse",
    "negative_sample_size",
    "duplicate_key",
    "zero_sample_size",
]
