"""Canonical verification dataset schema.

Defines the column names, types, and ordering for the verification output.
Every module in this package references these constants for consistency.
"""


# ---------------------------------------------------------------------------
# Provenance columns — always present, never renamed
# ---------------------------------------------------------------------------

PROVENANCE_COLUMNS = [
    "forecast_issue_time",
    "forecast_target_time",
    "observation_time",
    "lead_hours",
    "model",
    "latitude",
    "longitude",
    "elevation",
    "location_id",
]

# Arrow-compatible type hints for documentation / validation
PROVENANCE_TYPES = {
    "forecast_issue_time": "timestamp[us, tz=UTC]",
    "forecast_target_time": "timestamp[us, tz=UTC]",
    "observation_time": "timestamp[us, tz=UTC]",
    "lead_hours": "int32",
    "model": "string",
    "latitude": "float64",
    "longitude": "float64",
    "elevation": "float64",
    "location_id": "string",
}

# ---------------------------------------------------------------------------
# Variable column naming convention
# ---------------------------------------------------------------------------

FORECAST_PREFIX = "forecast_"
OBSERVED_PREFIX = "observed_"
ERROR_SUFFIX = "_error"


def forecast_var(var: str) -> str:
    """Return the canonical column name for a forecast variable."""
    return f"{FORECAST_PREFIX}{var}"


def observed_var(var: str) -> str:
    """Return the canonical column name for an observed variable."""
    return f"{OBSERVED_PREFIX}{var}"


def error_var(var: str) -> str:
    """Return the canonical column name for an error variable."""
    return f"{var}{ERROR_SUFFIX}"


# ---------------------------------------------------------------------------
# Full canonical column ordering
# ---------------------------------------------------------------------------


def canonical_column_order(variables: tuple[str, ...]) -> list[str]:
    """Return the full column order for a canonical verification DataFrame.

    Order:
      1. Provenance columns
      2. Forecast variable columns (one per variable)
      3. Observed variable columns (one per variable)
      4. Error columns (one per variable)

    Args:
        variables: Tuple of variable names (e.g. ("temperature_2m", "precipitation")).

    Returns:
        Ordered list of column names.
    """
    cols = list(PROVENANCE_COLUMNS)
    cols += [forecast_var(v) for v in variables]
    cols += [observed_var(v) for v in variables]
    cols += [error_var(v) for v in variables]
    return cols


def variable_columns(variables: tuple[str, ...]) -> list[str]:
    """Return all variable-related columns (forecast + observed + error).

    Order: forecast_vars, observed_vars, error_vars.
    """
    cols: list[str] = []
    cols += [forecast_var(v) for v in variables]
    cols += [observed_var(v) for v in variables]
    cols += [error_var(v) for v in variables]
    return cols
