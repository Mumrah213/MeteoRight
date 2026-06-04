"""Schema constants and column naming conventions for event verification.

Defines the column names, types, and ordering for event verification data,
confusion matrix statistics, and skill score outputs.

Every module in this package references these constants for consistency.

Event DataFrame Schema
----------------------
An event DataFrame extends the canonical verification DataFrame with binary
event columns. Each event type adds two columns:

    event_forecast_{name}  : bool — did the forecast predict the event?
    event_observed_{name}  : bool — did the event actually occur?

Example for a "frost" event:

    forecast_issue_time | ... | forecast_temperature_2m | observed_temperature_2m |
    event_forecast_frost | event_observed_frost
    ------------------- | ... | ----------------------- | ----------------------- |
    2025-01-01 00:00    | ... | -1.5                    | -2.0                    |
    True                | True

Confusion Statistics Schema
---------------------------
One row per (event_name, groupby_keys). Contains raw counts:

    event_name, variable, threshold, operator,
    [groupby_dims...],
    hits, misses, false_alarms, correct_negatives,
    excluded_nan, sample_size, total_rows

Skill Scores Schema
-------------------
Long-format, matching P3 metric output style. One row per
(event_name, metric_name, groupby_keys):

    event_name, [groupby_dims...],
    metric_name, metric_value, sample_size, total_rows, missing_count
"""

# ---------------------------------------------------------------------------
# Event column naming convention
# ---------------------------------------------------------------------------

EVENT_FORECAST_PREFIX = "event_forecast_"
EVENT_OBSERVED_PREFIX = "event_observed_"

# Valid event names (built-in events)
BUILTIN_EVENTS: tuple[str, ...] = (
    "frost",
    "heavy_precip",
    "hot_day",
    "no_rain",
)

# Event definitions for built-in events
# Each defines: variable, threshold, operator
BUILTIN_EVENT_DEFS: dict[str, dict] = {
    "frost": {
        "variable": "temperature_2m",
        "threshold": 0.0,
        "operator": "le",  # <= : temperature at or below freezing
        "description": "Temperature at or below freezing (frost event)",
    },
    "heavy_precip": {
        "variable": "precipitation",
        "threshold": 5.0,
        "operator": "ge",  # >= : heavy precipitation
        "description": "Heavy precipitation event (>= 5 mm)",
    },
    "hot_day": {
        "variable": "temperature_2m",
        "threshold": 30.0,
        "operator": "ge",  # >= : hot day
        "description": "Hot day event (>= 30 °C)",
    },
    "no_rain": {
        "variable": "precipitation",
        "threshold": 0.1,
        "operator": "lt",  # < : effectively dry (light precip ignored)
        "description": "Dry condition — precipitation effectively zero",
    },
    "cold": {
        "variable": "temperature_2m",
        "threshold": 3.0,
        "operator": "le",  # <= : cold conditions
        "description": "Cold event (temperature at or below 3 °C)",
    },
}

# Valid comparison operators
VALID_OPERATORS: tuple[str, ...] = ("le", "lt", "ge", "gt", "eq", "ne")

# Operator display names for documentation
OPERATOR_NAMES: dict[str, str] = {
    "le": "<=",
    "lt": "<",
    "ge": ">=",
    "gt": ">",
    "eq": "==",
    "ne": "!=",
}

# ---------------------------------------------------------------------------
# Confusion matrix column names
# ---------------------------------------------------------------------------

EVENT_NAME_COL = "event_name"
THRESHOLD_COL = "threshold"
OPERATOR_COL = "operator"
VARIABLE_COL = "variable"
HITS_COL = "hits"
MISSES_COL = "misses"
FALSE_ALARMS_COL = "false_alarms"
CORRECT_NEGATIVES_COL = "correct_negatives"
EXCLUDED_NAN_COL = "excluded_nan"
SAMPLE_SIZE_COL = "sample_size"
TOTAL_ROWS_COL = "total_rows"

# Confusion matrix columns — always present in confusion output
CONFUSION_COLUMNS: tuple[str, ...] = (
    EVENT_NAME_COL,
    VARIABLE_COL,
    THRESHOLD_COL,
    OPERATOR_COL,
    HITS_COL,
    MISSES_COL,
    FALSE_ALARMS_COL,
    CORRECT_NEGATIVES_COL,
    EXCLUDED_NAN_COL,
    SAMPLE_SIZE_COL,
    TOTAL_ROWS_COL,
)

# ---------------------------------------------------------------------------
# Skill score metric names
# ---------------------------------------------------------------------------

HIT_RATE_NAME = "hit_rate"  # Probability of Detection (POD)
FALSE_ALARM_RATE_NAME = "false_alarm_rate"
PRECISION_NAME = "precision"  # Positive Predictive Value (PPV)
CSI_NAME = "csi"  # Critical Success Index (Threat Score)
MISS_RATE_NAME = "miss_rate"
ACCURACY_NAME = "accuracy"
ODDS_RATIO_NAME = "odds_ratio"

# All supported skill score metric names
ALL_SKILL_METRICS: tuple[str, ...] = (
    HIT_RATE_NAME,
    FALSE_ALARM_RATE_NAME,
    PRECISION_NAME,
    CSI_NAME,
    MISS_RATE_NAME,
    ACCURACY_NAME,
    ODDS_RATIO_NAME,
)

# ---------------------------------------------------------------------------
# Skill scores output schema (long-format, matches P3 metric output)
# ---------------------------------------------------------------------------

METRIC_NAME_COL = "metric_name"
METRIC_VALUE_COL = "metric_value"
MISSING_COUNT_COL = "missing_count"

LONG_FORMAT_DIMENSION_COLS = [
    "lead_hours",
    "model",
    "month",
    "season",
]

LONG_FORMAT_FIXED_COLS = [
    EVENT_NAME_COL,
    VARIABLE_COL,
    METRIC_NAME_COL,
    METRIC_VALUE_COL,
    SAMPLE_SIZE_COL,
    TOTAL_ROWS_COL,
    MISSING_COUNT_COL,
]


def skill_columns(dimensions: list[str] | None = None) -> list[str]:
    """Return the full column order for a skill scores output DataFrame.

    Args:
        dimensions: List of dimension columns to include
            (e.g. ["lead_hours", "model"]). If None, no dimension columns
            are included (single-group output).

    Returns:
        Ordered list of column names.
    """
    if dimensions is None:
        dimensions = []
    valid_dims = [d for d in dimensions if d in LONG_FORMAT_DIMENSION_COLS]
    return valid_dims + LONG_FORMAT_FIXED_COLS


def confusion_columns(dimensions: list[str] | None = None) -> list[str]:
    """Return the full column order for a confusion statistics output DataFrame.

    Args:
        dimensions: List of dimension columns to include
            (e.g. ["lead_hours", "model"]). If None, no dimension columns
            are included.

    Returns:
        Ordered list of column names.
    """
    if dimensions is None:
        dimensions = []
    valid_dims = [d for d in dimensions if d in LONG_FORMAT_DIMENSION_COLS]
    return valid_dims + list(CONFUSION_COLUMNS)


# ---------------------------------------------------------------------------
# Season mapping (meteorological) — shared with metrics module
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
# Event column names helper
# ---------------------------------------------------------------------------


def forecast_event_col(event_name: str) -> str:
    """Return the column name for a forecast event flag.

    Args:
        event_name: Event name (e.g., "frost").

    Returns:
        Column name (e.g., "event_forecast_frost").
    """
    return f"{EVENT_FORECAST_PREFIX}{event_name}"


def observed_event_col(event_name: str) -> str:
    """Return the column name for an observed event flag.

    Args:
        event_name: Event name (e.g., "frost").

    Returns:
        Column name (e.g., "event_observed_frost").
    """
    return f"{EVENT_OBSERVED_PREFIX}{event_name}"


def get_event_definition(event_name: str) -> dict:
    """Get the definition for a built-in event.

    Args:
        event_name: Event name (e.g., "frost").

    Returns:
        Dict with keys: variable, threshold, operator, description.

    Raises:
        ValueError: If event_name is not a built-in event.
    """
    if event_name not in BUILTIN_EVENT_DEFS:
        raise ValueError(
            f"Unknown event '{event_name}'. Built-in events: {', '.join(BUILTIN_EVENTS)}"
        )
    return dict(BUILTIN_EVENT_DEFS[event_name])
