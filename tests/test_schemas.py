"""Tests for schema constants and helpers."""

from metrics.schemas import (
    ALL_METRICS,
    LEAD_HOURS_COL,
    LONG_FORMAT_FIXED_COLS,
    METRIC_NAME_COL,
    METRIC_VALUE_COL,
    MISSING_COUNT_COL,
    MODEL_COL,
    MONTH_COL,
    SAMPLE_SIZE_COL,
    SEASON_COL,
    SEASON_MAP,
    SEASON_ORDER,
    TOTAL_ROWS_COL,
    VARIABLE_COL,
    long_format_columns,
)


class TestMetricConstants:
    """Test metric name constants."""

    def test_mae_constant(self):
        assert ALL_METRICS == ("mae", "rmse", "bias", "std")


class TestColumnConstants:
    """Test column name constants."""

    def test_metric_name_col(self):
        assert METRIC_NAME_COL == "metric_name"

    def test_metric_value_col(self):
        assert METRIC_VALUE_COL == "metric_value"

    def test_sample_size_col(self):
        assert SAMPLE_SIZE_COL == "sample_size"

    def test_total_rows_col(self):
        assert TOTAL_ROWS_COL == "total_rows"

    def test_missing_count_col(self):
        assert MISSING_COUNT_COL == "missing_count"

    def test_variable_col(self):
        assert VARIABLE_COL == "variable"

    def test_dimension_cols(self):
        assert LEAD_HOURS_COL == "lead_hours"
        assert MODEL_COL == "model"
        assert MONTH_COL == "month"
        assert SEASON_COL == "season"


class TestSeasonMapping:
    """Test meteorological season mapping."""

    def test_djf(self):
        assert SEASON_MAP[12] == "DJF"
        assert SEASON_MAP[1] == "DJF"
        assert SEASON_MAP[2] == "DJF"

    def test_mam(self):
        assert SEASON_MAP[3] == "MAM"
        assert SEASON_MAP[4] == "MAM"
        assert SEASON_MAP[5] == "MAM"

    def test_jja(self):
        assert SEASON_MAP[6] == "JJA"
        assert SEASON_MAP[7] == "JJA"
        assert SEASON_MAP[8] == "JJA"

    def test_son(self):
        assert SEASON_MAP[9] == "SON"
        assert SEASON_MAP[10] == "SON"
        assert SEASON_MAP[11] == "SON"

    def test_all_months_mapped(self):
        assert set(SEASON_MAP.keys()) == set(range(1, 13))

    def test_season_order(self):
        assert SEASON_ORDER == ["DJF", "MAM", "JJA", "SON"]


class TestLongFormatColumns:
    """Test long_format_columns helper."""

    def test_no_dimensions(self):
        cols = long_format_columns()
        assert cols == LONG_FORMAT_FIXED_COLS

    def test_single_dimension(self):
        cols = long_format_columns(["lead_hours"])
        assert cols == ["lead_hours"] + LONG_FORMAT_FIXED_COLS

    def test_multiple_dimensions(self):
        cols = long_format_columns(["lead_hours", "model"])
        assert cols == ["lead_hours", "model"] + LONG_FORMAT_FIXED_COLS

    def test_all_dimensions(self):
        cols = long_format_columns(["lead_hours", "model", "month", "season"])
        assert cols == ["lead_hours", "model", "month", "season"] + LONG_FORMAT_FIXED_COLS
