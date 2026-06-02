"""Tests for missing data handling.

Verifies that NaN errors are properly excluded from metric computation
and that sample_size, total_rows, and missing_count are correctly tracked.
"""

import math

import numpy as np
import pandas as pd
import pytest

from metrics.aggregations import aggregate_metrics
from metrics.metrics import (
    compute_bias,
    compute_mae,
    compute_rmse,
    compute_std,
)
from metrics.schemas import MISSING_COUNT_COL, SAMPLE_SIZE_COL, TOTAL_ROWS_COL


def _make_df_with_nan_errors(
    n_total: int = 100,
    n_nan: int = 20,
) -> pd.DataFrame:
    """Create a verification DataFrame with some NaN errors.

    Args:
        n_total: Total number of rows.
        n_nan: Number of rows with NaN temperature_2m_error.

    Returns:
        DataFrame with provenance + variable columns.
    """
    np.random.seed(42)
    indices = np.random.choice(n_total, n_nan, replace=False)

    df = pd.DataFrame(
        {
            "forecast_issue_time": pd.date_range("2025-01-01", periods=n_total, freq="6h"),
            "forecast_target_time": pd.date_range("2025-01-02", periods=n_total, freq="6h"),
            "observation_time": pd.date_range("2025-01-02", periods=n_total, freq="6h"),
            "lead_hours": 24,
            "model": "model_a",
            "latitude": 55.0,
            "longitude": 13.0,
            "elevation": 100.0,
            "forecast_temperature_2m": 20.0 + np.random.randn(n_total),
            "observed_temperature_2m": 19.0 + np.random.randn(n_total),
            "temperature_2m_error": 1.0 + np.random.randn(n_total),
            "forecast_precipitation": np.abs(np.random.randn(n_total)),
            "observed_precipitation": np.abs(np.random.randn(n_total)),
            "precipitation_error": np.random.randn(n_total),
        }
    )

    # Set some errors to NaN
    df.loc[indices, "temperature_2m_error"] = np.nan

    return df


class TestMetricFunctionsWithNaN:
    """Test that metric functions handle NaN correctly."""

    def test_mae_excludes_nan(self):
        """MAE should only use non-NaN values."""
        errors = pd.Series([1.0, float("nan"), 3.0, float("nan"), 5.0])
        # Only [1, 3, 5] used: MAE = (1+3+5)/3 = 3.0
        assert compute_mae(errors) == pytest.approx(3.0)

    def test_rmse_excludes_nan(self):
        """RMSE should only use non-NaN values."""
        errors = pd.Series([1.0, float("nan"), 3.0])
        # Only [1, 3] used: RMSE = sqrt((1+9)/2) = sqrt(5)
        assert compute_rmse(errors) == pytest.approx(math.sqrt(5))

    def test_bias_excludes_nan(self):
        """Bias should only use non-NaN values."""
        errors = pd.Series([1.0, float("nan"), 3.0])
        # Only [1, 3] used: Bias = 2.0
        assert compute_bias(errors) == pytest.approx(2.0)

    def test_std_excludes_nan(self):
        """Std should only use non-NaN values."""
        errors = pd.Series([1.0, float("nan"), 3.0, float("nan"), 5.0])
        # Only [1, 3, 5] used
        expected = pd.Series([1.0, 3.0, 5.0]).std(ddof=1)
        assert compute_std(errors) == pytest.approx(expected)

    def test_all_nan_returns_nan(self):
        """All-NaN series returns NaN for all metrics."""
        errors = pd.Series([float("nan"), float("nan"), float("nan")])
        assert math.isnan(compute_mae(errors))
        assert math.isnan(compute_rmse(errors))
        assert math.isnan(compute_bias(errors))
        assert math.isnan(compute_std(errors))


class TestAggregationWithNaN:
    """Test that aggregation correctly tracks missing data."""

    def test_sample_size_tracking(self):
        """sample_size should equal count of non-NaN errors."""
        df = _make_df_with_nan_errors(n_total=100, n_nan=20)

        result = aggregate_metrics(
            df,
            groupby=["lead_hours"],
            metrics=["mae"],
            variables=["temperature_2m"],
        )

        # 100 total rows, 20 NaN → sample_size = 80
        assert result[SAMPLE_SIZE_COL].values[0] == 80
        assert result[TOTAL_ROWS_COL].values[0] == 100
        assert result[MISSING_COUNT_COL].values[0] == 20

    def test_missing_count_accuracy(self):
        """missing_count should equal total_rows - sample_size."""
        df = _make_df_with_nan_errors(n_total=200, n_nan=50)

        result = aggregate_metrics(
            df,
            groupby=["lead_hours"],
            metrics=["mae"],
            variables=["temperature_2m"],
        )

        assert result[MISSING_COUNT_COL].values[0] == 50
        assert result[TOTAL_ROWS_COL].values[0] == 200
        assert result[SAMPLE_SIZE_COL].values[0] == 150

    def test_mae_with_nan_differs_from_without(self):
        """MAE computed with NaN excluded should differ from including zeros."""
        df = _make_df_with_nan_errors(n_total=100, n_nan=20)

        # The metric function should give the same result whether we
        # pre-filter NaN or let it handle it
        errors = df["temperature_2m_error"]
        mae_with_nan_handling = compute_mae(errors)
        mae_pre_filtered = compute_mae(errors.dropna())
        assert mae_with_nan_handling == pytest.approx(mae_pre_filtered)

    def test_partial_nan_in_groups(self):
        """Test aggregation when different groups have different NaN rates."""
        np.random.seed(99)
        rows = []
        for lh in [1, 24]:
            for i in range(50):
                error = 1.0 + np.random.randn()
                # Group 1 has 50% NaN, group 2 has 0% NaN
                if lh == 1 and i < 25:
                    error = np.nan
                rows.append(
                    {
                        "forecast_issue_time": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i),
                        "forecast_target_time": pd.Timestamp("2025-01-01")
                        + pd.Timedelta(hours=i + lh),
                        "observation_time": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i + lh),
                        "lead_hours": lh,
                        "model": "model_a",
                        "latitude": 55.0,
                        "longitude": 13.0,
                        "elevation": 100.0,
                        "forecast_temperature_2m": 20.0,
                        "observed_temperature_2m": 19.0,
                        "temperature_2m_error": error,
                        "forecast_precipitation": 0.0,
                        "observed_precipitation": 0.0,
                        "precipitation_error": 0.0,
                    }
                )

        df = pd.DataFrame(rows)

        result = aggregate_metrics(
            df,
            groupby=["lead_hours"],
            metrics=["mae"],
            variables=["temperature_2m"],
        )

        # Group 1: 25 NaN out of 50 → sample_size=25, missing=25
        # Group 2: 0 NaN out of 50 → sample_size=50, missing=0
        for _, row in result.iterrows():
            if row["lead_hours"] == 1:
                assert row["sample_size"] == 25
                assert row["total_rows"] == 50
                assert row["missing_count"] == 25
            else:
                assert row["sample_size"] == 50
                assert row["total_rows"] == 50
                assert row["missing_count"] == 0
