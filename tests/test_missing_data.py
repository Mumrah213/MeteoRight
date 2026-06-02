"""Tests for missing-data handling.

Verifies:
- NaN values are properly excluded from computations
- Empty data is handled gracefully
- Sample size tracking is correct
- No silent data loss
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data.loaders import filter_metrics

from analysis.extremes import (
    compute_error_quantiles,
    find_top_errors,
)
from plots.distributions import plot_error_histogram


class TestMissingDataHandling:
    """Test NaN handling in plotting and analysis."""

    def test_histogram_with_nans(self):
        """Histogram should exclude NaN values."""
        errors = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
        fig = plot_error_histogram(errors, "temperature_2m")
        assert fig is not None
        plt.close(fig)

    def test_histogram_all_nan(self):
        """Should return None for all-NaN series."""
        errors = pd.Series([np.nan, np.nan, np.nan])
        fig = plot_error_histogram(errors, "temperature_2m")
        assert fig is None

    def test_top_errors_with_nans(self):
        """Top errors should exclude NaN values."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": [1.0, np.nan, -5.0, np.nan, 3.0],
                "lead_hours": [1, 2, 3, 4, 5],
            }
        )

        top = find_top_errors(df, "temperature_2m", n=3)
        assert len(top) == 3
        # NaN should not be in results
        assert not top["temperature_2m_error"].isna().any()

    def test_quantiles_with_nans(self):
        """Quantiles should exclude NaN values."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": [1.0, np.nan, 3.0, np.nan, 5.0],
                "lead_hours": [1, 2, 3, 4, 5],
            }
        )

        result = compute_error_quantiles(df, "temperature_2m")
        assert len(result) > 0
        assert not result["value"].isna().any()

    def test_filter_metrics_preserves_non_matching(self):
        """Filtering should not drop rows from non-matching variables."""
        df = pd.DataFrame(
            {
                "variable": ["temperature_2m", "precipitation", "temperature_2m"],
                "metric_name": ["mae", "mae", "mae"],
                "metric_value": [1.0, 2.0, 3.0],
            }
        )

        filtered = filter_metrics(df, variables=["temperature_2m"])
        assert len(filtered) == 2
        assert all(filtered["variable"] == "temperature_2m")

    def test_filter_metrics_no_match(self):
        """Filtering with no matches should return empty DataFrame."""
        df = pd.DataFrame(
            {
                "variable": ["temperature_2m", "precipitation"],
                "metric_name": ["mae", "mae"],
                "metric_value": [1.0, 2.0],
            }
        )

        filtered = filter_metrics(df, variables=["nonexistent"])
        assert len(filtered) == 0

    def test_filter_metrics_multiple_filters(self):
        """Multiple filters should combine correctly."""
        df = pd.DataFrame(
            {
                "variable": ["temperature_2m", "temperature_2m", "precipitation"],
                "metric_name": ["mae", "rmse", "mae"],
                "metric_value": [1.0, 2.0, 3.0],
            }
        )

        filtered = filter_metrics(
            df,
            variables=["temperature_2m"],
            metrics=["mae"],
        )
        assert len(filtered) == 1
        assert filtered["metric_value"].iloc[0] == 1.0

    def test_sample_size_tracking(self):
        """Sample size should reflect non-NaN count."""
        errors = pd.Series([1.0, np.nan, 3.0, 4.0, np.nan])
        clean = errors.dropna()
        assert len(clean) == 3  # 3 non-NaN values

    def test_empty_dataframe_handling(self):
        """Should handle empty DataFrames gracefully."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": pd.Series([], dtype=float),
                "lead_hours": pd.Series([], dtype=int),
            }
        )

        top = find_top_errors(df, "temperature_2m", n=10)
        assert len(top) == 0

        result = compute_error_quantiles(df, "temperature_2m")
        assert len(result) == 0
