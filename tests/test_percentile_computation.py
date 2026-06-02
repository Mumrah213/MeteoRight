"""Tests for percentile computation.

Verifies:
- Percentile calculations are correct
- Edge cases (empty, single value) are handled
- Quantile summaries are consistent
"""

import numpy as np
import pandas as pd
import pytest

from analysis.extremes import (
    compute_error_quantiles,
    compute_lead_time_quantiles,
)


class TestPercentileComputation:
    """Test percentile computation correctness."""

    def test_median_of_known_data(self):
        """Median of 1..10 should be 5.5."""
        errors = pd.Series(np.arange(1, 11, dtype=float))
        result = compute_error_quantiles(errors, "temperature_2m", percentiles=[50])
        assert len(result) == 1
        assert result["value"].iloc[0] == pytest.approx(5.5, abs=0.1)

    def test_quartiles_of_known_data(self):
        """Quartiles of 1..100 should be approximately 25, 50, 75."""
        errors = pd.Series(np.arange(1, 101, dtype=float))
        result = compute_error_quantiles(errors, "temperature_2m", percentiles=[25, 50, 75])

        values = dict(zip(result["percentile"], result["value"], strict=False))
        assert values[25] == pytest.approx(25.5, abs=1.0)
        assert values[50] == pytest.approx(50.5, abs=1.0)
        assert values[75] == pytest.approx(75.5, abs=1.0)

    def test_percentile_ordering(self):
        """Higher percentiles should have higher values."""
        errors = pd.Series(np.random.normal(0, 5, 1000))
        result = compute_error_quantiles(errors, "temperature_2m", percentiles=[10, 25, 50, 75, 90])

        values = result["value"].values
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_empty_series(self):
        """Empty series should return empty DataFrame."""
        errors = pd.Series([], dtype=float)
        result = compute_error_quantiles(errors, "temperature_2m")
        assert len(result) == 0

    def test_single_value(self):
        """Single value should return that value for all percentiles."""
        errors = pd.Series([5.0])
        result = compute_error_quantiles(errors, "temperature_2m", percentiles=[10, 50, 90])
        assert all(result["value"] == 5.0)

    def test_symmetric_distribution(self):
        """Median of symmetric distribution should be near zero."""
        errors = pd.Series(np.random.normal(0, 2, 10000))
        result = compute_error_quantiles(errors, "temperature_2m", percentiles=[50])
        assert abs(result["value"].iloc[0]) < 0.5

    def test_skewed_distribution(self):
        """Right-skewed distribution should have median < mean."""
        errors = pd.Series(np.random.exponential(2, 10000))
        result = compute_error_quantiles(errors, "temperature_2m", percentiles=[50])
        assert result["value"].iloc[0] < errors.mean()


class TestLeadTimeQuantiles:
    """Test lead-time grouped quantile computation."""

    def test_lead_time_quantiles_structure(self):
        """Should produce one row per (lead_time, percentile) combination."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": np.random.normal(0, 2, 500),
                "lead_hours": np.random.choice([1, 6, 12, 24, 48], 500),
            }
        )

        result = compute_lead_time_quantiles(df, "temperature_2m", percentiles=[10, 50, 90])

        expected_rows = 5 * 3  # 5 lead times × 3 percentiles
        assert len(result) == expected_rows
        assert set(result["lead_hours"]) == {1, 6, 12, 24, 48}
        assert set(result["percentile"]) == {10, 50, 90}

    def test_lead_time_quantiles_increasing_variance(self):
        """Variance should generally increase with lead time."""
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                "temperature_2m_error": np.random.normal(0, 1, n),
                "lead_hours": np.random.choice([1, 6, 12, 24, 48], n),
            }
        )
        # Add increasing variance with lead time
        for lt in df["lead_hours"].unique():
            mask = df["lead_hours"] == lt
            df.loc[mask, "temperature_2m_error"] *= 1 + lt * 0.05

        result = compute_lead_time_quantiles(df, "temperature_2m", percentiles=[10, 50, 90])

        # Check that P90 - P10 (spread) generally increases
        spreads = result.groupby("lead_hours")["value"].agg(lambda x: x.max() - x.min())
        assert spreads.iloc[-1] >= spreads.iloc[0]  # Last LT should have wider spread

    def test_empty_lead_time_quantiles(self):
        """Empty DataFrame should return empty result."""
        df = pd.DataFrame({"temperature_2m_error": [], "lead_hours": []})
        result = compute_lead_time_quantiles(df, "temperature_2m")
        assert len(result) == 0
