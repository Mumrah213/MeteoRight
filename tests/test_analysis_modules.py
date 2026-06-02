"""Tests for analysis modules.

Verifies:
- Degradation rate computation
- Half-skill lead time
- Seasonal bias summary
- Anomaly detection
- CLI argument parsing
"""

import numpy as np
import pandas as pd
import pytest

from analysis.anomalies import (
    detect_high_error_groups,
    detect_outliers_iqr,
    detect_seasonal_bias_anomalies,
)
from analysis.degradation import (
    compute_degradation_rate,
    compute_degradation_summary,
    compute_half_skill_lead_time,
    compute_skill_score,
)
from analysis.seasonal_patterns import (
    seasonal_bias_summary,
    seasonal_skill_summary,
)


def _make_metrics_df(**kwargs):
    """Create a synthetic metrics DataFrame for testing."""
    from tests.test_grouping_semantics import _make_metrics_df as _base

    return _base(**kwargs)


class TestDegradationRate:
    """Test degradation rate computation."""

    def test_positive_degradation_rate(self):
        """MAE should increase with lead time (positive rate)."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 6, 12, 24, 48, 72],
                "variable": ["temperature_2m"] * 6,
                "metric_name": ["mae"] * 6,
                "metric_value": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                "sample_size": [100] * 6,
            }
        )

        rate = compute_degradation_rate(df, "temperature_2m", "mae")
        assert rate > 0  # Should be positive (degradation)

    def test_zero_degradation_rate(self):
        """Constant error should give zero rate."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 6, 12, 24],
                "model": ["model_a"] * 4,
                "variable": ["temperature_2m"] * 4,
                "metric_name": ["mae"] * 4,
                "metric_value": [1.0, 1.0, 1.0, 1.0],
                "sample_size": [100] * 4,
            }
        )

        rate = compute_degradation_rate(df, "temperature_2m", "mae")
        assert rate == pytest.approx(0.0, abs=1e-10)

    def test_insufficient_data(self):
        """Should return NaN for fewer than 3 data points."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 6],
                "variable": ["temperature_2m"] * 2,
                "metric_name": ["mae"] * 2,
                "metric_value": [1.0, 2.0],
                "sample_size": [100] * 2,
            }
        )

        rate = compute_degradation_rate(df, "temperature_2m", "mae")
        assert np.isnan(rate)

    def test_degradation_summary(self):
        """Should produce summary for multiple variables."""
        df = _make_metrics_df(
            variables=["temperature_2m", "precipitation"],
            metrics=["mae"],
        )

        summary = compute_degradation_summary(df, ["temperature_2m", "precipitation"])
        assert len(summary) > 0


class TestHalfSkillLeadTime:
    """Test half-skill lead time computation."""

    def test_half_skill_found(self):
        """Should find lead time where error doubles."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 6, 12, 24, 48],
                "variable": ["temperature_2m"] * 5,
                "metric_name": ["mae"] * 5,
                "metric_value": [1.0, 1.5, 2.0, 2.5, 3.0],  # min=1.0, 2x=2.0
                "sample_size": [100] * 5,
            }
        )

        result = compute_half_skill_lead_time(df, "temperature_2m", "mae")
        assert result is not None
        assert result >= 12  # At lead time 12, error = 2.0 = 2× min

    def test_half_skill_not_reached(self):
        """Should return None if error never doubles."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 6, 12],
                "variable": ["temperature_2m"] * 3,
                "metric_name": ["mae"] * 3,
                "metric_value": [1.0, 1.1, 1.2],  # Never doubles
                "sample_size": [100] * 3,
            }
        )

        result = compute_half_skill_lead_time(df, "temperature_2m", "mae")
        assert result is None


class TestSkillScore:
    """Test skill score computation."""

    def test_better_than_reference(self):
        """Model with lower error should have positive skill score."""
        df = pd.DataFrame(
            {
                "lead_hours": [24, 24],
                "variable": ["temperature_2m", "temperature_2m"],
                "model": ["model_a", "model_b"],
                "metric_name": ["mae", "mae"],
                "metric_value": [1.0, 2.0],  # model_a is better
                "sample_size": [100, 100],
            }
        )

        score = compute_skill_score(df, "temperature_2m", "mae", "model_a", "model_b")
        assert score > 0  # model_a is better than model_b

    def test_worse_than_reference(self):
        """Model with higher error should have negative skill score."""
        df = pd.DataFrame(
            {
                "lead_hours": [24, 24],
                "variable": ["temperature_2m", "temperature_2m"],
                "model": ["model_a", "model_b"],
                "metric_name": ["mae", "mae"],
                "metric_value": [2.0, 1.0],  # model_a is worse
                "sample_size": [100, 100],
            }
        )

        score = compute_skill_score(df, "temperature_2m", "mae", "model_a", "model_b")
        assert score < 0


class TestSeasonalBias:
    """Test seasonal bias summary."""

    def test_seasonal_bias_summary(self):
        """Should compute bias by season."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 6, 12],
                "season": ["DJF", "MAM", "JJA"],
                "variable": ["temperature_2m"] * 3,
                "metric_name": ["bias"] * 3,
                "metric_value": [0.5, -0.3, 0.1],
                "sample_size": [100] * 3,
            }
        )

        result = seasonal_bias_summary(df, "temperature_2m")
        assert len(result) > 0
        assert "bias" in result.columns

    def test_seasonal_skill_summary(self):
        """Should compute skill by season and metric."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 6],
                "season": ["DJF", "MAM"],
                "variable": ["temperature_2m"] * 2,
                "metric_name": ["mae", "rmse"],
                "metric_value": [1.0, 2.0],
                "sample_size": [100] * 2,
            }
        )

        result = seasonal_skill_summary(df, "temperature_2m", metrics=["mae", "rmse"])
        assert len(result) > 0


class TestAnomalyDetection:
    """Test anomaly detection functions."""

    def test_detect_high_error_groups(self):
        """Should detect groups with high z-scores."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 6, 12, 24, 48, 72, 100, 200],
                "variable": ["temperature_2m"] * 8,
                "metric_name": ["mae"] * 8,
                "metric_value": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 10.0],
                "sample_size": [100] * 8,
            }
        )

        anomalies = detect_high_error_groups(df, "temperature_2m", n_sigma=1.5)
        assert len(anomalies) >= 1  # The 10.0 point should be flagged

    def test_detect_seasonal_bias_anomalies(self):
        """Should detect seasons with anomalous bias."""
        df = pd.DataFrame(
            {
                "season": ["DJF", "MAM", "JJA", "SON"],
                "variable": ["temperature_2m"] * 4,
                "metric_name": ["bias"] * 4,
                "metric_value": [0.1, 0.2, 5.0, 0.1],  # JJA is anomalous
                "sample_size": [100] * 4,
            }
        )

        anomalies = detect_seasonal_bias_anomalies(df, "temperature_2m", n_sigma=1.0)
        assert len(anomalies) >= 1

    def test_detect_outliers_iqr(self):
        """Should detect outliers using IQR method."""
        values = list(range(1, 21)) + [100.0]
        df = pd.DataFrame(
            {
                "lead_hours": list(range(1, 22)),
                "variable": ["temperature_2m"] * 21,
                "metric_name": ["mae"] * 21,
                "metric_value": values,
                "sample_size": [100] * 21,
            }
        )

        outliers = detect_outliers_iqr(df, "temperature_2m", iqr_multiplier=1.5)
        assert len(outliers) >= 1  # 100.0 should be an outlier
