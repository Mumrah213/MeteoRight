"""Tests for extreme-event filtering.

Verifies:
- Top-N error selection is correct
- Extreme miss filtering works
- Direction filtering (over/under) is correct
- Threshold-based filtering is correct
"""

import numpy as np
import pandas as pd
import pytest

from analysis.extremes import (
    compute_extreme_event_frequency,
    find_extreme_misses,
    find_top_errors,
)


def _make_verification_df(n=100):
    """Create a synthetic verification DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "temperature_2m_error": np.random.normal(0, 2, n),
            "lead_hours": np.random.choice([1, 6, 12, 24, 48], n),
            "model": np.random.choice(["icon_eu", "gfs"], n),
        }
    )


class TestTopErrors:
    """Test top-N error selection."""

    def test_find_top_errors(self):
        """Should return N largest absolute errors, sorted descending."""
        errors = np.array([1.0, -5.0, 3.0, -10.0, 2.0])
        df = pd.DataFrame(
            {
                "temperature_2m_error": errors,
                "lead_hours": [1, 6, 12, 24, 48],
            }
        )

        top = find_top_errors(df, "temperature_2m", n=3)
        assert len(top) == 3
        assert top["abs_error"].iloc[0] >= top["abs_error"].iloc[1] >= top["abs_error"].iloc[2]

    def test_find_top_errors_n_larger_than_data(self):
        """Should return all data if N > data size."""
        df = _make_verification_df(n=10)
        top = find_top_errors(df, "temperature_2m", n=100)
        assert len(top) == 10

    def test_find_top_errors_with_model_filter(self):
        """Should respect model filter."""
        df = _make_verification_df(n=100)
        top = find_top_errors(df, "temperature_2m", n=10, model="icon_eu")
        assert all(top["model"] == "icon_eu")

    def test_find_top_errors_with_lead_time_filter(self):
        """Should respect lead time filter."""
        df = _make_verification_df(n=100)
        top = find_top_errors(df, "temperature_2m", n=10, lead_hours=24)
        assert all(top["lead_hours"] == 24)

    def test_find_top_errors_abs_values(self):
        """Top errors should be sorted by absolute value."""
        errors = np.array([-100.0, 50.0, -75.0, 10.0, 25.0])
        df = pd.DataFrame(
            {
                "temperature_2m_error": errors,
                "lead_hours": [1, 2, 3, 4, 5],
            }
        )

        top = find_top_errors(df, "temperature_2m", n=3)
        abs_vals = top["abs_error"].values
        assert abs_vals[0] == 100.0
        assert abs_vals[1] == 75.0
        assert abs_vals[2] == 50.0

    def test_find_top_errors_no_error_column(self):
        """Should return empty DataFrame for missing error column."""
        df = pd.DataFrame({"other_error": [1.0, 2.0], "lead_hours": [1, 2]})
        top = find_top_errors(df, "temperature_2m", n=5)
        assert len(top) == 0


class TestExtremeMisses:
    """Test extreme miss filtering."""

    def test_find_extreme_misses_both_directions(self):
        """Should find misses in both directions."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": [5.0, -5.0, 3.0, -3.0, 1.0],
                "lead_hours": [1, 2, 3, 4, 5],
            }
        )

        misses = find_extreme_misses(df, "temperature_2m", threshold=3.0)
        assert len(misses) == 4  # All except 1.0

    def test_find_extreme_misses_overprediction(self):
        """Should find only overpredictions when direction='over'."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": [5.0, -5.0, 3.0, -3.0, 1.0],
                "lead_hours": [1, 2, 3, 4, 5],
            }
        )

        misses = find_extreme_misses(df, "temperature_2m", threshold=3.0, direction="over")
        assert all(misses["temperature_2m_error"] > 0)
        assert len(misses) == 2  # 5.0 and 3.0

    def test_find_extreme_misses_underprediction(self):
        """Should find only underpredictions when direction='under'."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": [5.0, -5.0, 3.0, -3.0, 1.0],
                "lead_hours": [1, 2, 3, 4, 5],
            }
        )

        misses = find_extreme_misses(df, "temperature_2m", threshold=3.0, direction="under")
        assert all(misses["temperature_2m_error"] < 0)
        assert len(misses) == 2  # -5.0 and -3.0

    def test_find_extreme_misses_threshold(self):
        """Should respect threshold."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": [1.0, 2.0, 3.0, 4.0, 5.0],
                "lead_hours": [1, 2, 3, 4, 5],
            }
        )

        misses = find_extreme_misses(df, "temperature_2m", threshold=4.0)
        assert len(misses) == 2  # 4.0 and 5.0


class TestExtremeEventFrequency:
    """Test extreme event frequency computation."""

    def test_extreme_event_frequency_basic(self):
        """Should compute correct frequency."""
        errors = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        df = pd.DataFrame(
            {
                "temperature_2m_error": errors,
                "lead_hours": [1, 2, 3, 4, 5],
            }
        )

        result = compute_extreme_event_frequency(df, "temperature_2m", threshold=50.0)
        assert result["total"] == 5
        assert result["extreme_count"] == 1
        assert result["frequency"] == 0.2

    def test_extreme_event_frequency_no_extremes(self):
        """Should return zero frequency when no extremes."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": [1.0, 2.0, 3.0],
                "lead_hours": [1, 2, 3],
            }
        )

        result = compute_extreme_event_frequency(df, "temperature_2m", threshold=100.0)
        assert result["extreme_count"] == 0
        assert result["frequency"] == 0.0

    def test_extreme_event_frequency_all_extremes(self):
        """Should return frequency of 1.0 when all are extremes."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": [100.0, 200.0, 300.0],
                "lead_hours": [1, 2, 3],
            }
        )

        result = compute_extreme_event_frequency(df, "temperature_2m", threshold=10.0)
        assert result["frequency"] == 1.0

    def test_extreme_event_frequency_mean_error(self):
        """Should compute correct mean error during extremes."""
        df = pd.DataFrame(
            {
                "temperature_2m_error": [1.0, 10.0, 20.0, 30.0],
                "lead_hours": [1, 2, 3, 4],
            }
        )

        result = compute_extreme_event_frequency(df, "temperature_2m", threshold=15.0)
        assert result["mean_error_when_extreme"] == pytest.approx(25.0, abs=0.1)

    def test_extreme_event_frequency_empty(self):
        """Should handle empty DataFrame."""
        df = pd.DataFrame({"temperature_2m_error": [], "lead_hours": []})
        result = compute_extreme_event_frequency(df, "temperature_2m", threshold=1.0)
        assert result["total"] == 0

    def test_extreme_event_frequency_missing_column(self):
        """Should handle missing error column."""
        df = pd.DataFrame({"other_error": [1.0, 2.0], "lead_hours": [1, 2]})
        result = compute_extreme_event_frequency(df, "temperature_2m", threshold=1.0)
        assert result["total"] == 0
