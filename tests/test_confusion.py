"""Tests for confusion matrix computation."""


import numpy as np
import pandas as pd
import pytest

from verification.confusion import (
    compute_confusion_counts,
    compute_confusion_matrix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_events_df(n: int = 10, include_nan: bool = False) -> pd.DataFrame:
    """Create a minimal DataFrame with event columns for testing."""
    np.random.seed(42)
    fcst = np.random.choice([True, False], size=n)
    obs = np.random.choice([True, False], size=n)

    df = pd.DataFrame(
        {
            "forecast_temperature_2m": np.random.uniform(-5, 20, size=n),
            "observed_temperature_2m": np.random.uniform(-5, 20, size=n),
            "event_forecast_frost": fcst,
            "event_observed_frost": obs,
        }
    )

    if include_nan:
        df.loc[2, "forecast_temperature_2m"] = np.nan
        df.loc[5, "observed_temperature_2m"] = np.nan

    return df


# ---------------------------------------------------------------------------
# compute_confusion_counts
# ---------------------------------------------------------------------------


class TestComputeConfusionCounts:
    """Test confusion counts from binary series."""

    def test_perfect_forecast(self):
        fc = pd.Series([True, True, False, False])
        obs = pd.Series([True, True, False, False])
        result = compute_confusion_counts(fc, obs)
        assert result["hits"] == 2
        assert result["misses"] == 0
        assert result["false_alarms"] == 0
        assert result["correct_negatives"] == 2
        assert result["excluded_nan"] == 0

    def test_random_forecast(self):
        fc = pd.Series([True, True, False, False])
        obs = pd.Series([True, False, True, False])
        result = compute_confusion_counts(fc, obs)
        assert result["hits"] == 1
        assert result["misses"] == 1
        assert result["false_alarms"] == 1
        assert result["correct_negatives"] == 1
        assert result["excluded_nan"] == 0

    def test_all_hits(self):
        fc = pd.Series([True, True])
        obs = pd.Series([True, True])
        result = compute_confusion_counts(fc, obs)
        assert result["hits"] == 2

    def test_all_misses(self):
        fc = pd.Series([False, False])
        obs = pd.Series([True, True])
        result = compute_confusion_counts(fc, obs)
        assert result["misses"] == 2

    def test_all_false_alarms(self):
        fc = pd.Series([True, True])
        obs = pd.Series([False, False])
        result = compute_confusion_counts(fc, obs)
        assert result["false_alarms"] == 2

    def test_all_correct_negatives(self):
        fc = pd.Series([False, False])
        obs = pd.Series([False, False])
        result = compute_confusion_counts(fc, obs)
        assert result["correct_negatives"] == 2

    def test_nan_exclusion(self):
        """Rows where both event flags are determined by non-NaN values."""
        fc = pd.Series([True, False, True])
        obs = pd.Series([True, False, False])
        result = compute_confusion_counts(fc, obs)
        # All three are valid (no NaN in the Series)
        assert result["excluded_nan"] == 0


# ---------------------------------------------------------------------------
# compute_confusion_matrix
# ---------------------------------------------------------------------------


class TestComputeConfusionMatrix:
    """Test confusion matrix from DataFrame with event columns."""

    def test_single_group(self):
        df = _make_events_df(n=10)
        result = compute_confusion_matrix(df, event_name="frost")
        assert len(result) == 1
        assert result["event_name"].iloc[0] == "frost"
        assert result["variable"].iloc[0] == "temperature_2m"
        assert result["threshold"].iloc[0] == 0
        assert result["operator"].iloc[0] == "le"
        assert result["sample_size"].iloc[0] > 0

    def test_grouped_by_lead_hours(self):
        df = pd.DataFrame(
            {
                "lead_hours": [0, 0, 12, 12, 24, 24],
                "model": ["icon_eu"] * 6,
                "forecast_temperature_2m": [0.5, 0.5, -1.0, -1.0, -2.0, 5.0],
                "observed_temperature_2m": [1.0, -0.5, 2.0, -2.0, -3.0, 10.0],
                "event_forecast_frost": [False, True, False, True, True, False],
                "event_observed_frost": [False, True, True, True, True, False],
            }
        )
        result = compute_confusion_matrix(df, event_name="frost", groupby=["lead_hours"])
        assert len(result) == 3  # 0, 12, 24 hours

        # Lead hour 0: forecast=[F,T], observed=[F,T] → 1 hit, 0 miss, 0 FA, 1 CN
        r0 = result[result["lead_hours"] == 0].iloc[0]
        assert r0["hits"] == 1
        assert r0["misses"] == 0
        assert r0["false_alarms"] == 0
        assert r0["correct_negatives"] == 1

        # Lead hour 12: forecast=[F,T], observed=[T,T] → 1 hit, 1 miss, 0 FA, 0 CN
        r12 = result[result["lead_hours"] == 12].iloc[0]
        assert r12["hits"] == 1
        assert r12["misses"] == 1

    def test_nan_exclusion(self):
        df = pd.DataFrame(
            {
                "lead_hours": [0, 0, 0],
                "model": ["icon_eu"] * 3,
                "forecast_temperature_2m": [-1.0, -1.0, np.nan],
                "observed_temperature_2m": [-1.0, 1.0, 0.0],
                "event_forecast_frost": [True, False, False],
                "event_observed_frost": [True, False, False],
            }
        )
        result = compute_confusion_matrix(df, event_name="frost", groupby=["lead_hours"])
        # Only rows 0 and 1 should be counted (row 2 has NaN forecast)
        assert result["excluded_nan"].iloc[0] == 1
        assert result["sample_size"].iloc[0] == 2

    def test_grouped_by_model(self):
        df = pd.DataFrame(
            {
                "lead_hours": [0] * 6,
                "model": ["icon_eu", "icon_eu", "icon_eu", "gfs", "gfs", "gfs"],
                "forecast_temperature_2m": [-1.0, -1.5, 1.0, 0.5, -0.5, 3.0],
                "observed_temperature_2m": [-0.5, -2.0, 0.0, 1.0, -1.0, 4.0],
                "event_forecast_frost": [True, True, False, False, True, False],
                "event_observed_frost": [True, True, False, False, False, False],
            }
        )
        result = compute_confusion_matrix(df, event_name="frost", groupby=["model"])
        assert len(result) == 2

        # icon_eu: forecast=[T,T,F], observed=[T,T,F] → 2 hits, 0 miss, 0 FA, 1 CN
        icon = result[result["model"] == "icon_eu"].iloc[0]
        assert icon["hits"] == 2
        assert icon["misses"] == 0

        # gfs: forecast=[F,T,F], observed=[F,F,F] → 0 hits, 0 miss, 1 FA, 2 CN
        gfs = result[result["model"] == "gfs"].iloc[0]
        assert gfs["false_alarms"] == 1
        assert gfs["correct_negatives"] == 2

    def test_invalid_groupby(self):
        df = _make_events_df()
        with pytest.raises(ValueError, match="Invalid groupby dimension"):
            compute_confusion_matrix(df, event_name="frost", groupby=["invalid"])

    def test_missing_event_columns(self):
        df = pd.DataFrame(
            {
                "forecast_temperature_2m": [-1.0],
                "observed_temperature_2m": [-1.0],
            }
        )
        with pytest.raises(ValueError, match="Missing columns"):
            compute_confusion_matrix(df, event_name="frost")

    def test_empty_dataframe(self):
        df = pd.DataFrame(
            {
                "lead_hours": [],
                "model": [],
                "forecast_temperature_2m": pd.Series(dtype=float),
                "observed_temperature_2m": pd.Series(dtype=float),
                "event_forecast_frost": pd.Series(dtype=bool),
                "event_observed_frost": pd.Series(dtype=bool),
            }
        )
        result = compute_confusion_matrix(df, event_name="frost")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestConfusionIntegration:
    """Integration tests with real data patterns."""

    def test_frost_scenario(self):
        """Test a realistic frost scenario."""
        df = pd.DataFrame(
            {
                "lead_hours": [0, 6, 12, 18, 24, 36, 48],
                "model": ["icon_eu"] * 7,
                "forecast_temperature_2m": [-2, -3, -1, 1, 3, -1, -2],
                "observed_temperature_2m": [-1, -4, 0, 2, 5, -2, -3],
            }
        )
        from verification.events import generate_events

        df = generate_events(df, event_name="frost")

        result = compute_confusion_matrix(df, event_name="frost", groupby=None)
        # Check that totals add up
        total = result.iloc[0]
        assert (
            total["hits"]
            + total["misses"]
            + total["false_alarms"]
            + total["correct_negatives"]
            + total["excluded_nan"]
            == total["total_rows"]
        )

    def test_precip_scenario(self):
        """Test a precipitation scenario."""
        df = pd.DataFrame(
            {
                "lead_hours": [0, 6, 12],
                "model": ["icon_eu"] * 3,
                "forecast_precipitation": [0.1, 3.0, 0.0],
                "observed_precipitation": [0.0, 4.0, 0.0],
            }
        )
        from verification.events import generate_events

        df = generate_events(df, event_name="heavy_precip")

        result = compute_confusion_matrix(df, event_name="heavy_precip", groupby=None)
        total = result.iloc[0]
        assert (
            total["hits"]
            + total["misses"]
            + total["false_alarms"]
            + total["correct_negatives"]
            + total["excluded_nan"]
            == total["total_rows"]
        )
