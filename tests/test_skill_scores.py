"""Tests for skill score computation."""


import numpy as np
import pandas as pd
import pytest

from verification.skill_scores import (
    SKILL_SCORE_FUNCTIONS,
    _compute_accuracy,
    _compute_csi,
    _compute_false_alarm_rate,
    _compute_hit_rate,
    _compute_miss_rate,
    _compute_odds_ratio,
    _compute_precision,
    compute_skill_scores,
)

# ---------------------------------------------------------------------------
# Individual skill score functions
# ---------------------------------------------------------------------------


class TestIndividualSkillScores:
    """Test individual skill score computations."""

    def test_hit_rate_perfect(self):
        hr, _ = _compute_hit_rate(10, 0)
        assert hr == pytest.approx(1.0)

    def test_hit_rate_all_missed(self):
        hr, _ = _compute_hit_rate(0, 10)
        assert hr == pytest.approx(0.0)

    def test_hit_rate_half(self):
        hr, _ = _compute_hit_rate(5, 5, 0, 0)
        assert hr == pytest.approx(0.5)

    def test_hit_rate_zero_denominator(self):
        hr, _ = _compute_hit_rate(0, 0)
        assert np.isnan(hr)

    def test_false_alarm_rate_perfect(self):
        far, _ = _compute_false_alarm_rate(0, 0, 0, 100)
        assert far == pytest.approx(0.0)

    def test_false_alarm_rate_all_wrong(self):
        far, _ = _compute_false_alarm_rate(0, 0, 100, 0)
        assert far == pytest.approx(1.0)

    def test_false_alarm_rate_zero_denominator(self):
        far, _ = _compute_false_alarm_rate(0, 0, 0, 0)
        assert np.isnan(far)

    def test_precision_perfect(self):
        prec, _ = _compute_precision(10, 0, 0, 0)
        assert prec == pytest.approx(1.0)

    def test_precision_all_false_alarms(self):
        prec, _ = _compute_precision(0, 0, 10, 0)
        assert prec == pytest.approx(0.0)

    def test_csi_perfect(self):
        csi, _ = _compute_csi(10, 0, 0)
        assert csi == pytest.approx(1.0)

    def test_csi_no_hits(self):
        csi, _ = _compute_csi(0, 10, 5)
        assert csi == pytest.approx(0.0)

    def test_csi_partial(self):
        csi, _ = _compute_csi(5, 5, 2, 0)
        assert csi == pytest.approx(5 / 12)

    def test_csi_all_nan(self):
        csi, _ = _compute_csi(0, 0, 0)
        assert np.isnan(csi)

    def test_miss_rate_complement_of_hit_rate(self):
        mr, _ = _compute_miss_rate(5, 5, 0, 0)
        hr, _ = _compute_hit_rate(5, 5, 0, 0)
        assert mr == pytest.approx(1.0 - hr)

    def test_accuracy_perfect(self):
        acc, _ = _compute_accuracy(50, 0, 0, 50)
        assert acc == pytest.approx(1.0)

    def test_accuracy_random(self):
        acc, _ = _compute_accuracy(25, 25, 25, 25)
        assert acc == pytest.approx(0.5)

    def test_accuracy_zero_denominator(self):
        acc, _ = _compute_accuracy(0, 0, 0, 0)
        assert np.isnan(acc)

    def test_odds_ratio_perfect(self):
        # Hits only, no misses/false_alarms
        # odds_ratio = (hits * cn) / (misses * fa) → division by zero
        or_val, _ = _compute_odds_ratio(10, 0, 0, 10)
        assert np.isnan(or_val)

    def test_odds_ratio_defined(self):
        # hits=3, misses=2, false_alarms=1, cn=4
        or_val, _ = _compute_odds_ratio(3, 2, 1, 4)
        expected = (3 * 4) / (2 * 1)
        assert or_val == pytest.approx(expected)

    def test_odds_ratio_zero_denom(self):
        or_val, _ = _compute_odds_ratio(10, 0, 0, 10)
        assert np.isnan(or_val)


# ---------------------------------------------------------------------------
# compute_skill_scores
# ---------------------------------------------------------------------------


class TestComputeSkillScores:
    """Test skill scores DataFrame computation."""

    def _make_confusion_df(self):
        """Create a minimal confusion matrix DataFrame."""
        return pd.DataFrame(
            {
                "lead_hours": [0, 6],
                "model": ["icon_eu", "icon_eu"],
                "event_name": ["frost", "frost"],
                "variable": ["temperature_2m", "temperature_2m"],
                "threshold": [0, 0],
                "operator": ["le", "le"],
                "hits": [8, 5],
                "misses": [2, 5],
                "false_alarms": [1, 3],
                "correct_negatives": [19, 17],
                "excluded_nan": [0, 0],
                "sample_size": [30, 30],
                "total_rows": [30, 30],
            }
        )

    def test_all_metrics_computed(self):
        confusion_df = self._make_confusion_df()
        result = compute_skill_scores(confusion_df)

        # Should have 7 metrics × 2 groups = 14 rows
        assert len(result) == 14
        metrics = result["metric_name"].unique()
        for metric in [
            "hit_rate",
            "false_alarm_rate",
            "precision",
            "csi",
            "miss_rate",
            "accuracy",
            "odds_ratio",
        ]:
            assert metric in metrics

    def test_csi_values(self):
        confusion_df = self._make_confusion_df()
        csi_df = compute_skill_scores(confusion_df)
        csi = csi_df[csi_df["metric_name"] == "csi"]

        # Lead 0: CSI = 8 / (8 + 2 + 1) = 8/11
        csi_0 = csi[csi["lead_hours"] == 0].iloc[0]["metric_value"]
        assert csi_0 == pytest.approx(8 / 11)

        # Lead 6: CSI = 5 / (5 + 5 + 3) = 5/13
        csi_6 = csi[csi["lead_hours"] == 6].iloc[0]["metric_value"]
        assert csi_6 == pytest.approx(5 / 13)

    def test_hit_rate_values(self):
        confusion_df = self._make_confusion_df()
        hr_df = compute_skill_scores(confusion_df)

        # Lead 0: HR = 8 / (8 + 2) = 0.8
        hr_0 = hr_df[hr_df["lead_hours"] == 0].iloc[0]["metric_value"]
        assert hr_0 == pytest.approx(0.8)

    def test_missing_count_propagated(self):
        confusion_df = self._make_confusion_df()
        result = compute_skill_scores(confusion_df)

        # excluded_nan is 0, so missing_count should be 0
        assert (result["missing_count"] == 0).all()

    def test_empty_confusion_df(self):
        result = compute_skill_scores(pd.DataFrame())
        assert len(result) == 0

    def test_nan_heavy_confusion(self):
        """Confusion with all NaN exclusions should produce NaN skill scores."""
        confusion_df = pd.DataFrame(
            {
                "lead_hours": [0],
                "model": ["icon_eu"],
                "event_name": ["frost"],
                "variable": ["temperature_2m"],
                "threshold": [0],
                "operator": ["le"],
                "hits": [0],
                "misses": [0],
                "false_alarms": [0],
                "correct_negatives": [0],
                "excluded_nan": [30],
                "sample_size": [0],
                "total_rows": [30],
            }
        )
        result = compute_skill_scores(confusion_df)
        # All skill scores should be NaN (sample_size == 0)
        assert result["metric_value"].isna().all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestSkillScoreRegistry:
    """Test that all expected metrics are registered."""

    def test_all_metrics_registered(self):
        expected = [
            "hit_rate",
            "false_alarm_rate",
            "precision",
            "csi",
            "miss_rate",
            "accuracy",
            "odds_ratio",
        ]
        for metric in expected:
            assert metric in SKILL_SCORE_FUNCTIONS

    def test_no_extra_metrics(self):
        expected = {
            "hit_rate",
            "false_alarm_rate",
            "precision",
            "csi",
            "miss_rate",
            "accuracy",
            "odds_ratio",
        }
        assert set(SKILL_SCORE_FUNCTIONS.keys()) == expected
