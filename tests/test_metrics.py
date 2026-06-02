"""Tests for pure metric computation functions.

Uses small synthetic datasets with known expected outputs.
"""

import math

import numpy as np
import pandas as pd
import pytest

from metrics.metrics import (
    compute_bias,
    compute_mae,
    compute_rmse,
    compute_std,
    get_metric_function,
)


class TestComputeMAE:
    """Tests for compute_mae."""

    def test_simple(self):
        """MAE of [1, 2, 3] = 2.0."""
        errors = pd.Series([1.0, 2.0, 3.0])
        assert compute_mae(errors) == pytest.approx(2.0)

    def test_negative_errors(self):
        """MAE of [-1, -2, -3] = 2.0."""
        errors = pd.Series([-1.0, -2.0, -3.0])
        assert compute_mae(errors) == pytest.approx(2.0)

    def test_mixed_errors(self):
        """MAE of [1, -2, 3] = 2.0."""
        errors = pd.Series([1.0, -2.0, 3.0])
        assert compute_mae(errors) == pytest.approx(2.0)

    def test_zeros(self):
        """MAE of all zeros = 0.0."""
        errors = pd.Series([0.0, 0.0, 0.0])
        assert compute_mae(errors) == 0.0

    def test_single_value(self):
        """MAE of [5.0] = 5.0."""
        errors = pd.Series([5.0])
        assert compute_mae(errors) == 5.0

    def test_nan_values_excluded(self):
        """NaN values are excluded from MAE computation."""
        errors = pd.Series([1.0, float("nan"), 3.0])
        assert compute_mae(errors) == pytest.approx(2.0)

    def test_all_nan(self):
        """MAE of all NaN = NaN."""
        errors = pd.Series([float("nan"), float("nan")])
        assert math.isnan(compute_mae(errors))

    def test_empty_series(self):
        """MAE of empty series = NaN."""
        errors = pd.Series([], dtype=float)
        assert math.isnan(compute_mae(errors))


class TestComputeRMSE:
    """Tests for compute_rmse."""

    def test_simple(self):
        """RMSE of [1, 2, 3] = sqrt((1+4+9)/3) = sqrt(14/3) ≈ 2.160."""
        errors = pd.Series([1.0, 2.0, 3.0])
        expected = math.sqrt((1 + 4 + 9) / 3)
        assert compute_rmse(errors) == pytest.approx(expected)

    def test_zeros(self):
        """RMSE of all zeros = 0.0."""
        errors = pd.Series([0.0, 0.0, 0.0])
        assert compute_rmse(errors) == 0.0

    def test_single_value(self):
        """RMSE of [5.0] = 5.0."""
        errors = pd.Series([5.0])
        assert compute_rmse(errors) == 5.0

    def test_nan_values_excluded(self):
        """NaN values are excluded from RMSE computation."""
        errors = pd.Series([1.0, float("nan"), 3.0])
        expected = math.sqrt((1 + 9) / 2)
        assert compute_rmse(errors) == pytest.approx(expected)

    def test_all_nan(self):
        """RMSE of all NaN = NaN."""
        errors = pd.Series([float("nan"), float("nan")])
        assert math.isnan(compute_rmse(errors))

    def test_rmse_gte_mae(self):
        """RMSE >= MAE always (by RMS-AM inequality)."""
        for _ in range(10):
            errors = pd.Series(np.random.randn(100))
            assert compute_rmse(errors) >= compute_mae(errors) - 1e-10


class TestComputeBias:
    """Tests for compute_bias."""

    def test_positive_bias(self):
        """Positive errors → positive bias (overprediction)."""
        errors = pd.Series([1.0, 2.0, 3.0])
        assert compute_bias(errors) == pytest.approx(2.0)

    def test_negative_bias(self):
        """Negative errors → negative bias (underprediction)."""
        errors = pd.Series([-1.0, -2.0, -3.0])
        assert compute_bias(errors) == pytest.approx(-2.0)

    def test_zero_bias(self):
        """Equal positive and negative errors → zero bias."""
        errors = pd.Series([1.0, -1.0, 2.0, -2.0])
        assert compute_bias(errors) == 0.0

    def test_single_value(self):
        """Bias of [5.0] = 5.0."""
        errors = pd.Series([5.0])
        assert compute_bias(errors) == 5.0

    def test_nan_values_excluded(self):
        """NaN values are excluded from bias computation."""
        errors = pd.Series([1.0, float("nan"), 3.0])
        assert compute_bias(errors) == pytest.approx(2.0)

    def test_all_nan(self):
        """Bias of all NaN = NaN."""
        errors = pd.Series([float("nan"), float("nan")])
        assert math.isnan(compute_bias(errors))


class TestComputeStd:
    """Tests for compute_std (ddof=1)."""

    def test_known_std(self):
        """Sample std of [1, 2, 3, 4, 5] with ddof=1."""
        errors = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        # Manual calculation: mean=3, sum_sq_diff=10, var=10/4=2.5, std=sqrt(2.5)
        expected = math.sqrt(2.5)
        assert compute_std(errors) == pytest.approx(expected)

    def test_ddof_1_not_0(self):
        """std(ddof=1) > std(ddof=0) for the same data."""
        errors = pd.Series([1.0, 2.0, 3.0])
        s = compute_std(errors)
        population_std = errors.std(ddof=0)
        assert s > population_std

    def test_two_values(self):
        """std of [0, 2] with ddof=1 = sqrt(2)."""
        errors = pd.Series([0.0, 2.0])
        expected = math.sqrt(2)
        assert compute_std(errors) == pytest.approx(expected)

    def test_single_value(self):
        """std of single value = NaN (need >= 2 for ddof=1)."""
        errors = pd.Series([5.0])
        assert math.isnan(compute_std(errors))

    def test_nan_values_excluded(self):
        """NaN values are excluded from std computation."""
        errors = pd.Series([1.0, float("nan"), 3.0])
        # std of [1, 3] with ddof=1: mean=2, sum_sq_diff=1+1=2, var=2/1=2, std=sqrt(2)
        expected = math.sqrt(2)
        assert compute_std(errors) == pytest.approx(expected)

    def test_all_nan(self):
        """std of all NaN = NaN."""
        errors = pd.Series([float("nan"), float("nan")])
        assert math.isnan(compute_std(errors))

    def test_empty_series(self):
        """std of empty series = NaN."""
        errors = pd.Series([], dtype=float)
        assert math.isnan(compute_std(errors))


class TestGetMetricFunction:
    """Tests for the metric registry."""

    def test_mae(self):
        assert get_metric_function("mae") is compute_mae

    def test_rmse(self):
        assert get_metric_function("rmse") is compute_rmse

    def test_bias(self):
        assert get_metric_function("bias") is compute_bias

    def test_std(self):
        assert get_metric_function("std") is compute_std

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric_function("unknown")
