"""Human-readable scenario tests for weather forecast verification.

These tests use realistic weather forecasting scenarios with plain language
descriptions. They complement the technical unit tests with examples that
are easier to understand for domain experts and new contributors.
"""

import math

import numpy as np
import pandas as pd
import pytest

from metrics.aggregations import aggregate_metrics
from metrics.metrics import compute_bias, compute_mae, compute_rmse, compute_std
from metrics.validation import validate_metrics

# =============================================================================
# SCENARIO 1: A forecaster predicts temperature for 3 days
# =============================================================================


class TestTemperatureForecastErrors:
    """Tests using a simple temperature forecasting scenario.

    Scenario: A forecaster predicts the temperature for 3 consecutive days.
    We compute how accurate the forecasts were.
    """

    @pytest.fixture
    def three_day_forecast(self):
        """Three days of temperature forecasts vs observations.

        Day 1: Forecast 20°C, Observed 18°C → Error +2°C (too warm)
        Day 2: Forecast 22°C, Observed 24°C → Error -2°C (too cold)
        Day 3: Forecast 19°C, Observed 19°C → Error  0°C (perfect)
        """
        return pd.Series([2.0, -2.0, 0.0])

    def test_average_absolute_error_is_1_point_3_degrees(self, three_day_forecast):
        """The average magnitude of errors is |2| + |-2| + |0| / 3 = 1.33°C."""
        mae = compute_mae(three_day_forecast)
        assert mae == pytest.approx(4 / 3)  # 1.333...

    def test_forecast_has_no_systematic_bias(self, three_day_forecast):
        """The forecast is not systematically too warm or too cold.

        Bias = (2 + -2 + 0) / 3 = 0
        """
        bias = compute_bias(three_day_forecast)
        assert bias == pytest.approx(0.0)

    def test_rmse_penalizes_larger_errors_more(self, three_day_forecast):
        """RMSE weights larger errors more heavily than MAE.

        RMSE = sqrt((4 + 4 + 0) / 3) = sqrt(8/3) ≈ 1.63°C
        This is higher than MAE (1.33) because the 2-degree errors are squared.
        """
        rmse = compute_rmse(three_day_forecast)
        mae = compute_mae(three_day_forecast)

        assert rmse == pytest.approx(math.sqrt(8 / 3))
        assert rmse > mae  # RMSE is always >= MAE


# =============================================================================
# SCENARIO 2: Comparing a warm-biased vs cold-biased model
# =============================================================================


class TestBiasDetection:
    """Tests for detecting systematic forecast bias.

    Scenario: We have two weather models. Model A consistently predicts
    temperatures that are too warm. Model B consistently predicts too cold.
    """

    @pytest.fixture
    def warm_biased_model(self):
        """Model that consistently overpredicts temperature by 2°C."""
        return pd.Series([2.0, 3.0, 1.5, 2.5, 2.0])

    @pytest.fixture
    def cold_biased_model(self):
        """Model that consistently underpredicts temperature by 2°C."""
        return pd.Series([-2.0, -3.0, -1.5, -2.5, -2.0])

    def test_warm_model_has_positive_bias(self, warm_biased_model):
        """A model that predicts too warm has positive bias."""
        bias = compute_bias(warm_biased_model)
        assert bias > 0
        assert bias == pytest.approx(2.2)  # average of the errors

    def test_cold_model_has_negative_bias(self, cold_biased_model):
        """A model that predicts too cold has negative bias."""
        bias = compute_bias(cold_biased_model)
        assert bias < 0
        assert bias == pytest.approx(-2.2)

    def test_both_models_have_same_mae(self, warm_biased_model, cold_biased_model):
        """MAE doesn't care about direction, only magnitude."""
        mae_warm = compute_mae(warm_biased_model)
        mae_cold = compute_mae(cold_biased_model)
        assert mae_warm == pytest.approx(mae_cold)


# =============================================================================
# SCENARIO 3: Forecast skill degrades with lead time
# =============================================================================


class TestForecastSkillByLeadTime:
    """Tests for how forecast accuracy changes with lead time.

    Scenario: Weather forecasts are usually more accurate for tomorrow
    than for next week. We verify this pattern in our metrics.
    """

    @pytest.fixture
    def forecast_data(self):
        """Synthetic data where errors increase with lead time.

        1-hour forecast: small errors (±0.5°C)
        24-hour forecast: medium errors (±2°C)
        168-hour forecast (7 days): large errors (±5°C)
        """
        np.random.seed(42)
        rows = []

        for lead_hours, error_scale in [(1, 0.5), (24, 2.0), (168, 5.0)]:
            for i in range(50):
                error = np.random.randn() * error_scale
                rows.append(
                    {
                        "forecast_issue_time": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i),
                        "forecast_target_time": pd.Timestamp("2025-01-01")
                        + pd.Timedelta(hours=i + lead_hours),
                        "observation_time": pd.Timestamp("2025-01-01")
                        + pd.Timedelta(hours=i + lead_hours),
                        "lead_hours": lead_hours,
                        "model": "test_model",
                        "latitude": 55.0,
                        "longitude": 13.0,
                        "elevation": 100.0,
                        "forecast_temperature_2m": 20.0,
                        "observed_temperature_2m": 20.0 - error,
                        "temperature_2m_error": error,
                        "forecast_precipitation": 0.0,
                        "observed_precipitation": 0.0,
                        "precipitation_error": 0.0,
                    }
                )

        return pd.DataFrame(rows)

    def test_one_hour_forecast_is_most_accurate(self, forecast_data):
        """1-hour forecasts should have the lowest MAE."""
        result = aggregate_metrics(
            forecast_data,
            groupby=["lead_hours"],
            metrics=["mae"],
            variables=["temperature_2m"],
        )

        mae_by_lead = dict(zip(result["lead_hours"], result["metric_value"], strict=False))

        assert mae_by_lead[1] < mae_by_lead[24] < mae_by_lead[168]

    def test_seven_day_forecast_has_largest_errors(self, forecast_data):
        """168-hour (7-day) forecasts should have the highest RMSE."""
        result = aggregate_metrics(
            forecast_data,
            groupby=["lead_hours"],
            metrics=["rmse"],
            variables=["temperature_2m"],
        )

        rmse_by_lead = dict(zip(result["lead_hours"], result["metric_value"], strict=False))

        assert rmse_by_lead[168] > rmse_by_lead[24] > rmse_by_lead[1]


# =============================================================================
# SCENARIO 4: Handling missing observations
# =============================================================================


class TestMissingObservations:
    """Tests for handling gaps in observation data.

    Scenario: Weather stations sometimes go offline, leaving gaps in the
    observed data. The metrics should handle these gracefully.
    """

    @pytest.fixture
    def data_with_gaps(self):
        """Forecast errors with some missing values (represented as NaN)."""
        return pd.Series([1.0, 2.0, float("nan"), 4.0, float("nan")])

    def test_missing_values_are_excluded_from_mae(self, data_with_gaps):
        """MAE only uses the 3 valid observations: (1 + 2 + 4) / 3 = 2.33."""
        mae = compute_mae(data_with_gaps)
        assert mae == pytest.approx(7 / 3)

    def test_all_missing_returns_nan(self):
        """If all observations are missing, the metric is undefined."""
        all_missing = pd.Series([float("nan"), float("nan"), float("nan")])

        assert math.isnan(compute_mae(all_missing))
        assert math.isnan(compute_rmse(all_missing))
        assert math.isnan(compute_bias(all_missing))


# =============================================================================
# SCENARIO 5: Comparing model performance
# =============================================================================


class TestModelComparison:
    """Tests for comparing two weather models.

    Scenario: We want to determine which of two models performs better.
    Model Alpha has consistent medium-sized errors.
    Model Beta occasionally has very large errors.
    """

    @pytest.fixture
    def model_alpha_errors(self):
        """Consistent 2°C errors every day."""
        return pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])

    @pytest.fixture
    def model_beta_errors(self):
        """Mostly good but occasional large error."""
        return pd.Series([0.5, 0.5, 0.5, 0.5, 8.0])

    def test_mae_prefers_model_beta(self, model_alpha_errors, model_beta_errors):
        """MAE: Beta (2.0) equals Alpha (2.0) - same average error."""
        mae_alpha = compute_mae(model_alpha_errors)
        mae_beta = compute_mae(model_beta_errors)

        assert mae_alpha == pytest.approx(2.0)
        assert mae_beta == pytest.approx(2.0)

    def test_rmse_penalizes_model_beta_for_outlier(self, model_alpha_errors, model_beta_errors):
        """RMSE penalizes Beta's 8°C outlier more heavily.

        Alpha RMSE = 2.0 (all errors equal)
        Beta RMSE = sqrt((0.25*4 + 64)/5) = sqrt(65/5) ≈ 3.6
        """
        rmse_alpha = compute_rmse(model_alpha_errors)
        rmse_beta = compute_rmse(model_beta_errors)

        assert rmse_alpha == pytest.approx(2.0)
        assert rmse_beta > rmse_alpha
        assert rmse_beta == pytest.approx(math.sqrt(65 / 5))

    def test_model_beta_has_higher_variability(self, model_alpha_errors, model_beta_errors):
        """Standard deviation captures Beta's inconsistency."""
        std_alpha = compute_std(model_alpha_errors)
        std_beta = compute_std(model_beta_errors)

        assert std_alpha == pytest.approx(0.0)  # perfectly consistent
        assert std_beta > 0  # varies due to outlier


# =============================================================================
# SCENARIO 6: Validating output quality
# =============================================================================


class TestOutputValidation:
    """Tests for catching data quality issues.

    Scenario: Before publishing forecast verification results, we run
    validation checks to catch potential problems.
    """

    def test_valid_results_pass_validation(self):
        """Clean data should pass validation without errors."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 1, 24, 24],
                "variable": ["temperature_2m"] * 4,
                "metric_name": ["mae", "rmse", "mae", "rmse"],
                "metric_value": [1.5, 2.0, 2.5, 3.0],
                "sample_size": [100, 100, 100, 100],
                "total_rows": [100, 100, 100, 100],
                "missing_count": [0, 0, 0, 0],
            }
        )

        issues = validate_metrics(df)
        errors = [i for i in issues if i.severity == "error"]

        assert len(errors) == 0

    def test_catches_negative_sample_size(self):
        """Sample size should never be negative."""
        df = pd.DataFrame(
            {
                "lead_hours": [1],
                "variable": ["temperature_2m"],
                "metric_name": ["mae"],
                "metric_value": [1.5],
                "sample_size": [-10],  # bug!
                "total_rows": [100],
                "missing_count": [0],
            }
        )

        issues = validate_metrics(df)
        errors = [i for i in issues if i.severity == "error"]

        assert len(errors) > 0
        assert any("negative_sample_size" in i.check for i in errors)

    def test_catches_absurd_rmse_ratio(self):
        """RMSE > 100x MAE indicates a computation error."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 1],
                "variable": ["temperature_2m", "temperature_2m"],
                "metric_name": ["mae", "rmse"],
                "metric_value": [1.0, 200.0],  # RMSE is 200x MAE
                "sample_size": [100, 100],
                "total_rows": [100, 100],
                "missing_count": [0, 0],
            }
        )

        issues = validate_metrics(df)
        errors = [i for i in issues if i.severity == "error"]

        assert len(errors) > 0
        assert any("absurd_rmse" in i.check for i in errors)

    def test_warns_about_high_missing_rate(self):
        """95% missing data should trigger a warning."""
        df = pd.DataFrame(
            {
                "lead_hours": [1],
                "variable": ["temperature_2m"],
                "metric_name": ["mae"],
                "metric_value": [1.5],
                "sample_size": [5],
                "total_rows": [100],
                "missing_count": [95],
            }
        )

        issues = validate_metrics(df)
        warnings = [i for i in issues if i.severity == "warning"]

        assert any("nan_heavy" in i.check for i in warnings)


# =============================================================================
# SCENARIO 7: Real-world mathematical properties
# =============================================================================


class TestMathematicalProperties:
    """Tests verifying mathematical properties always hold.

    These are invariants that must be true regardless of the data.
    """

    def test_mae_is_always_non_negative(self):
        """MAE cannot be negative (it's an average of absolute values)."""
        for _ in range(20):
            errors = pd.Series(np.random.randn(100) * 10)
            mae = compute_mae(errors)
            assert mae >= 0

    def test_rmse_is_always_greater_or_equal_to_mae(self):
        """RMSE >= MAE always (consequence of squaring)."""
        for _ in range(20):
            errors = pd.Series(np.random.randn(100) * 10)
            mae = compute_mae(errors)
            rmse = compute_rmse(errors)
            assert rmse >= mae - 1e-10  # small epsilon for float comparison

    def test_std_requires_at_least_two_values(self):
        """Standard deviation is undefined for a single value."""
        single = pd.Series([5.0])
        assert math.isnan(compute_std(single))

        two_values = pd.Series([3.0, 7.0])
        assert not math.isnan(compute_std(two_values))

    def test_zero_bias_does_not_mean_zero_error(self):
        """A model can have zero bias but large errors.

        If errors are +5 and -5, bias = 0 but MAE = 5.
        """
        balanced_errors = pd.Series([5.0, -5.0, 10.0, -10.0])

        assert compute_bias(balanced_errors) == pytest.approx(0.0)
        assert compute_mae(balanced_errors) == pytest.approx(7.5)
