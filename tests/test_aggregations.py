"""Tests for the aggregation engine.

Uses small synthetic verification DataFrames with known expected outputs.
"""

import math

import numpy as np
import pandas as pd
import pytest

from metrics.aggregations import aggregate_metrics
from metrics.schemas import (
    LEAD_HOURS_COL,
    METRIC_NAME_COL,
    METRIC_VALUE_COL,
    MISSING_COUNT_COL,
    SAMPLE_SIZE_COL,
    TOTAL_ROWS_COL,
    VARIABLE_COL,
)


def _make_synthetic_df(
    n_rows: int = 100,
    lead_hours: list[int] | None = None,
    models: list[str] | None = None,
    include_nan: bool = False,
) -> pd.DataFrame:
    """Create a synthetic verification DataFrame.

    Args:
        n_rows: Number of rows.
        lead_hours: Specific lead hours to use. If None, uses 1-24.
        models: Model names. If None, uses ["model_a"].
        include_nan: If True, sets ~10% of errors to NaN.

    Returns:
        DataFrame with provenance + variable columns.
    """
    import pandas as pd

    if lead_hours is None:
        lead_hours = [1, 6, 24]
    if models is None:
        models = ["model_a"]

    np.random.seed(42)

    rows = []
    for lh in lead_hours:
        for model in models:
            for i in range(n_rows // (len(lead_hours) * len(models))):
                forecast = np.random.randn() * 2 + 20
                observed = forecast + np.random.randn() * 1
                error = forecast - observed
                row = {
                    "forecast_issue_time": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i * 6),
                    "forecast_target_time": pd.Timestamp("2025-01-01")
                    + pd.Timedelta(hours=i * 6 + lh),
                    "observation_time": pd.Timestamp("2025-01-01") + pd.Timedelta(hours=i * 6 + lh),
                    "lead_hours": lh,
                    "model": model,
                    "latitude": 55.0,
                    "longitude": 13.0,
                    "elevation": 100.0,
                    "forecast_temperature_2m": forecast,
                    "observed_temperature_2m": observed,
                    "temperature_2m_error": error,
                    "forecast_precipitation": max(0, np.random.exponential(2)),
                    "observed_precipitation": max(0, np.random.exponential(2)),
                    "precipitation_error": max(0, np.random.exponential(1))
                    - max(0, np.random.exponential(1)),
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    if include_nan:
        # Set ~10% of temperature errors to NaN
        nan_mask = np.random.random(len(df)) < 0.1
        df.loc[nan_mask, "temperature_2m_error"] = np.nan

    return df


def _expected_mae_for_errors(errors: list[float]) -> float:
    """Compute expected MAE for a list of errors."""
    clean = [e for e in errors if e is not None and not math.isnan(e)]
    if not clean:
        return float("nan")
    return sum(abs(e) for e in clean) / len(clean)


class TestAggregateByLeadTime:
    """Test aggregation by lead_hours."""

    def test_basic_lead_time_aggregation(self):
        """Verify MAE grouped by lead_hours."""
        df = _make_synthetic_df(lead_hours=[1, 6, 24], models=["model_a"])

        result = aggregate_metrics(
            df,
            groupby=["lead_hours"],
            metrics=["mae"],
            variables=["temperature_2m"],
        )

        # Should have 3 rows (one per lead hour)
        assert len(result) == 3
        assert set(result[LEAD_HOURS_COL].tolist()) == {1, 6, 24}

        # Each row should have correct columns
        expected_cols = [
            LEAD_HOURS_COL,
            VARIABLE_COL,
            METRIC_NAME_COL,
            METRIC_VALUE_COL,
            SAMPLE_SIZE_COL,
            TOTAL_ROWS_COL,
            MISSING_COUNT_COL,
        ]
        assert result.columns.tolist() == expected_cols

        # All sample sizes should be positive
        assert (result[SAMPLE_SIZE_COL] > 0).all()

        # total_rows should equal sample_size (no NaN in this test)
        assert (result[TOTAL_ROWS_COL] == result[SAMPLE_SIZE_COL]).all()

    def test_multi_metric_aggregation(self):
        """Verify multiple metrics are computed."""
        df = _make_synthetic_df(lead_hours=[1, 24], models=["model_a"])

        result = aggregate_metrics(
            df,
            groupby=["lead_hours"],
            metrics=["mae", "rmse", "bias"],
            variables=["temperature_2m"],
        )

        # Should have 6 rows (2 lead_hours × 3 metrics)
        assert len(result) == 6

        # Check unique metrics
        assert set(result[METRIC_NAME_COL].tolist()) == {"mae", "rmse", "bias"}

        # Check unique lead hours
        assert set(result[LEAD_HOURS_COL].tolist()) == {1, 24}

    def test_multi_variable_aggregation(self):
        """Verify multiple variables are computed."""
        df = _make_synthetic_df(lead_hours=[1, 24], models=["model_a"])

        result = aggregate_metrics(
            df,
            groupby=["lead_hours"],
            metrics=["mae"],
            variables=["temperature_2m", "precipitation"],
        )

        # Should have 4 rows (2 lead_hours × 2 variables)
        assert len(result) == 4
        assert set(result[VARIABLE_COL].tolist()) == {"temperature_2m", "precipitation"}

    def test_multi_model_aggregation(self):
        """Verify grouping by model."""
        df = _make_synthetic_df(lead_hours=[1, 24], models=["model_a", "model_b"])

        result = aggregate_metrics(
            df,
            groupby=["lead_hours", "model"],
            metrics=["mae"],
            variables=["temperature_2m"],
        )

        # Should have 4 rows (2 lead_hours × 2 models)
        assert len(result) == 4

        # Check unique combinations
        combos = set(zip(result[LEAD_HOURS_COL], result["model"], strict=False))
        assert combos == {(1, "model_a"), (1, "model_b"), (24, "model_a"), (24, "model_b")}

    def test_rmse_gte_mae(self):
        """RMSE >= MAE within each group (by RMS-AM inequality)."""
        df = _make_synthetic_df(lead_hours=[1, 6, 24], models=["model_a"])

        result = aggregate_metrics(
            df,
            groupby=["lead_hours"],
            metrics=["mae", "rmse"],
            variables=["temperature_2m"],
        )

        # Group by lead_hours and compare
        for lh in result[LEAD_HOURS_COL].unique():
            group = result[result[LEAD_HOURS_COL] == lh]
            mae_row = group[group[METRIC_NAME_COL] == "mae"]
            rmse_row = group[group[METRIC_NAME_COL] == "rmse"]
            assert not mae_row.empty and not rmse_row.empty
            assert (
                rmse_row[METRIC_VALUE_COL].values[0] >= mae_row[METRIC_VALUE_COL].values[0] - 1e-10
            )

    def test_bias_sign_preserved(self):
        """Bias preserves sign (positive = overprediction)."""
        # Create data where forecast consistently overpredicts
        np.random.seed(123)
        df = pd.DataFrame(
            {
                "forecast_issue_time": pd.date_range("2025-01-01", periods=50),
                "forecast_target_time": pd.date_range("2025-01-02", periods=50),
                "observation_time": pd.date_range("2025-01-02", periods=50),
                "lead_hours": 24,
                "model": "model_a",
                "latitude": 55.0,
                "longitude": 13.0,
                "elevation": 100.0,
                "forecast_temperature_2m": 20.0 + np.random.randn(50) * 2,
                "observed_temperature_2m": 18.0 + np.random.randn(50) * 2,
                "temperature_2m_error": 2.0 + np.random.randn(50) * 2,  # positive bias
                "forecast_precipitation": np.abs(np.random.randn(50)),
                "observed_precipitation": np.abs(np.random.randn(50)),
                "precipitation_error": np.random.randn(50),
            }
        )

        result = aggregate_metrics(
            df,
            groupby=["lead_hours"],
            metrics=["bias"],
            variables=["temperature_2m"],
        )

        bias_val = result[result[METRIC_NAME_COL] == "bias"][METRIC_VALUE_COL].values[0]
        assert bias_val > 0, f"Expected positive bias (overprediction), got {bias_val}"


class TestAggregateValidation:
    """Test aggregation input validation."""

    def test_invalid_metric(self):
        df = _make_synthetic_df()
        with pytest.raises(ValueError, match="Unknown metric"):
            aggregate_metrics(
                df, groupby=["lead_hours"], metrics=["unknown"], variables=["temperature_2m"]
            )

    def test_invalid_dimension(self):
        df = _make_synthetic_df()
        with pytest.raises(ValueError, match="Invalid groupby dimension"):
            aggregate_metrics(
                df, groupby=["invalid_dim"], metrics=["mae"], variables=["temperature_2m"]
            )

    def test_missing_error_column(self):
        df = _make_synthetic_df()
        with pytest.raises(ValueError, match="Missing error column"):
            aggregate_metrics(
                df, groupby=["lead_hours"], metrics=["mae"], variables=["nonexistent"]
            )

    def test_duplicate_metrics(self):
        df = _make_synthetic_df()
        with pytest.raises(ValueError, match="Duplicate metrics"):
            aggregate_metrics(
                df, groupby=["lead_hours"], metrics=["mae", "mae"], variables=["temperature_2m"]
            )

    def test_duplicate_dimensions(self):
        df = _make_synthetic_df()
        with pytest.raises(ValueError, match="Duplicate groupby dimensions"):
            aggregate_metrics(
                df,
                groupby=["lead_hours", "lead_hours"],
                metrics=["mae"],
                variables=["temperature_2m"],
            )


class TestAggregateSampleSize:
    """Test sample_size, total_rows, missing_count computation."""

    def test_with_nan(self):
        """Verify sample_size < total_rows when NaN errors exist."""
        df = _make_synthetic_df(lead_hours=[1, 24], models=["model_a"], include_nan=True)

        result = aggregate_metrics(
            df,
            groupby=["lead_hours"],
            metrics=["mae"],
            variables=["temperature_2m"],
        )

        # total_rows should equal sample_size + missing_count
        assert (result[TOTAL_ROWS_COL] == result[SAMPLE_SIZE_COL] + result[MISSING_COUNT_COL]).all()

        # missing_count should be > 0 (we added NaN values)
        assert (result[MISSING_COUNT_COL] > 0).all()

        # sample_size should be < total_rows
        assert (result[SAMPLE_SIZE_COL] < result[TOTAL_ROWS_COL]).all()

    def test_without_nan(self):
        """Verify sample_size == total_rows when no NaN errors exist."""
        df = _make_synthetic_df(lead_hours=[1, 24], models=["model_a"], include_nan=False)

        result = aggregate_metrics(
            df,
            groupby=["lead_hours"],
            metrics=["mae"],
            variables=["temperature_2m"],
        )

        assert (result[TOTAL_ROWS_COL] == result[SAMPLE_SIZE_COL]).all()
        assert (result[MISSING_COUNT_COL] == 0).all()
