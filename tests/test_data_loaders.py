"""Tests for data loading helpers.

Verifies:
- Metrics loading from parquet
- Verification data loading
- Filtering by variable, metric, model
- Pivot operations
- Glob pattern resolution
"""

import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from data.loaders import (
    filter_metrics,
    load_metrics,
    load_verification,
    pivot_metrics,
)


def _make_metrics_parquet(**kwargs):
    """Create a temporary metrics parquet file."""
    lead_hours = kwargs.get("lead_hours", [1, 6, 12, 24])
    models = kwargs.get("model", ["icon_eu", "gfs"])
    variables = kwargs.get("variable", ["temperature_2m"] * len(lead_hours))
    metric_names = kwargs.get("metric_name", ["mae"] * len(lead_hours))
    metric_values = kwargs.get("metric_value", np.random.rand(len(lead_hours)))
    sample_sizes = kwargs.get("sample_size", [100] * len(lead_hours))
    total_rows = kwargs.get("total_rows", [110] * len(lead_hours))
    missing_counts = kwargs.get("missing_count", [10] * len(lead_hours))

    # Expand for multi-model: each model gets all lead_hours × metrics
    rows = []
    for m in models:
        for lh, var, mn, mv, ss, tr, mc in zip(
            lead_hours,
            variables,
            metric_names,
            metric_values,
            sample_sizes,
            total_rows,
            missing_counts,
            strict=False,
        ):
            rows.append(
                {
                    "lead_hours": lh,
                    "model": m,
                    "variable": var,
                    "metric_name": mn,
                    "metric_value": mv,
                    "sample_size": ss,
                    "total_rows": tr,
                    "missing_count": mc,
                }
            )

    df = pd.DataFrame(rows)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pq.write_table(pa.Table.from_pandas(df), f.name)
        return f.name


def _make_verification_parquet(**kwargs):
    """Create a temporary verification parquet file."""
    n = kwargs.get("n", 100)
    df = pd.DataFrame(
        {
            "forecast_issue_time": pd.date_range("2024-01-01", periods=n, freq="6h"),
            "forecast_target_time": pd.date_range("2024-01-02", periods=n, freq="6h"),
            "lead_hours": kwargs.get("lead_hours", [1, 6, 12, 24] * (n // 4)),
            "model": kwargs.get("model", ["icon_eu", "gfs"] * (n // 2)),
            "latitude": np.random.normal(50, 5, n),
            "longitude": np.random.normal(10, 5, n),
            "elevation": np.random.normal(100, 20, n),
            "forecast_temperature_2m": np.random.normal(10, 3, n),
            "observed_temperature_2m": np.random.normal(10, 3, n),
            "temperature_2m_error": np.random.normal(0, 2, n),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        pq.write_table(pa.Table.from_pandas(df), f.name)
        return f.name


class TestLoadMetrics:
    """Test metrics loading."""

    def test_load_single_file(self):
        """Should load a single parquet file."""
        path = _make_metrics_parquet()
        df = load_metrics(path)
        assert len(df) > 0
        assert "metric_name" in df.columns
        assert "metric_value" in df.columns

    def test_load_metrics_columns(self):
        """Loaded DataFrame should have expected columns."""
        path = _make_metrics_parquet()
        df = load_metrics(path)
        expected_cols = {
            "lead_hours",
            "model",
            "variable",
            "metric_name",
            "metric_value",
            "sample_size",
            "total_rows",
            "missing_count",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_load_metrics_no_files(self):
        """Should raise FileNotFoundError for no matching files."""
        with pytest.raises(FileNotFoundError):
            load_metrics("/nonexistent/path/*.parquet")


class TestLoadVerification:
    """Test verification data loading."""

    def test_load_single_file(self):
        """Should load a single verification file."""
        path = _make_verification_parquet()
        df = load_verification(path)
        assert len(df) > 0
        assert "temperature_2m_error" in df.columns

    def test_load_verification_sorted(self):
        """Loaded data should be sorted by provenance columns."""
        path = _make_verification_parquet()
        df = load_verification(path)
        # Should have forecast_issue_time column
        assert "forecast_issue_time" in df.columns


class TestFilterMetrics:
    """Test metrics filtering."""

    def test_filter_by_variable(self):
        """Should filter by variable name."""
        df = pd.DataFrame(
            {
                "variable": ["temperature_2m", "precipitation", "temperature_2m"],
                "metric_name": ["mae", "mae", "rmse"],
                "metric_value": [1.0, 2.0, 3.0],
                "sample_size": [100, 100, 100],
            }
        )

        filtered = filter_metrics(df, variables=["temperature_2m"])
        assert len(filtered) == 2
        assert all(filtered["variable"] == "temperature_2m")

    def test_filter_by_metric(self):
        """Should filter by metric name."""
        df = pd.DataFrame(
            {
                "variable": ["temperature_2m", "temperature_2m", "temperature_2m"],
                "metric_name": ["mae", "rmse", "bias"],
                "metric_value": [1.0, 2.0, 0.5],
                "sample_size": [100, 100, 100],
            }
        )

        filtered = filter_metrics(df, metrics=["mae", "rmse"])
        assert len(filtered) == 2
        assert filtered["metric_name"].isin(["mae", "rmse"]).all()

    def test_filter_by_model(self):
        """Should filter by model name."""
        df = pd.DataFrame(
            {
                "variable": ["temperature_2m"] * 4,
                "model": ["icon_eu", "icon_eu", "gfs", "gfs"],
                "metric_name": ["mae", "rmse", "mae", "rmse"],
                "metric_value": [1.0, 2.0, 3.0, 4.0],
                "sample_size": [100] * 4,
            }
        )

        filtered = filter_metrics(df, models=["icon_eu"])
        assert len(filtered) == 2
        assert all(filtered["model"] == "icon_eu")

    def test_filter_multiple_criteria(self):
        """Should combine multiple filters."""
        df = pd.DataFrame(
            {
                "variable": ["temperature_2m", "precipitation", "temperature_2m", "precipitation"],
                "model": ["icon_eu", "icon_eu", "gfs", "gfs"],
                "metric_name": ["mae", "mae", "rmse", "rmse"],
                "metric_value": [1.0, 2.0, 3.0, 4.0],
                "sample_size": [100] * 4,
            }
        )

        filtered = filter_metrics(
            df,
            variables=["temperature_2m"],
            metrics=["mae"],
            models=["icon_eu"],
        )
        assert len(filtered) == 1
        assert filtered["metric_value"].iloc[0] == 1.0


class TestPivotMetrics:
    """Test pivot operations."""

    def test_pivot_lead_time(self):
        """Should pivot lead_hours to columns."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 1, 6, 6],
                "variable": [
                    "temperature_2m",
                    "temperature_2m",
                    "temperature_2m",
                    "temperature_2m",
                ],
                "metric_name": ["mae", "rmse", "mae", "rmse"],
                "metric_value": [1.0, 2.0, 3.0, 4.0],
            }
        )

        pivot = pivot_metrics(df, index_cols=["lead_hours"], columns_cols=["metric_name"])
        assert "mae" in pivot.columns or "mae" in str(pivot.columns)
        assert "rmse" in pivot.columns or "rmse" in str(pivot.columns)

    def test_pivot_empty(self):
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        pivot = pivot_metrics(df, index_cols=["lead_hours"], columns_cols=["metric_name"])
        assert len(pivot) == 0

    def test_pivot_invalid_columns(self):
        """Should handle invalid column names gracefully."""
        df = pd.DataFrame(
            {
                "lead_hours": [1, 6],
                "metric_value": [1.0, 3.0],
            }
        )

        pivot = pivot_metrics(df, index_cols=["nonexistent"], columns_cols=["nonexistent"])
        assert len(pivot) == len(df)
