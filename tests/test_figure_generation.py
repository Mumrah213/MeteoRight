"""Tests for figure generation stability.

Verifies that plotting functions execute without error on synthetic data.
Does NOT test visual appearance — only that figures are created successfully.
"""

import tempfile

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.distributions import (
    plot_error_boxplot,
    plot_error_histogram,
    plot_error_percentiles,
)
from plots.lead_time import (
    plot_lead_time_all_metrics,
    plot_lead_time_degradation,
    plot_lead_time_multi_variable,
)
from plots.model_comparison import (
    plot_model_bias_comparison,
    plot_model_comparison,
    plot_model_grouped,
)
from plots.precipitation import (
    classify_events,
    plot_event_contingency,
    plot_event_frequency,
    plot_hit_miss_by_threshold,
)
from plots.seasonal import (
    plot_monthly_metrics,
    plot_seasonal_degradation,
    plot_seasonal_metrics,
)
from plots.utils import (
    label_axes,
    make_figure,
    make_multi_panel_figure,
    save_figure,
)


def _make_metrics_df(**kwargs):
    """Create a synthetic metrics DataFrame."""
    from tests.test_grouping_semantics import _make_metrics_df

    return _make_metrics_df(
        seasons=kwargs.get("seasons", ["DJF", "MAM", "JJA", "SON"]),
        **{k: v for k, v in kwargs.items() if k != "seasons"},
    )


def _make_verification_df(**kwargs):
    """Create a synthetic verification DataFrame."""
    n = kwargs.get("n", 500)
    lead_hours = kwargs.get("lead_hours", [1, 6, 12, 24, 48])
    models = kwargs.get("models", ["icon_eu", "gfs"])

    rows = []
    for _ in range(n):
        lh = np.random.choice(lead_hours)
        model = np.random.choice(models)
        temp_error = np.random.normal(0, 2.0 + lh * 0.1)
        precip_error = np.random.exponential(1.0 + lh * 0.05)
        precip_obs = max(0, np.random.exponential(2.0))
        precip_fcst = max(0, precip_obs + precip_error)

        rows.append(
            {
                "forecast_issue_time": pd.Timestamp("2024-01-01")
                + pd.Timedelta(hours=np.random.randint(0, 720)),
                "forecast_target_time": pd.Timestamp("2024-01-01")
                + pd.Timedelta(hours=np.random.randint(0, 720)),
                "lead_hours": lh,
                "model": model,
                "latitude": 50.0 + np.random.normal(0, 10),
                "longitude": 10.0 + np.random.normal(0, 10),
                "elevation": 100 + np.random.normal(0, 50),
                "forecast_temperature_2m": 10.0 + np.random.normal(0, 5),
                "observed_temperature_2m": 10.0 + np.random.normal(0, 5) - temp_error,
                "temperature_2m_error": temp_error,
                "forecast_precipitation": precip_fcst,
                "observed_precipitation": precip_obs,
                "precipitation_error": precip_error,
            }
        )

    return pd.DataFrame(rows)


class TestPlotUtilities:
    """Test basic plotting utilities."""

    def test_make_figure(self):
        """make_figure should create a figure and axes."""
        fig, ax = make_figure(title="Test", xlabel="X", ylabel="Y")
        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "Test"
        plt.close(fig)

    def test_make_multi_panel_figure(self):
        """make_multi_panel_figure should create a grid of axes."""
        fig, axes = make_multi_panel_figure(n_rows=2, n_cols=2)
        assert fig is not None
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_label_axes(self):
        """label_axes should set labels correctly."""
        fig, ax = plt.subplots()
        result = label_axes(ax, "X label", "Y label", "Title")
        assert result.get_xlabel() == "X label"
        assert result.get_ylabel() == "Y label"
        assert result.get_title() == "Title"
        plt.close(fig)

    def test_save_figure(self):
        """save_figure should write a file."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        save_figure(fig, path)

        import os

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        os.unlink(path)
        plt.close(fig)


class TestLeadTimePlots:
    """Test lead-time degradation plot generation."""

    def test_plot_lead_time_degradation(self):
        """Should create a figure without error."""
        df = _make_metrics_df()
        fig = plot_lead_time_degradation(df, "temperature_2m", "mae")
        assert fig is not None
        plt.close(fig)

    def test_plot_lead_time_degradation_with_models(self):
        """Should handle multiple models."""
        df = _make_metrics_df(models=["icon_eu", "gfs"])
        fig = plot_lead_time_degradation(df, "temperature_2m", "mae", models=["icon_eu", "gfs"])
        assert fig is not None
        plt.close(fig)

    def test_plot_lead_time_all_metrics(self):
        """Should create multi-metric plot."""
        df = _make_metrics_df()
        fig = plot_lead_time_all_metrics(df, "temperature_2m")
        assert fig is not None
        plt.close(fig)

    def test_plot_lead_time_multi_variable(self):
        """Should handle multiple variables."""
        df = _make_metrics_df(variables=["temperature_2m", "precipitation"])
        fig = plot_lead_time_multi_variable(df, ["temperature_2m", "precipitation"], "mae")
        assert fig is not None
        plt.close(fig)

    def test_plot_lead_time_degradation_no_data(self):
        """Should return None for missing data."""
        df = _make_metrics_df()
        fig = plot_lead_time_degradation(df, "nonexistent", "mae")
        assert fig is None


class TestModelComparisonPlots:
    """Test model comparison plot generation."""

    def test_plot_model_comparison(self):
        """Should create model comparison plot."""
        df = _make_metrics_df(models=["icon_eu", "gfs"])
        fig = plot_model_comparison(df, "temperature_2m", "mae", models=["icon_eu", "gfs"])
        assert fig is not None
        plt.close(fig)

    def test_plot_model_bias_comparison(self):
        """Should create bias comparison plot."""
        df = _make_metrics_df(metrics=["bias"])
        fig = plot_model_bias_comparison(df, "temperature_2m", models=["icon_eu"])
        assert fig is not None
        plt.close(fig)

    def test_plot_model_grouped(self):
        """Should create grouped model plot."""
        df = _make_metrics_df(models=["icon_eu", "gfs"])
        fig = plot_model_grouped(df, "temperature_2m", ["mae", "rmse"], models=["icon_eu", "gfs"])
        assert fig is not None
        plt.close(fig)


class TestSeasonalPlots:
    """Test seasonal plot generation."""

    def test_plot_seasonal_metrics(self):
        """Should create seasonal metrics plot."""
        df = _make_metrics_df(
            seasons=["DJF", "MAM", "JJA", "SON"],
            lead_hours=[1, 24, 72],
            metrics=["mae"],
        )
        fig = plot_seasonal_metrics(df, "temperature_2m", "mae")
        assert fig is not None
        plt.close(fig)

    def test_plot_monthly_metrics(self):
        """Should create monthly metrics plot."""
        df = _make_metrics_df(
            months=list(range(1, 13)),
            lead_hours=[1, 24],
            metrics=["mae"],
        )
        fig = plot_monthly_metrics(df, "temperature_2m", "mae")
        assert fig is not None
        plt.close(fig)

    def test_plot_seasonal_degradation(self):
        """Should create seasonal degradation plot."""
        df = _make_metrics_df(
            seasons=["DJF", "MAM", "JJA", "SON"],
            lead_hours=[1, 6, 12, 24, 48],
            metrics=["mae"],
        )
        fig = plot_seasonal_degradation(df, "temperature_2m")
        assert fig is not None
        plt.close(fig)


class TestDistributionPlots:
    """Test distribution plot generation."""

    def test_plot_error_histogram(self):
        """Should create error histogram."""
        errors = pd.Series(np.random.normal(0, 2.0, 200))
        fig = plot_error_histogram(errors, "temperature_2m")
        assert fig is not None
        plt.close(fig)

    def test_plot_error_histogram_empty(self):
        """Should return None for empty error series."""
        errors = pd.Series([], dtype=float)
        fig = plot_error_histogram(errors, "temperature_2m")
        assert fig is None

    def test_plot_error_boxplot(self):
        """Should create error boxplot."""
        df = _make_verification_df(n=500)
        fig = plot_error_boxplot(df, "temperature_2m")
        assert fig is not None
        plt.close(fig)

    def test_plot_error_percentiles(self):
        """Should create error percentile plot."""
        df = _make_verification_df(n=500)
        fig = plot_error_percentiles(df, "temperature_2m")
        assert fig is not None
        plt.close(fig)


class TestPrecipitationPlots:
    """Test precipitation plot generation."""

    def test_classify_events(self):
        """Should classify events correctly."""
        df = pd.DataFrame(
            {
                "forecast_precipitation": [5.0, 0.5, 5.0, 0.5],
                "observed_precipitation": [5.0, 0.5, 0.5, 5.0],
            }
        )

        result = classify_events(df, "precipitation", threshold=1.0)
        assert "event_category" in result.columns
        # Row 0: forecast=5, observed=5 → Hit
        # Row 1: forecast=0.5, observed=0.5 → CorrectRejection
        # Row 2: forecast=5, observed=0.5 → FalseAlarm
        # Row 3: forecast=0.5, observed=5 → Miss
        assert list(result["event_category"]) == ["Hit", "CorrectRejection", "FalseAlarm", "Miss"]

    def test_plot_hit_miss_by_threshold(self):
        """Should create hit/miss plot."""
        df = _make_verification_df(n=500)
        fig = plot_hit_miss_by_threshold(df, "precipitation", thresholds=[0.1, 1, 5, 10])
        assert fig is not None
        plt.close(fig)

    def test_plot_event_frequency(self):
        """Should create event frequency plot."""
        df = _make_verification_df(n=500)
        fig = plot_event_frequency(df, "precipitation", thresholds=[0.1, 1, 5, 10])
        assert fig is not None
        plt.close(fig)

    def test_plot_event_contingency(self):
        """Should create contingency table plot."""
        df = _make_verification_df(n=500)
        fig = plot_event_contingency(df, "precipitation", threshold=1.0)
        assert fig is not None
        plt.close(fig)
