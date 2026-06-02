"""Tests for grouping semantics.

Verifies that metrics grouping by lead_hours, model, season, etc.
produces expected results.
"""

import pandas as pd

from plots.seasonal import SEASON_ORDER


def _make_metrics_df(
    n_rows: int = 100,
    variables: list[str] | None = None,
    metrics: list[str] | None = None,
    models: list[str] | None = None,
    lead_hours: list[int] | None = None,
    seasons: list[str] | None = None,
    months: list[int] | None = None,
) -> pd.DataFrame:
    """Create a synthetic metrics DataFrame for testing."""
    if variables is None:
        variables = ["temperature_2m"]
    if metrics is None:
        metrics = ["mae", "rmse", "bias"]
    if models is None:
        models = ["icon_eu", "gfs"]
    if lead_hours is None:
        lead_hours = [1, 6, 12, 24, 48, 72]
    if seasons is None:
        seasons = SEASON_ORDER
    if months is None:
        months = list(range(1, 13))

    rows = []
    for var in variables:
        for metric in metrics:
            for model in models:
                for lh in lead_hours:
                    for season in seasons:
                        for month in months:
                            # Deterministic values based on parameters
                            base_error = lh * 0.1
                            if var == "precipitation":
                                base_error *= 2.0
                            noise = (hash((var, metric, model, lh, season, month)) % 100) / 1000.0

                            rows.append(
                                {
                                    "lead_hours": lh,
                                    "model": model,
                                    "season": season,
                                    "month": month,
                                    "variable": var,
                                    "metric_name": metric,
                                    "metric_value": base_error + noise,
                                    "sample_size": 100 + (hash((var, lh, model)) % 50),
                                    "total_rows": 110,
                                    "missing_count": 10,
                                }
                            )

    return pd.DataFrame(rows)


class TestGroupingSemantics:
    """Test that filtering and grouping produce expected results."""

    def test_filter_by_variable(self):
        """Filtering by variable should return only that variable."""
        df = _make_metrics_df(variables=["temperature_2m", "precipitation"])

        temp_df = df[df["variable"] == "temperature_2m"]
        assert all(temp_df["variable"] == "temperature_2m")
        assert len(temp_df) > 0

        precip_df = df[df["variable"] == "precipitation"]
        assert all(precip_df["variable"] == "precipitation")
        assert len(precip_df) > 0

    def test_filter_by_metric(self):
        """Filtering by metric should return only that metric."""
        df = _make_metrics_df(metrics=["mae", "rmse"])

        mae_df = df[df["metric_name"] == "mae"]
        assert all(mae_df["metric_name"] == "mae")

        rmse_df = df[df["metric_name"] == "rmse"]
        assert all(rmse_df["metric_name"] == "rmse")

    def test_filter_by_model(self):
        """Filtering by model should return only that model."""
        df = _make_metrics_df(models=["icon_eu", "gfs"])

        icon_df = df[df["model"] == "icon_eu"]
        assert all(icon_df["model"] == "icon_eu")

        gfs_df = df[df["model"] == "gfs"]
        assert all(gfs_df["model"] == "gfs")

    def test_multi_filter(self):
        """Multiple filters should combine with AND."""
        df = _make_metrics_df(
            variables=["temperature_2m"],
            metrics=["mae"],
            models=["icon_eu"],
        )

        filtered = df[
            (df["variable"] == "temperature_2m")
            & (df["metric_name"] == "mae")
            & (df["model"] == "icon_eu")
        ]

        assert len(filtered) > 0
        assert all(filtered["variable"] == "temperature_2m")
        assert all(filtered["metric_name"] == "mae")
        assert all(filtered["model"] == "icon_eu")

    def test_groupby_lead_hours(self):
        """Grouping by lead_hours should produce unique lead times."""
        df = _make_metrics_df(lead_hours=[1, 6, 12, 24])

        grouped = df.groupby("lead_hours")
        lead_times = list(grouped.groups.keys())

        assert set(lead_times) == {1, 6, 12, 24}

    def test_groupby_season(self):
        """Grouping by season should produce expected seasons."""
        df = _make_metrics_df(seasons=["DJF", "MAM", "JJA", "SON"])

        grouped = df.groupby("season")
        seasons = list(grouped.groups.keys())

        assert set(seasons) == {"DJF", "MAM", "JJA", "SON"}

    def test_empty_filter(self):
        """Filtering with no matches should return empty DataFrame."""
        df = _make_metrics_df()

        filtered = df[df["variable"] == "nonexistent"]
        assert len(filtered) == 0

    def test_groupby_preserves_order(self):
        """Groupby with sort=True should produce sorted groups."""
        df = _make_metrics_df(lead_hours=[72, 24, 1, 48])

        grouped = df.groupby("lead_hours", sort=True)
        lead_times = list(grouped.groups.keys())
        assert lead_times == sorted(lead_times)
