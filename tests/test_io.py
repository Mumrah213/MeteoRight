"""Tests for I/O layer — essential cases only."""

import pandas as pd

from metrics.io import write_metrics


def test_write_parquet_with_metadata(tmp_path):
    """Metrics are written as parquet with metadata."""
    df = pd.DataFrame(
        {
            "lead_hours": [1, 6, 24],
            "variable": ["temperature_2m"] * 3,
            "metric_name": ["mae", "rmse", "bias"],
            "metric_value": [1.0, 1.5, 0.5],
            "sample_size": [100] * 3,
            "total_rows": [100] * 3,
            "missing_count": [0] * 3,
        }
    )

    meta = {"groupby": ["lead_hours"], "metrics": ["mae", "rmse", "bias"]}
    output = tmp_path / "metrics" / "by_lead_time" / "temperature_2m.parquet"
    result = write_metrics(df, output, metadata=meta)

    assert result.exists()
    read_back = pd.read_parquet(result)
    assert len(read_back) == len(df)
    assert read_back["metric_value"].tolist() == df["metric_value"].tolist()


def test_round_trip(tmp_path):
    """Write then read produces equivalent data."""
    df = pd.DataFrame(
        {
            "lead_hours": [1, 6, 24],
            "variable": ["temperature_2m"] * 3,
            "metric_name": ["mae", "rmse", "bias"],
            "metric_value": [1.0, 1.5, 0.5],
            "sample_size": [100, 100, 100],
            "total_rows": [100, 100, 100],
            "missing_count": [0, 0, 0],
        }
    )

    output = tmp_path / "test.parquet"
    write_metrics(df, output)
    read_back = pd.read_parquet(output)

    assert read_back["metric_value"].tolist() == df["metric_value"].tolist()
    assert read_back["sample_size"].tolist() == df["sample_size"].tolist()
