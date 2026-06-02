"""Tests for CLI argument parsing — essential cases only."""

from metrics.cli import parse_args


def test_required_args():
    """Required arguments are parsed correctly."""
    args = parse_args(
        [
            "--verification", "data/verification/*.parquet",
            "--group-by", "lead_hours",
            "--metrics", "mae,rmse",
            "--variables", "temperature_2m",
            "--output", "data/metrics",
        ]
    )
    assert str(args.verification) == "data/verification/*.parquet"
    assert args.group_by == ["lead_hours"]
    assert args.metrics == ["mae", "rmse"]
    assert args.variables == ["temperature_2m"]
    assert str(args.output) == "data/metrics"


def test_glob_pattern():
    """Glob patterns are preserved as-is."""
    args = parse_args(
        [
            "--verification", "data/verification/verification_2025_*.parquet",
            "--group-by", "lead_hours",
            "--metrics", "mae",
            "--variables", "temperature_2m",
            "--output", "data/metrics",
        ]
    )
    assert "2025_*" in args.verification


def test_single_file():
    """Single file path works."""
    args = parse_args(
        [
            "--verification", "data/verification/verification_2025.parquet",
            "--group-by", "lead_hours",
            "--metrics", "mae",
            "--variables", "temperature_2m",
            "--output", "data/metrics",
        ]
    )
    assert args.verification == "data/verification/verification_2025.parquet"


def test_defaults():
    """Default values for optional arguments."""
    args = parse_args(
        [
            "--verification", "data/verification/*.parquet",
            "--group-by", "lead_hours",
            "--output", "data/metrics",
        ]
    )
    assert args.metrics == ["mae", "rmse", "bias"]
    assert args.variables == ["temperature_2m", "precipitation"]
