"""End-to-end functional test with mock data.

Tests the full pipeline: load → align → verify → error → validate → save.
"""


import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from verification.aligner import align_forecasts_with_observations
from verification.errors import compute_error_columns
from verification.loaders import load_forecasts, load_observations
from verification.provenance import verify_provenance_integrity
from verification.schema import canonical_column_order
from verification.validator import VerificationValidator, save_validation_report

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    """Create mock forecast and observation parquet files."""
    data_dir = tmp_path / "data"
    forecasts_dir = data_dir / "forecasts" / "model=ecmwf_ifs_hres" / "year=2025" / "month=01"
    observations_dir = data_dir / "observations" / "year=2025" / "month=01"

    forecasts_dir.mkdir(parents=True)
    observations_dir.mkdir(parents=True)

    # --- Mock forecasts ---
    # Three issue times, each targeting future hours with valid positive leads
    # All targets stay within Jan 15 (loader filters by target_time date range)
    base_time = datetime(2025, 1, 15, 0, 0, tzinfo=UTC)
    rows = []
    # 00Z issue → targets at 06Z, 12Z, 18Z (leads 6h, 12h, 18h)
    # 06Z issue → targets at 12Z, 18Z (leads 6h, 12h)
    # 12Z issue → targets at 18Z (lead 6h)
    issue_targets = {
        0: [6, 12, 18],
        6: [12, 18],
        12: [18],
    }
    for issue_offset, target_offsets in issue_targets.items():
        issue_time = base_time + timedelta(hours=issue_offset)
        for target_offset in target_offsets:
            target_time = base_time + timedelta(hours=target_offset)
            lead = (target_time - issue_time).total_seconds() // 3600
            rows.append(
                {
                    "forecast_issue_time": issue_time,
                    "forecast_target_time": target_time,
                    "lead_hours": int(lead),
                    "model": "ecmwf_ifs_hres",
                    "latitude": 55.605,
                    "longitude": 13.003,
                    "elevation": 50.0,
                    "temperature_2m": 5.0 + target_offset * 0.1,
                    "precipitation": 0.5 + target_offset * 0.1,
                    "year": 2025,
                    "month": 1,
                }
            )

    # --- Mock observations ---
    # Only 2 of the 3 target times on Jan 15 have observations (missing 18Z)
    obs_rows = []
    for target_offset in [6, 12]:  # 18Z is missing!
        obs_time = base_time + timedelta(hours=target_offset)
        obs_rows.append(
            {
                "observation_time": obs_time,
                "latitude": 55.605,
                "longitude": 13.003,
                "elevation": 50.0,
                "temperature_2m": 5.5 + target_offset * 0.1,  # slightly different from forecast
                "precipitation": 0.3 + target_offset * 0.1,
                "year": 2025,
                "month": 1,
            }
        )

    forecast_df = pd.DataFrame(rows)
    for col in ["forecast_issue_time", "forecast_target_time"]:
        forecast_df[col] = pd.to_datetime(forecast_df[col], utc=True)
    forecast_df["lead_hours"] = forecast_df["lead_hours"].astype("int32")
    forecast_df.to_parquet(forecasts_dir / "data.parquet", index=False)

    obs_df = pd.DataFrame(obs_rows)
    obs_df["observation_time"] = pd.to_datetime(obs_df["observation_time"], utc=True)
    obs_df.to_parquet(observations_dir / "data.parquet", index=False)

    return data_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_forecasts(mock_data_dir: Path):
    """Test forecast loading with partition awareness and date filtering."""
    df = load_forecasts(mock_data_dir, "ecmwf_ifs_hres", "2025-01-15", "2025-01-15")
    assert len(df) == 6  # 3 issue times × variable targets (all within Jan 15)
    assert df["forecast_issue_time"].dt.tz is not None
    assert df["forecast_target_time"].dt.tz is not None
    assert "temperature_2m" in df.columns
    assert "precipitation" in df.columns


def test_load_observations(mock_data_dir: Path):
    """Test observation loading with partition awareness and date filtering."""
    df = load_observations(mock_data_dir, "2025-01-15", "2025-01-15")
    assert len(df) == 2  # only 2 target times have observations
    assert df["observation_time"].dt.tz is not None


def test_align_left_join(mock_data_dir: Path):
    """Test LEFT JOIN preserves all forecast rows."""
    forecasts = load_forecasts(mock_data_dir, "ecmwf_ifs_hres", "2025-01-15", "2025-01-15")
    observations = load_observations(mock_data_dir, "2025-01-15", "2025-01-15")

    aligned = align_forecasts_with_observations(
        forecasts, observations, ("temperature_2m", "precipitation")
    )

    # All 6 forecast rows should survive (LEFT JOIN)
    assert len(aligned) == 6

    # Both timestamps preserved
    assert "forecast_target_time" in aligned.columns
    assert "observation_time" in aligned.columns

    # 3 rows should have matching observations (00Z@06Z, 00Z@12Z, 06Z@12Z)
    matched = aligned["observation_time"].notna().sum()
    assert matched == 3

    # 3 rows should have NaN observations (00Z@18Z, 06Z@18Z, 12Z@18Z)
    missing = aligned["observation_time"].isna().sum()
    assert missing == 3


def test_provenance_preserved(mock_data_dir: Path):
    """Test that provenance invariant holds after alignment."""
    forecasts = load_forecasts(mock_data_dir, "ecmwf_ifs_hres", "2025-01-15", "2025-01-15")
    observations = load_observations(mock_data_dir, "2025-01-15", "2025-01-15")

    aligned = align_forecasts_with_observations(
        forecasts, observations, ("temperature_2m", "precipitation")
    )

    # Should not raise — provenance is intact
    verify_provenance_integrity(aligned)


def test_error_computation(mock_data_dir: Path):
    """Test error = forecast - observed computation."""
    forecasts = load_forecasts(mock_data_dir, "ecmwf_ifs_hres", "2025-01-15", "2025-01-15")
    observations = load_observations(mock_data_dir, "2025-01-15", "2025-01-15")

    aligned = align_forecasts_with_observations(
        forecasts, observations, ("temperature_2m", "precipitation")
    )
    with_errors = compute_error_columns(aligned, ("temperature_2m", "precipitation"))

    # Error columns exist
    assert "temperature_2m_error" in with_errors.columns
    assert "precipitation_error" in with_errors.columns

    # Matched rows have finite errors
    matched = with_errors[with_errors["observation_time"].notna()]
    assert matched["temperature_2m_error"].notna().all()
    assert matched["precipitation_error"].notna().all()

    # Unmatched rows have NaN errors
    unmatched = with_errors[with_errors["observation_time"].isna()]
    assert unmatched["temperature_2m_error"].isna().all()
    assert unmatched["precipitation_error"].isna().all()


def test_validation_detects_missing_observations(mock_data_dir: Path):
    """Test that validator detects missing observations from LEFT JOIN."""
    forecasts = load_forecasts(mock_data_dir, "ecmwf_ifs_hres", "2025-01-15", "2025-01-15")
    observations = load_observations(mock_data_dir, "2025-01-15", "2025-01-15")

    aligned = align_forecasts_with_observations(
        forecasts, observations, ("temperature_2m", "precipitation")
    )
    aligned = compute_error_columns(aligned, ("temperature_2m", "precipitation"))

    validator = VerificationValidator(("temperature_2m", "precipitation"))
    issues = validator.validate(aligned)

    # Should have at least one warning about missing observations
    missing_issues = [i for i in issues if i.check == "missing_observations"]
    assert len(missing_issues) == 1
    assert missing_issues[0].count == 3  # 3 forecast rows with no observation (LEFT JOIN result)


def test_canonical_column_order(mock_data_dir: Path):
    """Test that output has correct canonical column order after CLI reordering."""
    forecasts = load_forecasts(mock_data_dir, "ecmwf_ifs_hres", "2025-01-15", "2025-01-15")
    observations = load_observations(mock_data_dir, "2025-01-15", "2025-01-15")

    aligned = align_forecasts_with_observations(
        forecasts, observations, ("temperature_2m", "precipitation")
    )
    aligned = compute_error_columns(aligned, ("temperature_2m", "precipitation"))

    # CLI Step 5 reorders to canonical order
    expected = canonical_column_order(("temperature_2m", "precipitation"))
    aligned = aligned[expected]
    assert list(aligned.columns) == expected


def test_save_and_load(mock_data_dir: Path, tmp_path: Path):
    """Test saving verification dataset and loading it back."""
    forecasts = load_forecasts(mock_data_dir, "ecmwf_ifs_hres", "2025-01-15", "2025-01-15")
    observations = load_observations(mock_data_dir, "2025-01-15", "2025-01-15")

    aligned = align_forecasts_with_observations(
        forecasts, observations, ("temperature_2m", "precipitation")
    )
    aligned = compute_error_columns(aligned, ("temperature_2m", "precipitation"))

    output_path = tmp_path / "verification_2025-01-15_to_2025-01-15.parquet"
    aligned.to_parquet(output_path, engine="pyarrow", index=False)

    # Load back and verify
    loaded = pd.read_parquet(output_path)
    assert len(loaded) == len(aligned)
    assert list(loaded.columns) == list(aligned.columns)
    assert loaded["forecast_issue_time"].dt.tz is not None


def test_validation_report(mock_data_dir: Path, tmp_path: Path):
    """Test validation report JSON output."""
    forecasts = load_forecasts(mock_data_dir, "ecmwf_ifs_hres", "2025-01-15", "2025-01-15")
    observations = load_observations(mock_data_dir, "2025-01-15", "2025-01-15")

    aligned = align_forecasts_with_observations(
        forecasts, observations, ("temperature_2m", "precipitation")
    )
    aligned = compute_error_columns(aligned, ("temperature_2m", "precipitation"))

    validator = VerificationValidator(("temperature_2m", "precipitation"))
    issues = validator.validate(aligned)

    report_path = tmp_path / "validation_report.json"
    save_validation_report(issues, report_path)

    with open(report_path) as f:
        report = json.load(f)

    assert report["total_issues"] > 0
    assert "errors" in report
    assert "warnings" in report
    assert "info" in report
    assert len(report["issues"]) == len(issues)


def test_cli_main(mock_data_dir: Path, tmp_path: Path, capsys):
    """Test CLI end-to-end with mock data."""
    output_dir = tmp_path / "output"
    argv = [
        "--data-dir",
        str(mock_data_dir),
        "--model",
        "ecmwf_ifs_hres",
        "--location",
        "55.605,13.003",
        "--start",
        "2025-01-15",
        "--end",
        "2025-01-15",
        "--variables",
        "temperature_2m,precipitation",
        "--output",
        str(output_dir),
    ]

    from verification.cli import main

    main(argv)

    captured = capsys.readouterr()
    assert "Verification dataset built successfully" in captured.out

    # Verify output files exist
    parquet_files = list(output_dir.glob("verification_*.parquet"))
    assert len(parquet_files) == 1

    report_files = list(output_dir.glob("validation_report.json"))
    assert len(report_files) == 1
