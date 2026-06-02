"""Tests for the verification dataset building module.

Tests:
- Verification construction produces correct output
- Error columns are computed correctly
- Derived time dimensions (month, season) are added
- Merged dataset has expected columns
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from datasets.build_verification import SEASON_MAP, build_verification
from datasets.progress import ProgressReporter


def _create_fake_forecasts(
    start_date: str = "2024-01-01",
    n_days: int = 5,
    variables: list | None = None,
) -> pd.DataFrame:
    """Create fake forecast DataFrame for testing."""
    if variables is None:
        variables = ["temperature_2m", "precipitation"]

    records = []
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    for day in range(n_days):
        target_time = base_date + timedelta(days=day)
        for lead in [0, 6, 12, 24]:
            record = {
                "forecast_issue_time": target_time - timedelta(hours=lead),
                "forecast_target_time": target_time + timedelta(hours=lead),
                "lead_hours": lead,
                "model": "icon_eu",
                "latitude": 55.605,
                "longitude": 12.574,
                "elevation": 10.0,
            }
            for var in variables:
                record[var] = float(day * 10 + lead)
            records.append(record)

    return pd.DataFrame(records)


def _create_fake_observations(
    start_date: str = "2024-01-01",
    n_days: int = 5,
    variables: list | None = None,
) -> pd.DataFrame:
    """Create fake observation DataFrame for testing."""
    if variables is None:
        variables = ["temperature_2m", "precipitation"]

    records = []
    base_date = datetime.strptime(start_date, "%Y-%m-%d")
    for day in range(n_days):
        obs_time = base_date + timedelta(days=day)
        record = {
            "observation_time": obs_time,
            "latitude": 55.605,
            "longitude": 12.574,
            "elevation": 10.0,
        }
        for var in variables:
            if var == "temperature_2m":
                record[var] = float(day * 10 + 0.5)  # Slightly offset from forecast
            elif var == "precipitation":
                record[var] = float(day * 5)
            else:
                record[var] = 0.0
        records.append(record)

    return pd.DataFrame(records)


class TestBuildVerification:
    def test_basic_verification(self):
        """Test basic verification construction."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_dir = tmp / "raw"
            data_dir.mkdir(parents=True)

            # Create fake forecast and observation files
            # Forecast structure: model=X/year=Y/month=M/data.parquet
            fcst_subdir = data_dir / "forecasts" / "model=icon_eu" / "year=2024" / "month=1"
            fcst_subdir.mkdir(parents=True)
            fcst = _create_fake_forecasts()
            fcst.to_parquet(fcst_subdir / "data.parquet")

            # Observation structure: year=Y/month=M/data.parquet
            obs_subdir = data_dir / "observations" / "year=2024" / "month=1"
            obs_subdir.mkdir(parents=True)
            obs = _create_fake_observations()
            obs.to_parquet(obs_subdir / "data.parquet")

            progress = ProgressReporter()
            result = build_verification(
                data_dir=data_dir,
                verification_dir=tmp / "verification",
                variables=("temperature_2m", "precipitation"),
                progress=progress,
            )

            assert "verification_rows" in result
            assert result["verification_rows"] > 0

            # Check output file exists
            output_path = tmp / "verification" / "verification.parquet"
            assert output_path.exists()

            # Check output has expected columns
            df = pd.read_parquet(output_path)
            assert "forecast_temperature_2m" in df.columns
            assert "observed_temperature_2m" in df.columns
            assert "temperature_2m_error" in df.columns
            assert "month" in df.columns
            assert "season" in df.columns

    def test_error_computation(self):
        """Test that error columns are forecast - observed."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_dir = tmp / "raw"
            data_dir.mkdir(parents=True)

            fcst_subdir = data_dir / "forecasts" / "model=icon_eu" / "year=2024" / "month=1"
            fcst_subdir.mkdir(parents=True)
            fcst = pd.DataFrame(
                [
                    {
                        "forecast_issue_time": datetime(2024, 1, 1),
                        "forecast_target_time": datetime(2024, 1, 2),
                        "lead_hours": 24,
                        "model": "icon_eu",
                        "latitude": 55.6,
                        "longitude": 12.6,
                        "elevation": 10.0,
                        "temperature_2m": 15.0,
                    },
                ]
            )
            fcst.to_parquet(fcst_subdir / "data.parquet")

            obs_subdir = data_dir / "observations" / "year=2024" / "month=1"
            obs_subdir.mkdir(parents=True)
            obs = pd.DataFrame(
                [
                    {
                        "observation_time": datetime(2024, 1, 2),
                        "latitude": 55.6,
                        "longitude": 12.6,
                        "elevation": 10.0,
                        "temperature_2m": 13.0,
                    },
                ]
            )
            obs.to_parquet(obs_subdir / "data.parquet")

            build_verification(
                data_dir=data_dir,
                verification_dir=tmp / "verification",
                variables=("temperature_2m",),
            )

            df = pd.read_parquet(tmp / "verification" / "verification.parquet")
            error = df["temperature_2m_error"].iloc[0]
            assert abs(error - 2.0) < 0.01  # forecast (15) - observed (13)

    def test_rejects_wrong_location(self):
        """Test that verification refuses time-matched data from another city."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_dir = tmp / "raw"
            data_dir.mkdir(parents=True)

            fcst_subdir = data_dir / "forecasts" / "model=icon_eu" / "year=2024" / "month=1"
            fcst_subdir.mkdir(parents=True)
            fcst = pd.DataFrame(
                [
                    {
                        "forecast_issue_time": datetime(2024, 1, 1),
                        "forecast_target_time": datetime(2024, 1, 2),
                        "lead_hours": 24,
                        "model": "icon_eu",
                        "latitude": 51.5,
                        "longitude": -0.125,
                        "elevation": 25.0,
                        "temperature_2m": 15.0,
                    },
                ]
            )
            fcst.to_parquet(fcst_subdir / "data.parquet")

            obs_subdir = data_dir / "observations" / "year=2024" / "month=1"
            obs_subdir.mkdir(parents=True)
            obs = pd.DataFrame(
                [
                    {
                        "observation_time": datetime(2024, 1, 2),
                        "latitude": 55.641476,
                        "longitude": 12.596349,
                        "elevation": 10.0,
                        "temperature_2m": 13.0,
                    },
                ]
            )
            obs.to_parquet(obs_subdir / "data.parquet")

            try:
                build_verification(
                    data_dir=data_dir,
                    verification_dir=tmp / "verification",
                    variables=("temperature_2m",),
                )
            except ValueError as exc:
                assert "different locations" in str(exc)
            else:
                raise AssertionError("Expected wrong-location verification build to fail")

    def test_season_mapping(self):
        """Test that season mapping is correct for all months."""
        assert SEASON_MAP[12] == "DJF"
        assert SEASON_MAP[1] == "DJF"
        assert SEASON_MAP[2] == "DJF"
        assert SEASON_MAP[3] == "MAM"
        assert SEASON_MAP[4] == "MAM"
        assert SEASON_MAP[5] == "MAM"
        assert SEASON_MAP[6] == "JJA"
        assert SEASON_MAP[7] == "JJA"
        assert SEASON_MAP[8] == "JJA"
        assert SEASON_MAP[9] == "SON"
        assert SEASON_MAP[10] == "SON"
        assert SEASON_MAP[11] == "SON"
