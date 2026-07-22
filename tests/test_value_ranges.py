"""Tests for physical value-range validation.

Values outside PLAUSIBLE_RANGES normally mean a unit mismatch, an unmasked
sentinel (-999) or a corrupt row rather than genuinely extreme weather.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from verification.validator import PLAUSIBLE_RANGES, VerificationValidator


def _verification_df(n: int = 10) -> pd.DataFrame:
    """Build a minimal canonical verification frame with in-range values."""
    return pd.DataFrame(
        {
            "forecast_issue_time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "forecast_target_time": pd.date_range("2024-01-02", periods=n, freq="h"),
            "observation_time": pd.date_range("2024-01-02", periods=n, freq="h"),
            "lead_hours": range(n),
            "model": "icon_eu",
            "latitude": 55.6,
            "longitude": 12.5,
            "forecast_temperature_2m": np.linspace(0.0, 10.0, n),
            "observed_temperature_2m": np.linspace(0.0, 10.0, n),
        }
    )


def _range_issues(df: pd.DataFrame, variables: tuple[str, ...] = ("temperature_2m",)):
    validator = VerificationValidator(variables=variables)
    return [i for i in validator.validate(df) if i.check == "value_ranges"]


def test_in_range_values_produce_info():
    """A clean frame reports info severity and a zero count."""
    issues = _range_issues(_verification_df())
    assert len(issues) == 1
    assert issues[0].severity == "info"
    assert issues[0].count == 0


def test_sentinel_value_flagged():
    """An unmasked -999 sentinel is reported as an error."""
    df = _verification_df()
    df.loc[0, "observed_temperature_2m"] = -999.0

    issues = _range_issues(df)
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert issues[0].count == 1
    assert "observed_temperature_2m" in issues[0].message


def test_unit_error_flagged():
    """Kelvin values in a Celsius column are flagged on every row."""
    df = _verification_df()
    df["forecast_temperature_2m"] = np.linspace(273.0, 283.0, len(df))

    issues = _range_issues(df)
    assert issues[0].severity == "error"
    assert issues[0].count == len(df)


def test_nan_values_ignored():
    """Missing values are not range violations — another check owns them."""
    df = _verification_df()
    df.loc[0, "observed_temperature_2m"] = np.nan

    issues = _range_issues(df)
    assert issues[0].severity == "info"
    assert issues[0].count == 0


def test_unknown_variable_skipped():
    """Variables without a known range produce no issues at all."""
    assert _range_issues(_verification_df(), variables=("some_unknown_var",)) == []


def test_both_bounds_checked():
    """Violations below the minimum and above the maximum are both counted."""
    df = _verification_df()
    df.loc[0, "observed_temperature_2m"] = -100.0
    df.loc[1, "observed_temperature_2m"] = 100.0

    issues = _range_issues(df)
    assert issues[0].count == 2


@pytest.mark.parametrize("variable", sorted(PLAUSIBLE_RANGES))
def test_ranges_are_well_formed(variable: str):
    """Every declared range is a finite, correctly ordered interval."""
    vmin, vmax = PLAUSIBLE_RANGES[variable]
    assert np.isfinite(vmin) and np.isfinite(vmax)
    assert vmin < vmax
