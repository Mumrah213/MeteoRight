"""Regression tests for circular (angular) wind-direction error.

The bug these guard against: the live verification path computed
wind_direction_10m error as plain subtraction, so a 350 deg -> 10 deg miss was
recorded as a 340 deg error instead of the correct 20 deg shortest angle.
"""

import numpy as np
import pandas as pd

from util.circular import (
    CIRCULAR_VARIABLES,
    circular_error,
    circular_error_series,
    is_circular,
)


def test_scalar_shortest_angle():
    # forecast clockwise of observation across the 0/360 seam
    assert circular_error(10.0, 350.0) == 20.0
    assert circular_error(350.0, 10.0) == -20.0
    # plain interior difference is unchanged
    assert circular_error(100.0, 90.0) == 10.0
    # antipodal wraps to the -180 boundary, never +180
    assert circular_error(0.0, 180.0) == -180.0
    # always within [-180, 180)
    for f in range(0, 360, 7):
        for o in range(0, 360, 11):
            assert -180.0 <= circular_error(float(f), float(o)) < 180.0


def test_series_matches_scalar():
    fcst = pd.Series([10.0, 350.0, 100.0, 0.0])
    obs = pd.Series([350.0, 10.0, 90.0, 180.0])
    got = circular_error_series(fcst, obs)
    expected = pd.Series([circular_error(f, o) for f, o in zip(fcst, obs, strict=True)])
    pd.testing.assert_series_equal(got.reset_index(drop=True), expected, check_names=False)


def test_series_nan_propagates():
    got = circular_error_series(pd.Series([np.nan, 10.0]), pd.Series([350.0, np.nan]))
    assert pd.isna(got.iloc[0])
    assert pd.isna(got.iloc[1])


def test_registry():
    assert is_circular("wind_direction_10m")
    assert not is_circular("temperature_2m")
    assert "wind_direction_10m" in CIRCULAR_VARIABLES


def test_compute_error_columns_uses_circular():
    """The live verification/errors.py path must wrap wind direction."""
    from verification.errors import compute_error_columns

    df = pd.DataFrame(
        {
            "forecast_wind_direction_10m": [10.0, 350.0],
            "observed_wind_direction_10m": [350.0, 10.0],
            "forecast_temperature_2m": [5.0, 6.0],
            "observed_temperature_2m": [4.0, 4.0],
        }
    )
    out = compute_error_columns(df, ("wind_direction_10m", "temperature_2m"))
    # wind direction: shortest angle, magnitude 20 not 340
    assert out["wind_direction_10m_error"].abs().max() == 20.0
    # temperature: plain subtraction, untouched
    assert list(out["temperature_2m_error"]) == [1.0, 2.0]


def test_cmd_verify_inline_loop_uses_circular(tmp_path):
    """cmd_verify has its own inline error loop; it must wrap too.

    Build a minimal forecasts/ + observations/ tree and run cmd_verify, then
    assert the wind-direction error column is the short angle.
    """
    import argparse

    from cli import cmd_verify

    target = pd.Timestamp("2026-05-25T00:00:00", tz="UTC")
    fcst = pd.DataFrame(
        {
            "forecast_issue_time": [target],
            "forecast_target_time": [target],
            "lead_hours": [0],
            "model": ["ecmwf_ifs025"],
            "latitude": [55.5],
            "longitude": [12.5],
            "elevation": [0.0],
            "wind_direction_10m": [10.0],
        }
    )
    obs = pd.DataFrame(
        {
            "observation_time": [target],
            "latitude": [55.5],
            "longitude": [12.5],
            "elevation": [0.0],
            "wind_direction_10m": [350.0],
        }
    )
    fdir = tmp_path / "forecasts" / "model=ecmwf_ifs025" / "year=2026" / "month=05"
    odir = tmp_path / "observations" / "year=2026" / "month=05"
    fdir.mkdir(parents=True)
    odir.mkdir(parents=True)
    fcst.to_parquet(fdir / "data.parquet")
    obs.to_parquet(odir / "data.parquet")

    out_dir = tmp_path / "verification"
    args = argparse.Namespace(
        data_dir=str(tmp_path),
        variables="wind_direction_10m",
        output=str(out_dir),
    )
    assert cmd_verify(args) == 0

    result = pd.read_parquet(out_dir / "verification.parquet")
    assert result["wind_direction_10m_error"].abs().max() == 20.0


def test_error_computation_reexports_shared_util():
    """Stack B must use the shared definition (no third copy that can drift)."""
    from forecast.verification import error_computation as ec

    assert ec.CIRCULAR_VARIABLES is CIRCULAR_VARIABLES
    assert ec._circular_error is circular_error
