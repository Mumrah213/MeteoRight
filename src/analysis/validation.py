"""Data quality validation for weather analysis.

Provides quick sanity checks on loaded DataFrames to catch common issues
during exploration: missing values, out-of-range values, time gaps, etc.

Usage
-----
>>> from src.analysis import validation as v
>>> obs = load_historical("temperature_2m", start="2024-01")
>>> checks = v.check_data_quality(obs)
>>> for name, result in checks.items():
...     print(f"{name}: {result}")
...
all_valid: False
missing_values: 123
out_of_range: 0
time_gaps: 3
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Weather variable ranges ────────────────────────────────────────────────

# Reasonable ranges for common weather variables [min, max] in SI units
WEATHER_RANGES: dict[str, tuple[float, float]] = {
    # Temperature (Kelvin) — approx -60°C to +60°C
    "temperature_2m": (-60.0, 60.0),
    "temperature_100m": (-60.0, 70.0),
    "temperature_1200m": (-80.0, 80.0),
    "temperature_1800m": (-80.0, 80.0),
    "temperature_80m": (-60.0, 60.0),
    # Precipitation (mm/h) — always non-negative, max ~100 mm/h
    "precipitation": (0.0, 100.0),
    "precipitation_total": (0.0, 1000.0),
    # Wind speed (m/s)
    "wind_speed_10m": (0.0, 150.0),
    "wind_speed_100m": (0.0, 150.0),
    "wind_direction_10m": (0.0, 360.0),
    "wind_direction_100m": (0.0, 360.0),
    # Pressure (Pa or hPa — be liberal)
    "surface_pressure": (80000.0, 110000.0),
    "sea_level_pressure": (87000.0, 109000.0),
    # Other
    "cloud_coverage": (0.0, 100.0),
    "relative_humidity_2m": (0.0, 100.0),
    "dew_point_2m": (-60.0, 60.0),
    # Snow
    "snow_depth": (0.0, 10.0),
    "snowfall": (0.0, 100.0),
    # Radiation
    "direct_radiation": (0.0, 1500.0),
    "diffuse_radiation": (0.0, 1500.0),
}


# ── Quality checks ──────────────────────────────────────────────────────────


def check_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Run all data quality checks on a DataFrame.

    Returns a dict of check results suitable for quick inspection in notebooks.

    Parameters
    ----------
    df : DataFrame
        Loaded data from load_historical or load_forecast.

    Returns
    -------
    Dict with keys:
        all_valid: bool
        missing_values: int
        out_of_range: int
        time_gaps: int (number of gaps found)
        duplicate_times: int
        sample_size: int
    """
    results: dict[str, Any] = {
        "all_valid": True,
        "missing_values": 0,
        "out_of_range": 0,
        "time_gaps": 0,
        "duplicate_times": 0,
        "sample_size": len(df),
    }

    if df.empty:
        results["all_valid"] = False
        return results

    # ── Missing values ─────────────────────────────────────────────────
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            results["missing_values"] += int(n_missing)

    # ── Duplicate timestamps ───────────────────────────────────────────
    for ts_col in ["observation_time", "forecast_target_time"]:
        if ts_col in df.columns:
            n_dups = df.duplicated(subset=[ts_col]).sum()
            results["duplicate_times"] += int(n_dups)

    # ── Out-of-range values ────────────────────────────────────────────
    if "variable" in df.columns and "value" in df.columns:
        oor_count = 0
        for _, row in df.iterrows():
            var = row["variable"]
            val = row["value"]
            if pd.isna(val):
                continue
            if var in WEATHER_RANGES:
                vmin, vmax = WEATHER_RANGES[var]
                if not (vmin <= val <= vmax):
                    oor_count += 1
        results["out_of_range"] = oor_count

    # ── Time gaps ──────────────────────────────────────────────────────
    for ts_col in ["observation_time", "forecast_target_time"]:
        if ts_col in df.columns:
            times = pd.to_datetime(df[ts_col]).dropna().sort_values()
            if len(times) > 1:
                # Expect hourly data (3600s = 1h)
                diffs = times.diff().dt.total_seconds().dropna()
                expected_interval = 3600.0  # 1 hour
                tolerance = 0.5 * expected_interval  # 30 min tolerance

                gaps = diffs[(diffs - expected_interval).abs() > tolerance]
                results["time_gaps"] += len(gaps)

    # ── Overall validity ───────────────────────────────────────────────
    results["all_valid"] = (
        results["missing_values"] == 0
        and results["out_of_range"] == 0
        and results["time_gaps"] == 0
        and results["duplicate_times"] == 0
    )

    return results


# ── Aligned data validation ─────────────────────────────────────────────────


def check_alignment_quality(aligned_df: pd.DataFrame) -> dict[str, Any]:
    """Validate alignment results.

    Checks for common alignment issues: zero-match rate, missing observations,
    unexpected NaN patterns.

    Parameters
    ----------
    aligned_df : DataFrame
        From align().

    Returns
    -------
    Dict with keys:
        total_rows: int
        matched_rows: int
        match_rate: float (0-1)
        forecast_only_rows: int
        observation_only_rows: int
        has_forecast_value: bool
        has_observation_value: bool
    """
    if aligned_df.empty:
        return {
            "total_rows": 0,
            "matched_rows": 0,
            "match_rate": 0.0,
            "forecast_only_rows": 0,
            "observation_only_rows": 0,
            "has_forecast_value": False,
            "has_observation_value": False,
        }

    total = len(aligned_df)
    matched = len(aligned_df[aligned_df["match_status"] == "matched"])
    forecast_only = len(aligned_df[aligned_df["match_status"] == "forecast_only"])
    obs_only = len(aligned_df[aligned_df["match_status"] == "observation_only"])

    has_fcst = "forecast_value" in aligned_df.columns
    has_obs = "observation_value" in aligned_df.columns

    return {
        "total_rows": total,
        "matched_rows": matched,
        "match_rate": matched / max(total, 1),
        "forecast_only_rows": forecast_only,
        "observation_only_rows": obs_only,
        "has_forecast_value": has_fcst,
        "has_observation_value": has_obs,
    }


# ── Schema validation ───────────────────────────────────────────────────────


def validate_schema(df: pd.DataFrame, expected_type: str = "observations") -> list[str]:
    """Validate DataFrame schema against expected format.

    Checks that loaded DataFrames have the expected columns for their type.

    Parameters
    ----------
    df : DataFrame
        Loaded data.
    expected_type : str
        Either "observations" or "forecasts".

    Returns
    -------
    List of warning messages (empty if schema is valid).
    """
    warnings: list[str] = []

    if df.empty:
        return ["DataFrame is empty"]

    if expected_type == "observations":
        required = ["observation_time", "latitude", "longitude", "variable", "value"]
    elif expected_type == "forecasts":
        required = [
            "forecast_issue_time",
            "forecast_target_time",
            "lead_hours",
            "model",
            "latitude",
            "longitude",
            "variable",
            "value",
        ]
    else:
        return [f"Unknown expected_type: {expected_type}"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(f"Missing columns: {', '.join(missing)}")

    # Check types
    for col in ["latitude", "longitude", "value"]:
        if col in df.columns:
            if not np.issubdtype(df[col].dropna().dtype, np.number):
                warnings.append(f"Column '{col}' is not numeric: {df[col].dtype}")

    return warnings


# ── Convenience: comprehensive report ───────────────────────────────────────


def quality_report(
    df: pd.DataFrame,
    aligned_df: pd.DataFrame | None = None,
    expected_type: str = "observations",
) -> str:
    """Generate a text quality report suitable for notebook output.

    Parameters
    ----------
    df : DataFrame
        Loaded data (observations or forecasts).
    aligned_df : DataFrame or None
        Aligned data, if available.
    expected_type : str
        "observations" or "forecasts".

    Returns
    -------
    Formatted string report.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("WEATHER DATA QUALITY REPORT")
    lines.append("=" * 60)

    # Basic info
    lines.append(f"\nType: {expected_type}")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {', '.join(df.columns)}")

    # Data quality
    quality = check_data_quality(df)
    lines.append("\nData Quality:")
    lines.append(f"  Valid: {'✓' if quality['all_valid'] else '✗'}")
    lines.append(f"  Missing values: {quality['missing_values']:,}")
    lines.append(f"  Out-of-range values: {quality['out_of_range']:,}")
    lines.append(f"  Time gaps: {quality['time_gaps']}")
    lines.append(f"  Duplicate timestamps: {quality['duplicate_times']:,}")

    # Schema
    schema_warnings = validate_schema(df, expected_type)
    if schema_warnings:
        lines.append("\nSchema Warnings:")
        for w in schema_warnings:
            lines.append(f"  ⚠ {w}")
    else:
        lines.append("\n  Schema: OK ✓")

    # Alignment quality
    if aligned_df is not None and not aligned_df.empty:
        alignment = check_alignment_quality(aligned_df)
        lines.append("\nAlignment:")
        lines.append(f"  Match rate: {alignment['match_rate']:.1%}")
        lines.append(f"  Matched: {alignment['matched_rows']:,}")
        lines.append(f"  Forecast only: {alignment['forecast_only_rows']:,}")
        lines.append(f"  Observation only: {alignment['observation_only_rows']:,}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
