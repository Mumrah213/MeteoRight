"""Data loaders for the analysis layer.

Loads weather data directly from parquet files produced by the ingestion
pipeline and returns clean, standardized pandas DataFrames.

Parquet layout assumed:
  Observations:  {output_dir}/observations/year={Y}/month={M}/data.parquet
  Forecasts:     {output_dir}/forecasts/model={MODEL}/year={Y}/month={M}/data.parquet

Canonical column standardization:
  All loaders normalize to the analysis canonical schema defined in
  src/analysis/canonical.py.

Usage
-----
>>> from src.analysis.loaders import load_historical, load_forecast

>>> obs = load_historical("temperature_2m", start="2024-01")
>>> fcst = load_forecast("icon_eu", "temperature_2m", start="2024-01")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

warnings.filterwarnings("ignore", message=".*empty or all-NA entries.*")

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

# Default data directory (matches StorageConfig)
DEFAULT_DATA_DIR = Path("data")

# Columns that exist in wide-format observation parquet files
OBSERVATION_WIDE_COLUMNS = [
    "observation_time",
    "latitude",
    "longitude",
    "elevation",
]

# Columns that exist in wide-format forecast parquet files
FORECAST_WIDE_COLUMNS = [
    "forecast_issue_time",
    "forecast_target_time",
    "lead_hours",
    "model",
    "latitude",
    "longitude",
    "elevation",
]

# Partition columns (added by storage.py)
PARTITION_COLUMNS = {"year", "month"}

# ── Utility: discover parquet files ─────────────────────────────────────────


def _discover_parquet(
    base_path: Path,
    year: int | None = None,
    month: int | None = None,
    model: str | None = None,
) -> list[Path]:
    """Discover partitioned parquet files.

    Args:
        base_path: Root directory for the dataset (e.g., data/forecasts).
        year: Filter to this year (None = all).
        month: Filter to this month (None = all).
        model: Filter to this model (None = all, forecasts only).

    Returns:
        List of Path objects pointing to data.parquet files.
    """
    if not base_path.exists():
        return []

    results: list[Path] = []
    for partition_dir in sorted(base_path.rglob("data.parquet")):
        rel = partition_dir.parent.relative_to(base_path)
        parts = rel.parts

        # Parse partition keys
        parsed_year = None
        parsed_month = None
        parsed_model = None

        if model is not None and len(parts) >= 3:
            # model=X/year=Y/month=M
            parsed_model = parts[0].split("=")[1] if "=" in parts[0] else None
            parsed_year = int(parts[1].split("=")[1])
            parsed_month = int(parts[2].split("=")[1])
        elif len(parts) >= 2:
            # year=Y/month=M
            parsed_year = int(parts[0].split("=")[1])
            parsed_month = int(parts[1].split("=")[1])

        # Apply filters
        if year is not None and parsed_year != year:
            continue
        if month is not None and parsed_month != month:
            continue
        if model is not None and parsed_model != model:
            continue

        results.append(partition_dir)

    return results


# ── Utility: wide → long conversion ─────────────────────────────────────────


def _wide_to_long(
    df: pd.DataFrame,
    id_cols: list[str],
    value_prefix: str = "",
) -> pd.DataFrame:
    """Convert wide-format weather DataFrame to long format.

    Wide format (one row per timestamp):
        time, latitude, longitude, temperature_2m, precipitation, ...

    Long format (one row per variable at each timestamp):
        time, latitude, longitude, variable, value

    Args:
        df: Wide-format DataFrame.
        id_cols: Columns to keep as-is (time, lat, lon, etc.).
        value_prefix: Optional prefix to strip from variable names.

    Returns:
        Long-format DataFrame with columns: id_cols + ["variable", "value"].
    """
    # Identify variable columns (all non-id, non-partition columns)
    known_cols = set(id_cols + list(PARTITION_COLUMNS))
    var_cols = [c for c in df.columns if c not in known_cols]

    if not var_cols:
        # No variable columns found — return as-is
        logger.warning("No variable columns found in DataFrame")
        return df

    # Filter to relevant columns
    keep = [c for c in id_cols if c in df.columns]
    keep += var_cols
    df = df[keep].copy()

    # Melt to long format
    df = df.melt(
        id_vars=keep[: -len(var_cols)],  # All id columns except the last var group
        value_vars=var_cols,
        var_name="variable",
        value_name="value",
    )

    # Strip value_prefix if present
    if value_prefix and not var_cols[0].startswith(value_prefix):
        pass  # Already clean
    else:
        # Remove common prefixes
        df["variable"] = df["variable"].str.replace("^temp_", "", regex=True)
        df["variable"] = df["variable"].str.replace("^precip_", "", regex=True)

    return df


# ── Loaders ─────────────────────────────────────────────────────────────────


def load_historical(
    variable: str | None = None,
    start: str | None = None,
    end: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    station: str | None = None,
    data_dir: Path | None = None,
    year: int | None = None,
    month: int | None = None,
) -> pd.DataFrame:
    """Load observation data from parquet files.

    Reads from {data_dir}/observations/year={Y}/month={M}/data.parquet.

    Returns a long-format DataFrame with columns:
        observation_time, latitude, longitude, variable, value
    Plus optional elevation if present.

    Args:
        variable: Variable name filter (e.g. "temperature_2m").
            If None, all variables are loaded.
        start: Start date filter (YYYY-MM-DD or YYYY-MM). Inclusive.
        end: End date filter (YYYY-MM-DD or YYYY-MM). Inclusive.
        latitude: Filter to specific latitude.
        longitude: Filter to specific longitude.
        station: Deprecated alias for data_dir.
        data_dir: Root data directory. Defaults to "data/".
        year: Filter to specific year.
        month: Filter to specific month.

    Returns:
        Long-format DataFrame with columns:
            observation_time, latitude, longitude, variable, value,
            (elevation if present)

    Examples
    --------
    >>> df = load_historical("temperature_2m", start="2024-01")
    >>> df.head()
                 observation_time  latitude  longitude       variable  value
        0 2024-01-01 00:00:00+00:00   55.605    12.574  temperature_2m    2.5
        1 2024-01-01 01:00:00+00:00   55.605    12.574  temperature_2m    2.3
    """
    data_dir = _resolve_data_dir(data_dir, station)

    # Build year/month filters from start/end if provided
    year_filter = year
    month_filter = month
    if start:
        parts = start.split("-")
        year_filter = int(parts[0]) if year_filter is None else year_filter
    if end:
        parts = end.split("-")
        year_filter = int(parts[0]) if year_filter is None else year_filter
    if start and end and start.split("-")[0] != end.split("-")[0] and year is None:
        year_filter = None

    # Discover parquet files
    obs_dir = data_dir / "observations"
    parquet_files = _discover_parquet(obs_dir, year=year_filter, month=month_filter)

    if not parquet_files:
        logger.warning("No observation parquet files found in %s", obs_dir)
        return pd.DataFrame()

    # Load and concatenate
    frames: list[pd.DataFrame] = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        df["_source_file"] = str(path)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True, copy=False)

    # Drop tracking column before wide-to-long conversion
    result.drop(columns=["_source_file"], errors="ignore", inplace=True)

    # Parse datetime columns
    if "observation_time" in result.columns:
        result["observation_time"] = pd.to_datetime(result["observation_time"], utc=True)

    # Convert to long format
    result = _wide_to_long(result, OBSERVATION_WIDE_COLUMNS)

    # Filter by variable if specified
    if variable and "variable" in result.columns:
        result = result[result["variable"] == variable]

    # Filter by date range
    if start or end:
        if "observation_time" in result.columns:
            mask = pd.Series(True, index=result.index)
            if start:
                mask &= result["observation_time"] >= pd.Timestamp(start, tz="UTC")
            if end:
                # Convert end date to exclusive upper bound (next day)
                end_ts = pd.Timestamp(end, tz="UTC")
                # If only YYYY-MM provided, add day 01 then advance to next day
                if len(end) <= 7:  # "2026-05" format
                    end_ts = end_ts.replace(day=1) + pd.Timedelta(days=1)
                else:  # "2026-05-31" format
                    end_ts = end_ts + pd.Timedelta(days=1)
                mask &= result["observation_time"] < end_ts
            result = result[mask]

    # Filter by location
    if latitude is not None:
        result = result[result["latitude"] == latitude]
    if longitude is not None:
        result = result[result["longitude"] == longitude]

    # Select canonical columns
    canonical = [
        c
        for c in ["observation_time", "latitude", "longitude", "variable", "value"]
        if c in result.columns
    ]
    if "elevation" in result.columns:
        canonical.append("elevation")
    result = result[[c for c in canonical if c in result.columns]]

    logger.info("Loaded %d observation rows", len(result))
    return result.reset_index(drop=True)


def load_forecast(
    model: str,
    variable: str | None = None,
    start: str | None = None,
    end: str | None = None,
    leadtime_hours: int | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    station: str | None = None,
    data_dir: Path | None = None,
    year: int | None = None,
    month: int | None = None,
) -> pd.DataFrame:
    """Load forecast data from parquet files.

    Reads from {data_dir}/forecasts/model={MODEL}/year={Y}/month={M}/data.parquet.

    Returns a long-format DataFrame with columns:
        forecast_issue_time, forecast_target_time, lead_hours,
        model, latitude, longitude, variable, value

    Args:
        model: Model name (e.g. "icon_eu", "gfs", "ecmwf_ifs_hres").
        variable: Variable name filter (e.g. "temperature_2m").
            If None, all variables are loaded.
        start: Start date filter (YYYY-MM-DD or YYYY-MM). Inclusive.
        end: End date filter (YYYY-MM-DD or YYYY-MM). Inclusive.
        leadtime_hours: Filter to specific lead time in hours.
        latitude: Filter to specific latitude.
        longitude: Filter to specific longitude.
        station: Deprecated alias for data_dir.
        data_dir: Root data directory. Defaults to "data/".
        year: Filter to specific year.
        month: Filter to specific month.

    Returns:
        Long-format DataFrame with columns:
            forecast_issue_time, forecast_target_time, lead_hours,
            model, latitude, longitude, variable, value,
            (elevation if present)

    Examples
    --------
    >>> df = load_forecast("icon_eu", "temperature_2m", start="2024-01")
    >>> df.head()
       forecast_issue_time forecast_target_time  lead_hours  model  ...
    0   2024-01-01 00:00:00   2024-01-01 00:00:00           0  icon_eu  ...
    """
    data_dir = _resolve_data_dir(data_dir, station)

    # Build year/month filters from start/end if provided
    year_filter = year
    month_filter = month
    if start:
        parts = start.split("-")
        year_filter = int(parts[0]) if year_filter is None else year_filter
    if end:
        parts = end.split("-")
        year_filter = int(parts[0]) if year_filter is None else year_filter
    if start and end and start.split("-")[0] != end.split("-")[0] and year is None:
        year_filter = None

    # Discover parquet files
    fcst_dir = data_dir / "forecasts"
    parquet_files = _discover_parquet(fcst_dir, model=model, year=year_filter, month=month_filter)

    if not parquet_files:
        logger.warning("No forecast parquet files found for model=%s in %s", model, fcst_dir)
        return pd.DataFrame()

    # Load and concatenate
    frames: list[pd.DataFrame] = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        df["_source_file"] = str(path)
        frames.append(df)

    result = pd.concat(frames, ignore_index=True, copy=False)

    # Drop tracking column before wide-to-long conversion
    result.drop(columns=["_source_file"], errors="ignore", inplace=True)

    # Parse datetime columns
    for dt_col in ["forecast_issue_time", "forecast_target_time"]:
        if dt_col in result.columns:
            result[dt_col] = pd.to_datetime(result[dt_col], utc=True)

    # Ensure lead_hours is numeric
    if "lead_hours" in result.columns:
        result["lead_hours"] = pd.to_numeric(result["lead_hours"], errors="coerce")

    # Convert to long format
    result = _wide_to_long(result, FORECAST_WIDE_COLUMNS)

    # Filter by variable if specified
    if variable and "variable" in result.columns:
        result = result[result["variable"] == variable]

    # Filter by date range
    if start or end:
        if "forecast_target_time" in result.columns:
            mask = pd.Series(True, index=result.index)
            if start:
                mask &= result["forecast_target_time"] >= pd.Timestamp(start, tz="UTC")
            if end:
                end_ts = pd.Timestamp(end, tz="UTC")
                if len(end) <= 7:  # "2026-05" format
                    end_ts = end_ts.replace(day=1) + pd.Timedelta(days=1)
                else:  # "2026-05-31" format
                    end_ts = end_ts + pd.Timedelta(days=1)
                mask &= result["forecast_target_time"] < end_ts
            result = result[mask]

    # Filter by leadtime
    if leadtime_hours is not None and "lead_hours" in result.columns:
        result = result[result["lead_hours"] == leadtime_hours]

    # Filter by location
    if latitude is not None:
        result = result[result["latitude"] == latitude]
    if longitude is not None:
        result = result[result["longitude"] == longitude]

    # Select canonical columns
    canonical = [
        c
        for c in [
            "forecast_issue_time",
            "forecast_target_time",
            "lead_hours",
            "model",
            "latitude",
            "longitude",
            "variable",
            "value",
        ]
        if c in result.columns
    ]
    if "elevation" in result.columns:
        canonical.append("elevation")
    result = result[[c for c in canonical if c in result.columns]]

    logger.info("Loaded %d forecast rows (model=%s)", len(result), model)
    return result.reset_index(drop=True)


def load_all_variables(
    start: str | None = None,
    end: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    station: str | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load all observation variables at once.

    Convenience wrapper around load_historical that returns all variables
    without filtering.

    Args:
        start: Start date filter.
        end: End date filter.
        latitude: Filter by latitude.
        longitude: Filter by longitude.
        station: Deprecated alias for data_dir.
        data_dir: Root data directory.

    Returns:
        Long-format DataFrame with all variables.
    """
    return load_historical(
        variable=None,
        start=start,
        end=end,
        latitude=latitude,
        longitude=longitude,
        station=station,
        data_dir=data_dir,
    )


def load_forecast_all_variables(
    model: str,
    start: str | None = None,
    end: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    station: str | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load all forecast variables for a given model.

    Convenience wrapper around load_forecast that returns all variables.

    Args:
        model: Model name.
        start: Start date filter.
        end: End date filter.
        latitude: Filter by latitude.
        longitude: Filter by longitude.
        station: Deprecated alias for data_dir.
        data_dir: Root data directory.

    Returns:
        Long-format DataFrame with all variables.
    """
    return load_forecast(
        model=model,
        variable=None,
        start=start,
        end=end,
        latitude=latitude,
        longitude=longitude,
        station=station,
        data_dir=data_dir,
    )


# ── Internal ────────────────────────────────────────────────────────────────


def _resolve_data_dir(
    data_dir: Path | None,
    station: str | None = None,
) -> Path:
    """Resolve data directory from arguments or config.

    Priority:
    1. Explicit data_dir argument
    2. Station name → mapped directory
    3. Default "data/"
    """
    if data_dir is not None:
        return Path(data_dir)

    if station:
        # Support legacy station→directory mapping
        station_dir = Path("data") / station
        if station_dir.exists():
            return station_dir

    return DEFAULT_DATA_DIR


def get_available_models(data_dir: Path | None = None) -> list[str]:
    """List available forecast models from the data directory.

    Args:
        data_dir: Root data directory. Defaults to "data/".

    Returns:
        List of model names found in the forecasts directory.
    """
    data_dir = _resolve_data_dir(data_dir, None)
    fcst_dir = data_dir / "forecasts"

    if not fcst_dir.exists():
        return []

    models: list[str] = []
    for model_dir in sorted(fcst_dir.iterdir()):
        if model_dir.is_dir() and model_dir.name.startswith("model="):
            model_name = model_dir.name.split("=", 1)[1]
            models.append(model_name)

    return models


def get_available_variables(df: pd.DataFrame) -> list[str]:
    """Extract unique variable names from a loaded DataFrame.

    Args:
        df: DataFrame returned by load_historical or load_forecast.

    Returns:
        List of unique variable names.
    """
    if "variable" not in df.columns:
        return []
    return sorted(df["variable"].dropna().unique().tolist())


def get_time_range(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Extract the time range from a DataFrame.

    Auto-detects which timestamp column exists:
    - observations: observation_time
    - forecasts: forecast_target_time

    Args:
        df: DataFrame returned by a loader.

    Returns:
        (min_time, max_time) tuple, or (None, None) if no data.
    """
    for ts_col in ["observation_time", "forecast_target_time"]:
        if ts_col in df.columns:
            times = pd.to_datetime(df[ts_col]).dropna()
            if len(times) > 0:
                return (times.min(), times.max())

    return (None, None)


# ── Backward compatibility ──────────────────────────────────────────────────


# Legacy aliases for users migrating from v2
def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a single parquet file directly.

    Low-level helper for cases where the above loaders don't fit.

    Args:
        path: Path to parquet file.

    Returns:
        DataFrame read from the parquet file.
    """
    return pd.read_parquet(Path(path))


def list_data_files(
    base_dir: Path | None = None,
    data_type: str = "all",
) -> list[Path]:
    """List all parquet data files in the data directory.

    Args:
        base_dir: Root data directory. Defaults to "data/".
        data_type: One of "observations", "forecasts", "all".

    Returns:
        List of Path objects.
    """
    base = _resolve_data_dir(base_dir, None)
    results: list[Path] = []

    if data_type in ("observations", "all"):
        results.extend(_discover_parquet(base / "observations"))
    if data_type in ("forecasts", "all"):
        for model_dir in (base / "forecasts").glob("model=*"):
            results.extend(_discover_parquet(model_dir))

    return sorted(results)


# Open-Meteo hourly variables (https://open-meteo.com/en/docs)
# These are the standard variables available from both the Forecast API and Archive API.
OPEN_METEO_HOURLY_VARIABLES: dict[str, str] = {
    "temperature_2m": "Air temperature at 2 meters above ground [°C]",
    "relative_humidity_2m": "Relative humidity at 2 meters above ground [%]",
    "dew_point_2m": "Dew point temperature at 2 meters above ground [°C]",
    "precipitation": "Precipitation [mm]",
    "rain": "Rain from large scale weather systems [mm]",
    "weather_code": "Weather code (integrated, see WMO codes)",
    "pressure_msl": "Surface air pressure [hPa]",
    "surface_pressure": "Surface pressure [hPa]",
    "wind_speed_10m": "Wind speed at 10 meters above ground [km/h]",
    "wind_direction_10m": "Wind direction at 10 meters above ground [°]",
    "wind_gusts_10m": "Wind gusts at 10 meters above ground [km/h]",
    "wind_speed_120m": "Wind speed at 120 meters (hub height of modern turbines) [km/h]",
    "wind_direction_120m": "Wind direction at 120 meters [°]",
    "shortwave_radiation": "Shortwave radiation [W/m²]",
    "direct_radiation": "Direct radiation [W/m²]",
    "direct_normal_irradiance": "Direct normal irradiance [W/m²]",
    "diffuse_radiation": "Diffuse radiation [W/m²]",
    "global_tilted_irradiance": "Global tilted irradiance [W/m²]",
    "soil_temperature_0cm": " soil temperature at 0 cm depth [°C]",
    "soil_moisture_0_to_1cm": "Soil moisture at 0-1 cm depth [m³/m³]",
    "soil_moisture_1_to_3cm": "Soil moisture at 1-3 cm depth [m³/m³]",
    "soil_moisture_3_to_9cm": "Soil moisture at 3-9 cm depth [m³/m³]",
    "soil_moisture_9_to_27cm": "Soil moisture at 9-27 cm depth [m³/m³]",
    "soil_moisture_27_to_81cm": "Soil moisture at 27-81 cm depth [m³/m³]",
}


def get_available_api_variables() -> list[str]:
    """Return list of available Open-Meteo hourly variables.

    Returns:
        Sorted list of variable names available from the Open-Meteo API.

    Examples
    --------
    >>> vars = get_available_api_variables()
    >>> print(vars[:5])  # ['dew_point_2m', 'direct_radiation', ...]
    """
    return sorted(OPEN_METEO_HOURLY_VARIABLES.keys())


def get_variable_description(variable: str) -> str | None:
    """Get the description for a variable.

    Args:
        variable: Variable name (e.g. "temperature_2m").

    Returns:
        Description string, or None if variable not found.
    """
    return OPEN_METEO_HOURLY_VARIABLES.get(variable)


def list_available_variables(
    data_dir: str | None = None, model: str | None = None
) -> dict[str, list[str]]:
    """List variables available in local parquet data.

    Args:
        data_dir: Data directory (defaults to "data/").
        model: Model name filter (only for forecasts).

    Returns:
        Dict with keys "observations" and/or "forecasts" containing
        lists of variable names found in the parquet files.

    Examples
    --------
    >>> vars = list_available_variables()
    >>> print(vars["observations"])  # ['precipitation', 'temperature_2m', ...]
    """
    result: dict[str, list[str]] = {}

    # Check observations
    obs_dir = Path(data_dir or "data") / "observations"
    obs_parquet = _discover_parquet(obs_dir)
    if obs_parquet:
        obs_df = pd.read_parquet(obs_parquet[0])
        result["observations"] = get_available_variables(obs_df)

    # Check forecasts
    if model:
        fcst_dir = Path(data_dir or "data") / "forecasts" / f"model={model}"
        fcst_parquet = _discover_parquet(fcst_dir)
        if fcst_parquet:
            fcst_df = pd.read_parquet(fcst_parquet[0])
            result["forecasts"] = get_available_variables(fcst_df)
    else:
        # Check all available models
        fcst_base = Path(data_dir or "data") / "forecasts"
        for model_dir in sorted(fcst_base.glob("model=*")):
            model_name = model_dir.name.replace("model=", "")
            fcst_parquet = _discover_parquet(model_dir)
            if fcst_parquet:
                fcst_df = pd.read_parquet(fcst_parquet[0])
                result[model_name] = get_available_variables(fcst_df)

    return result
