"""Thin wrappers for the Open-Meteo backend.

These are simple, direct functions — no retry logic, no orchestration.
Just query the API and return JSON or CSV.

The base URL is resolved from ``METEORIGHT_OPEN_METEO_BASE_URL`` so the same
functions work against the public Open-Meteo API (the default) or a self-hosted
instance (e.g. ``http://127.0.0.1:8080``). Set the env var to point at a local
backend:

    export METEORIGHT_OPEN_METEO_BASE_URL="http://127.0.0.1:8080"

Usage:
    >>> from api.fetch import fetch_forecast, fetch_history
    >>> data = fetch_forecast(55.605, 12.574, models=["ecmwf_ifs_hres"])
    >>> history = fetch_history(55.605, 12.574, past_days=30)
"""


import os

import requests

# Public Open-Meteo host, used when no self-hosted backend is configured.
PUBLIC_BASE_URL = "https://api.open-meteo.com"


def _base_url() -> str:
    """Resolve the Open-Meteo host, preferring a self-hosted backend if set."""
    return (os.getenv("METEORIGHT_OPEN_METEO_BASE_URL") or PUBLIC_BASE_URL).rstrip("/")


def _build_url(endpoint: str, params: dict) -> str:
    """Build a full API URL targeting the configured Open-Meteo backend."""
    # Normalize params: convert lists to comma-separated strings
    normalized = {}
    for k, v in params.items():
        if isinstance(v, list):
            normalized[k] = ",".join(v)
        else:
            normalized[k] = v

    query = "&".join(
        f"{k}={requests.utils.quote(str(v), safe=',/')}" for k, v in normalized.items()
    )
    return f"{_base_url()}{endpoint}?{query}"


def fetch_forecast(
    lat: float,
    lon: float,
    models: list[str] | str = "ecmwf_ifs_hres",
    hourly: list[str] | None = None,
    daily: list[str] | None = None,
    past_days: int = 0,
    format_type: str = "json",
) -> dict:
    """Fetch a weather forecast from the self-hosted Open-Meteo backend.

    Args:
        lat: Latitude.
        lon: Longitude.
        models: Model name or list of models.
        hourly: Hourly variables to fetch.
        daily: Daily variables to fetch.
        past_days: Include past N days (default 0 = forecast only).
        format_type: "json" or "csv".

    Returns:
        Parsed JSON dict, or parsed CSV string.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "models": models,
        "format": format_type,
    }

    if isinstance(hourly, list) or hourly is not None:
        params["hourly"] = hourly

    if isinstance(daily, list) or daily is not None:
        params["daily"] = daily

    if past_days > 0:
        params["past_days"] = past_days

    url = _build_url("/v1/forecast", params)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    if format_type == "csv":
        return resp.text

    return resp.json()


def fetch_history(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    models: list[str] | str = "ecmwf_ifs025",
    hourly: list[str] | None = None,
    daily: list[str] | None = None,
    format_type: str = "json",
) -> dict:
    """Fetch historical weather data from the self-hosted Open-Meteo backend.

    Args:
        lat: Latitude.
        lon: Longitude.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        models: Model name or list of models.
        hourly: Hourly variables to fetch.
        daily: Daily variables to fetch.
        format_type: "json" or "csv".

    Returns:
        Parsed JSON dict, or parsed CSV string.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "models": models,
        "format": format_type,
    }

    if isinstance(hourly, list) or hourly is not None:
        params["hourly"] = hourly

    if isinstance(daily, list) or daily is not None:
        params["daily"] = daily

    url = _build_url("/v1/archive", params)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    if format_type == "csv":
        return resp.text

    return resp.json()


def fetch_climate_normals(
    lat: float,
    lon: float,
    monthly: list[str] | None = None,
    daily: list[str] | None = None,
    daily_window: int = 30,
    format_type: str = "json",
) -> dict:
    """Fetch approximate climate normals from the self-hosted Open-Meteo backend.

    Note: Climate normals require data to be synced in the local instance.
    Not all models provide climate data.

    Args:
        lat: Latitude.
        lon: Longitude.
        monthly: Monthly variables.
        daily: Daily variables.
        daily_window: Rolling window size for daily normals.
        format_type: "json" or "csv".

    Returns:
        Parsed JSON dict, or parsed CSV string.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "format": format_type,
        "daily_window": daily_window,
    }

    if isinstance(monthly, list) or monthly is not None:
        params["monthly"] = monthly

    if isinstance(daily, list) or daily is not None:
        params["daily"] = daily

    url = _build_url("/v1/climate", params)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    if format_type == "csv":
        return resp.text

    return resp.json()


def fetch_multiple_locations(
    locations: list[tuple[float, float]],
    models: list[str] | str = "ecmwf_ifs_hres",
    hourly: list[str] | None = None,
    daily: list[str] | None = None,
) -> dict:
    """Fetch forecast for multiple locations in one query.

    Args:
        locations: List of (lat, lon) tuples.
        models: Model name or list of models.
        hourly: Hourly variables to fetch.
        daily: Daily variables to fetch.

    Returns:
        Parsed JSON dict with all locations.
    """
    params = {
        "latitude": ",".join(str(lat) for lat, _ in locations),
        "longitude": ",".join(str(lon) for _, lon in locations),
        "models": models,
        "format": "json",
    }

    if isinstance(hourly, list) or hourly is not None:
        params["hourly"] = hourly

    if isinstance(daily, list) or daily is not None:
        params["daily"] = daily

    url = _build_url("/v1/forecast", params)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    return resp.json()
