"""Fetch forecast and observation series for one point from a backend.

The orchestrator needs forecast values and matching ground truth for the same
point and target times. Against a self-hosted Open-Meteo backend that means:

* forecast: ``/v1/forecast`` with ``models=<domain>`` over a date range
* truth:    ``/v1/archive`` (ERA5 reanalysis) over the same range

Both come back as hourly arrays which we turn into tidy DataFrames keyed by
target time, ready for the verification join. These are deliberately separate
from the CLI's ``cmd_download`` (which targets the public Single Runs API with
``run=``/``model=``); this path speaks the self-hosted convention.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from downloader.api import _make_client


def _hourly_frame(payload: dict[str, Any], variables: tuple[str, ...]) -> pd.DataFrame:
    """Turn an Open-Meteo hourly payload into a tidy DataFrame.

    Columns: ``time`` (UTC tz-aware) plus one column per requested variable.
    """
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    data: dict[str, Any] = {"time": pd.to_datetime(times, utc=True)}
    for var in variables:
        data[var] = hourly.get(var, [None] * len(times))
    return pd.DataFrame(data)


def fetch_forecast_series(
    lat: float,
    lon: float,
    model_domain: str,
    variables: tuple[str, ...],
    start: str,
    end: str,
    *,
    base_url: str,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetch a model's forecast hourly series for [start, end] at one point."""
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "models": model_domain,
        "start_date": start,
        "end_date": end,
    }
    with _make_client(timeout) as client:
        response = client.get(base_url, params=params)
        response.raise_for_status()
        return _hourly_frame(response.json(), variables)


def fetch_observation_series(
    lat: float,
    lon: float,
    variables: tuple[str, ...],
    start: str,
    end: str,
    *,
    archive_url: str,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetch ERA5 archive (truth) hourly series for [start, end] at one point."""
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "start_date": start,
        "end_date": end,
    }
    with _make_client(timeout) as client:
        response = client.get(archive_url, params=params)
        response.raise_for_status()
        return _hourly_frame(response.json(), variables)


def archive_url_for(base_url: str) -> str:
    """Derive the archive endpoint from a forecast base URL."""
    if "/v1/" in base_url:
        return base_url.rsplit("/v1/", 1)[0] + "/v1/archive"
    return base_url.rstrip("/") + "/v1/archive"
