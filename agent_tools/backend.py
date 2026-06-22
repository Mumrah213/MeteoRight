"""Runtime capability probe for an Open-Meteo backend.

The agent tools must not hardcode which models a backend serves or what date
range it covers, because that drifts the moment more data is synced. This module
asks the live backend what it actually has, so the rest of the tool layer can
resolve a question against current reality.

A self-hosted Open-Meteo backend serves models by *domain name* (e.g.
``ecmwf_ifs025``) via the plural ``models=`` query parameter, binds IPv4 only,
and returns an empty ``hourly`` array (HTTP 200) for a model/date with no data.
We exploit that: a model "has data" for a date if its hourly array is non-empty.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from downloader.api import _make_client

# Candidate self-hosted domain names to probe when the caller doesn't supply a
# list. This is only a starting set for discovery — models present on the
# backend but absent here can be added by the caller; models here but absent on
# the backend are simply dropped. It is NOT the source of truth for availability.
DEFAULT_CANDIDATE_MODELS = (
    "ecmwf_ifs025",
    "ecmwf_ifs04",
    "dwd_icon",
    "dwd_icon_eu",
    "dwd_icon_d2",
    "ncep_gfs025",
    "ncep_gfs013",
    "dmi_harmonie_arome_europe",
    "metno_nordic",
    "knmi_harmonie_arome_europe",
)

# A neutral probe point (Copenhagen) and the variable used to detect data.
_PROBE_LAT = 55.605
_PROBE_LON = 12.574
_PROBE_VAR = "temperature_2m"

# How far back to look when discovering the coverage window.
_COVERAGE_LOOKBACK_DAYS = 120


def _err(exc_type: str, message: str) -> dict[str, Any]:
    return {"type": exc_type, "message": message}


def _hourly_count(payload: dict[str, Any], var: str = _PROBE_VAR) -> int:
    """Number of non-null hourly values for ``var`` in an Open-Meteo response."""
    hourly = payload.get("hourly") or {}
    values = hourly.get(var) or []
    return sum(1 for v in values if v is not None)


def _probe_day(client: httpx.Client, base_url: str, model: str, day: date) -> int:
    """Return the non-null hourly count for ``model`` on ``day``.

    ``0`` means the backend answered but had no data; ``-1`` means the request
    could not be completed (connection error or non-200), so callers can tell
    "unreachable" apart from "no data here".
    """
    params = {
        "latitude": str(_PROBE_LAT),
        "longitude": str(_PROBE_LON),
        "hourly": _PROBE_VAR,
        "timezone": "UTC",
        "models": model,
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
    }
    try:
        response = client.get(base_url, params=params)
    except httpx.HTTPError:
        return -1
    if response.status_code != 200:
        return -1
    try:
        return _hourly_count(response.json())
    except ValueError:
        return -1


def _discover_coverage(
    client: httpx.Client, base_url: str, model: str, *, lookback_days: int
) -> dict[str, str] | None:
    """Find the [start, end] date range with data by sampling backwards.

    Samples every few days from today back to ``lookback_days`` to find any day
    with data, then walks outward day-by-day to the contiguous edges. Returns
    ``None`` if no day in the window has data.
    """
    today = datetime.now(UTC).date()

    # Coarse scan to find any day with data.
    anchor: date | None = None
    step = 3
    for offset in range(0, lookback_days + 1, step):
        day = today - timedelta(days=offset)
        if _probe_day(client, base_url, model, day) > 0:
            anchor = day
            break
    if anchor is None:
        return None

    # Walk forward to the latest contiguous day with data.
    end = anchor
    while end < today and _probe_day(client, base_url, model, end + timedelta(days=1)) > 0:
        end += timedelta(days=1)

    # Walk backward to the earliest contiguous day with data.
    start = anchor
    floor = today - timedelta(days=lookback_days)
    while start > floor and _probe_day(client, base_url, model, start - timedelta(days=1)) > 0:
        start -= timedelta(days=1)

    return {"start": start.isoformat(), "end": end.isoformat()}


def describe_backend(
    base_url: str = "http://127.0.0.1:8080/v1/forecast",
    *,
    candidate_models: tuple[str, ...] | None = None,
    discover_coverage: bool = True,
    timeout: float = 30.0,
    lookback_days: int = _COVERAGE_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """Probe an Open-Meteo backend for the models and coverage it actually serves.

    Args:
        base_url: Forecast endpoint of the backend. Use ``127.0.0.1`` (not
            ``localhost``) for a local self-hosted server, which is IPv4-only.
        candidate_models: Domain names to probe. Defaults to
            ``DEFAULT_CANDIDATE_MODELS``. Pass extra names to discover models not
            in the default set.
        discover_coverage: If True, also find each model's available date range
            (slower; several requests per model).
        timeout: Per-request timeout in seconds.
        lookback_days: How far back to search for coverage.

    Returns:
        ``{"base_url", "reachable": bool,
           "models": [{"backend_name", "has_data": True,
                       "coverage": {"start", "end"} | None}],
           "error": None | {"type", "message"}}``
        Models with no data are omitted from the list. On total failure
        (backend unreachable) ``reachable`` is False and ``models`` is empty.
    """
    candidates = candidate_models or DEFAULT_CANDIDATE_MODELS
    result: dict[str, Any] = {
        "base_url": base_url,
        "reachable": False,
        "models": [],
        "error": None,
    }

    try:
        client = _make_client(timeout)
    except Exception as exc:  # pragma: no cover - client construction is trivial
        result["error"] = _err(type(exc).__name__, str(exc))
        return result

    today = datetime.now(UTC).date()
    models: list[dict[str, Any]] = []
    reached_any = False
    try:
        with client:
            for model in candidates:
                # A model is "present" if it has data anywhere in the lookback
                # window — not just recently, since a backend may hold an older
                # archive. Reuse the coarse scan inside _discover_coverage to
                # find the contiguous range; when coverage discovery is off,
                # fall back to a coarse presence scan.
                coverage = None
                present = False
                for offset in range(0, lookback_days + 1, 3):
                    count = _probe_day(client, base_url, model, today - timedelta(days=offset))
                    if count >= 0:
                        # Backend answered (even if empty) -> it is reachable.
                        reached_any = True
                    if count > 0:
                        present = True
                        break

                if not present:
                    continue

                result["reachable"] = True
                if discover_coverage:
                    coverage = _discover_coverage(
                        client, base_url, model, lookback_days=lookback_days
                    )
                models.append(
                    {"backend_name": model, "has_data": True, "coverage": coverage}
                )
    except httpx.HTTPError as exc:
        if not models:
            result["error"] = _err("BackendUnreachable", f"{base_url}: {exc}")
        return result
    except Exception as exc:  # defensive: never raise across the tool boundary
        result["error"] = _err(type(exc).__name__, str(exc))
        return result

    result["models"] = models
    if not models and result["error"] is None:
        if reached_any:
            result["error"] = _err(
                "NoModelsWithData",
                f"Backend at {base_url} is reachable but no candidate models "
                f"returned data within {lookback_days} days",
            )
        else:
            result["error"] = _err(
                "BackendUnreachable",
                f"No response from {base_url} for any candidate model",
            )
    return result
