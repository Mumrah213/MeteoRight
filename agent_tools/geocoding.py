"""Resolve place names to coordinates.

Preset-first (the curated cities in config.LOCATION_PRESETS, which need no
network and have stable coordinates), then fall back to the free Open-Meteo
geocoding API for anything else. Returns a structured dict and never raises
across the tool boundary.
"""

from __future__ import annotations

import unicodedata
from typing import Any

import httpx

from config.settings import LOCATION_PRESETS
from downloader.api import _make_client

GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"


def _normalize_key(name: str) -> str:
    """Lowercase and strip diacritics so 'Malmö' matches the 'malmo' preset."""
    decomposed = unicodedata.normalize("NFKD", name.strip().lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _err(exc_type: str, message: str) -> dict[str, Any]:
    return {"type": exc_type, "message": message}


def geocode_location(name: str, *, use_network: bool = True) -> dict[str, Any]:
    """Resolve a place name to coordinates.

    Args:
        name: Place name, e.g. "Malmö" or "Copenhagen".
        use_network: If False, only the local presets are consulted.

    Returns:
        ``{"name", "lat", "lon", "country", "timezone",
           "source": "preset" | "open-meteo-geocoding",
           "error": None | {"type", "message"}}``
        On failure ``lat``/``lon`` are ``None`` and ``error`` is set.
    """
    base = {
        "name": name,
        "lat": None,
        "lon": None,
        "country": None,
        "timezone": None,
        "source": None,
        "error": None,
    }

    key = _normalize_key(name)
    preset = LOCATION_PRESETS.get(key)
    if preset:
        return {
            **base,
            "name": preset["name"],
            "lat": preset["lat"],
            "lon": preset["lon"],
            "country": preset.get("country"),
            "timezone": preset.get("timezone"),
            "source": "preset",
        }

    if not use_network:
        return {**base, "error": _err("LocationNotFound", f"No preset for '{name}'")}

    params = {"name": name, "count": 1, "format": "json"}
    try:
        with _make_client(timeout=15.0) as client:
            response = client.get(GEOCODING_API, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        return {**base, "error": _err("GeocodingRequestFailed", str(exc))}
    except ValueError as exc:
        return {**base, "error": _err("GeocodingBadResponse", str(exc))}

    results = payload.get("results") or []
    if not results:
        return {**base, "error": _err("LocationNotFound", f"No geocoding match for '{name}'")}

    top = results[0]
    return {
        **base,
        "name": top.get("name", name),
        "lat": top.get("latitude"),
        "lon": top.get("longitude"),
        "country": top.get("country"),
        "timezone": top.get("timezone"),
        "source": "open-meteo-geocoding",
    }
