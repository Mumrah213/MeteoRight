"""Resolve a named area (one or more places) to a single grid point.

For an area like "Malmö-Copenhagen" we geocode each place, take the midpoint,
and snap to the nearest native model grid point. The single-point result keeps
the downstream verification pipeline unchanged (no spatial pooling); the point
is the one the forecast model actually resolves for that area.
"""

from __future__ import annotations

from typing import Any

from agent_tools.geocoding import geocode_location
from forecast.grid_points import find_grid_points, haversine_km


def _err(exc_type: str, message: str) -> dict[str, Any]:
    return {"type": exc_type, "message": message}


def _split_places(area: str | list[str]) -> list[str]:
    """Accept a list, or a string like 'Malmö-Copenhagen' / 'Malmö, Copenhagen'."""
    if isinstance(area, list):
        return [p.strip() for p in area if p.strip()]
    text = area.replace(" area", "").replace(" region", "")
    for sep in (",", "-", "/", " and ", " & "):
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    return [text.strip()]


def resolve_area(
    area: str | list[str],
    *,
    use_network: bool = True,
    search_radius_km: float = 60.0,
) -> dict[str, Any]:
    """Resolve an area to its representative (nearest-to-center) grid point.

    Args:
        area: A place, list of places, or a string like "Malmö-Copenhagen".
        use_network: Passed to geocoding (False = presets only).
        search_radius_km: Half-side of the grid search box around the center.

    Returns:
        ``{"places": [geocode dicts],
           "center": {"lat", "lon"},
           "grid_point": {"lat", "lon", "distance_km"} | None,
           "error": None | {"type", "message"}}``
    """
    names = _split_places(area)
    if not names:
        return {"places": [], "center": None, "grid_point": None,
                "error": _err("EmptyArea", "No place names given")}

    places = [geocode_location(n, use_network=use_network) for n in names]
    resolved = [p for p in places if p["lat"] is not None and p["lon"] is not None]
    if not resolved:
        first_err = next((p["error"] for p in places if p["error"]), None)
        return {"places": places, "center": None, "grid_point": None,
                "error": first_err or _err("LocationNotFound", "Could not geocode any place")}

    center_lat = sum(p["lat"] for p in resolved) / len(resolved)
    center_lon = sum(p["lon"] for p in resolved) / len(resolved)
    center = {"lat": center_lat, "lon": center_lon}

    # find_grid_points is pure (no network); pick the point nearest the center.
    points = find_grid_points(center_lat, center_lon, radius_km=search_radius_km, models=("ifs",))
    if not points:
        # Fall back to the analytic nearest 0.25° cell if the box was too small.
        return {"places": places, "center": center, "grid_point": None,
                "error": _err("NoGridPoint",
                              f"No grid point within {search_radius_km} km of center")}

    nearest = min(
        points,
        key=lambda gp: haversine_km(center_lat, center_lon, gp.latitude, gp.longitude),
    )
    return {
        "places": places,
        "center": center,
        "grid_point": {
            "lat": nearest.latitude,
            "lon": nearest.longitude,
            "distance_km": round(
                haversine_km(center_lat, center_lon, nearest.latitude, nearest.longitude), 3
            ),
        },
        "error": None,
    }
