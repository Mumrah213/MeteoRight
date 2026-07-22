#!/usr/bin/env python3
"""Find the four grid points enclosing each site, per model.

Bilinear interpolation needs the corners of the grid cell containing the
target, and every model uses a different grid spacing. Rather than assume a
spacing, this probes the API: request a small cross of offsets around each
site and collect the distinct grid coordinates the model snaps them to.

Writes examples/europe_multimodel/grid_corners.json, consumed by
build_grid_dataset.py.

Usage:
    python examples/europe_multimodel/find_grid_corners.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from historical.api import fetch_historical_forecast

from build_dataset import LOCATIONS, MODELS  # noqa: E402  (same directory)

OUTPUT = REPO_ROOT / "examples/europe_multimodel/grid_corners.json"
PROBE_DATE = "2026-03-01"

# Offsets large enough to cross a cell boundary on the coarsest grid (ECMWF,
# 0.25 deg) while staying near the site. The probe keeps whatever distinct
# cells come back, so finer grids simply return more of them.
OFFSETS = [
    (dlat, dlon)
    for dlat in (-0.30, -0.13, 0.0, 0.13, 0.30)
    for dlon in (-0.40, -0.18, 0.0, 0.18, 0.40)
]

SLEEP_SECONDS = 1.5


def _enclosing_corners(
    cells: set[tuple[float, float]], lat: float, lon: float
) -> list[tuple[float, float]]:
    """The four grid cells bracketing (lat, lon), one per quadrant.

    Returns an empty list when any quadrant is unpopulated, which means the
    probe offsets did not reach across a cell boundary on that side.
    """
    # Pick the two bracketing grid lines on each axis independently, then take
    # their four intersections. A site lying exactly on a grid line brackets
    # with that line on both sides, so the cell degenerates to a segment and
    # bilinear reduces to linear interpolation along the other axis — still a
    # valid interpolation, and better than discarding the site.
    lats = sorted({c[0] for c in cells})
    lons = sorted({c[1] for c in cells})

    def _bracket(values: list[float], target: float) -> list[float]:
        below = [v for v in values if v <= target]
        above = [v for v in values if v >= target]
        if not below or not above:
            return []
        low, high = below[-1], above[0]
        return [low] if low == high else [low, high]

    lat_pair = _bracket(lats, lat)
    lon_pair = _bracket(lons, lon)
    if not lat_pair or not lon_pair:
        return []

    available = {(c[0], c[1]) for c in cells}
    corners = [(la, lo) for la in lat_pair for lo in lon_pair if (la, lo) in available]
    # Every intersection must have actually been observed, or the cube would
    # carry a hole the interpolation cannot fill.
    return corners if len(corners) == len(lat_pair) * len(lon_pair) else []


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    result: dict[str, dict[str, list[list[float]]]] = {}

    for loc in LOCATIONS:
        result[loc.slug] = {}
        for model in MODELS:
            cells: set[tuple[float, float]] = set()
            for dlat, dlon in OFFSETS:
                try:
                    response = fetch_historical_forecast(
                        loc.lat + dlat,
                        loc.lon + dlon,
                        PROBE_DATE,
                        PROBE_DATE,
                        ("temperature_2m",),
                        model,
                    )
                except Exception as exc:  # noqa: BLE001 - probe is best-effort
                    print(f"  {loc.slug}/{model} offset({dlat},{dlon}) failed: {exc}")
                    continue
                cells.add((round(float(response["latitude"]), 4), round(float(response["longitude"]), 4)))
                time.sleep(SLEEP_SECONDS)

            # Take the nearest cell in each quadrant around the site, not the
            # four nearest overall: on a coarse grid the four nearest can all
            # sit on one side, and bilinear would then extrapolate rather than
            # interpolate. One cell per quadrant guarantees the site lies
            # inside the resulting cell.
            corners = _enclosing_corners(cells, loc.lat, loc.lon)
            status = "ok" if corners else "NO ENCLOSING CELL"
            result[loc.slug][model] = [list(c) for c in corners]
            print(f"{loc.slug:<12} {model:<16} {len(cells)} cells -> {len(corners)} corners  {status}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
