#!/usr/bin/env python3
"""Download and clip a coastline for the showcase map figures.

Fetches the Natural Earth 10 m coastline and clips it to the study region, so
the committed file is a few kilobytes rather than the 10 MB global dataset.
The clipped GeoJSON is committed; this script exists so it can be regenerated
or retargeted at a different region.

Natural Earth is public domain (naturalearthdata.com/about/terms-of-use).

Usage:
    python showcase/make_coastline.py [--bbox W S E N] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "examples/geo/coastline.geojson"

SOURCE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_coastline.geojson"
)

# Øresund and the approaches, a little wider than the 50 km grid run so the
# coastline reaches the panel edges instead of stopping inside them.
DEFAULT_BBOX = (11.5, 54.9, 13.7, 56.4)


def clip_segment(
    coordinates: list[list[float]], bbox: tuple[float, float, float, float]
) -> list[list[list[float]]]:
    """Split a line into the runs of points that fall inside the bbox.

    Keeps one point either side of each crossing so the drawn line reaches the
    panel edge rather than stopping at the last interior vertex.
    """
    west, south, east, north = bbox
    inside = [west <= x <= east and south <= y <= north for x, y in coordinates]

    runs: list[list[list[float]]] = []
    current: list[list[float]] = []
    for index, point in enumerate(coordinates):
        neighbour_inside = (index > 0 and inside[index - 1]) or (
            index + 1 < len(inside) and inside[index + 1]
        )
        if inside[index] or neighbour_inside:
            current.append(point)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    return [run for run in runs if len(run) > 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("W", "S", "E", "N"),
        default=DEFAULT_BBOX,
        help="Clip bounds as west south east north (degrees)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default=SOURCE_URL)
    args = parser.parse_args()

    print(f"Fetching {args.source}")
    try:
        with urllib.request.urlopen(args.source, timeout=120) as response:
            data = json.loads(response.read())
    except OSError as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1

    bbox = tuple(args.bbox)
    features = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "LineString":
            continue
        for run in clip_segment(geometry["coordinates"], bbox):
            features.append(
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        # Six decimals is ~0.1 m — far finer than a coastline
                        # drawn at 1 px, and it keeps the file small.
                        "coordinates": [[round(x, 6), round(y, 6)] for x, y in run],
                    },
                }
            )

    if not features:
        print(f"No coastline found in bbox {bbox}", file=sys.stderr)
        return 1

    clipped = {
        "type": "FeatureCollection",
        "properties": {
            "source": "Natural Earth 10m coastline (public domain)",
            "bbox": list(bbox),
        },
        "features": features,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clipped, separators=(",", ":")))

    points = sum(len(f["geometry"]["coordinates"]) for f in features)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB)")
    print(f"  {len(features)} segments, {points} points, bbox {bbox}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
