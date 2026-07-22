#!/usr/bin/env python3
"""Download the grid-corner forecasts needed for bilinear interpolation.

The main dataset holds one forecast point per site: whichever grid cell each
model snaps the site to. Bilinear interpolation needs the four cells enclosing
the site, so this downloads the remaining corners listed in
grid_corners.json (written by find_grid_corners.py).

Each corner is stored as its own location, named "<site>__c<n>", so the
existing loaders treat them as ordinary points. The verification step then
interpolates the four corners of a site onto that site's observation location.

Usage:
    python examples/europe_multimodel/build_grid_dataset.py
    python examples/europe_multimodel/build_grid_dataset.py --smoke   # 1 site
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from historical.download_planner import (
    LocationSpec,
    build_download_plan,
    execute_download_plan,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import (  # noqa: E402
    FULL_END,
    FULL_START,
    LEAD_DAYS,
    LOCATIONS,
    MODELS,
    SMOKE_END,
    SMOKE_START,
    VARIABLES,
)

CORNERS = REPO_ROOT / "examples/europe_multimodel/grid_corners.json"
OUTPUT_DIR = REPO_ROOT / "examples/europe_multimodel/data_grid"

# The corner probe returns coordinates already snapped to each model's grid.
# Requesting a snapped coordinate returns that same cell, so a corner can be
# fetched by asking for its own centre.
MAX_GRID_DISTANCE_KM = 60.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    if not CORNERS.exists():
        print(f"Missing {CORNERS}; run find_grid_corners.py first", file=sys.stderr)
        return 1
    corners = json.loads(CORNERS.read_text())

    sites = {loc.slug: loc for loc in LOCATIONS}
    if args.smoke:
        sites = {k: v for k, v in list(sites.items())[:1]}

    start, end = (SMOKE_START, SMOKE_END) if args.smoke else (FULL_START, FULL_END)
    output_dir = OUTPUT_DIR.with_name("data_grid_smoke") if args.smoke else OUTPUT_DIR

    # One LocationSpec per (site, model, corner). Corners are deduplicated
    # within a model only: each model needs its own four enclosing cells, so a
    # coordinate shared with another model must still be fetched for both.
    plans = []
    skipped: list[str] = []
    for slug, site in sites.items():
        for model, cells in corners.get(slug, {}).items():
            # Two corners is the degenerate-but-valid case of a site lying on
            # a grid line: bilinear reduces to linear along the other axis.
            if len(cells) < 2:
                skipped.append(f"{slug}/{model}")
                continue
            seen: set[tuple[float, float]] = set()
            for lat, lon in cells:
                key = (round(lat, 4), round(lon, 4))
                if key in seen:
                    continue
                seen.add(key)
                spec = LocationSpec(
                    name=f"{site.name}__{model}__c{len(seen) - 1}",
                    lat=float(lat),
                    lon=float(lon),
                    max_grid_distance_km=MAX_GRID_DISTANCE_KM,
                )
                plans.append((spec, model))

    if skipped:
        print(f"Skipping {len(skipped)} site/model pairs without an enclosing cell: {skipped}")

    # Group by model so each model only requests its own corners.
    tasks = []
    for spec, model in plans:
        plan = build_download_plan(
            locations=[spec],
            start_date=start,
            end_date=end,
            variables=VARIABLES,
            models=(model,),
            kinds=("previous_runs",),
            lead_days=LEAD_DAYS,
            chunk_months=1,
        )
        tasks.extend(plan.tasks)

    from historical.download_planner import DownloadPlan

    plan = DownloadPlan(tasks=tuple(tasks))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = plan.write_manifest(output_dir / "metadata" / "manifest.json")
    print(f"Planned {plan.request_count} requests -> {manifest}")

    if args.plan_only:
        return 0

    result = execute_download_plan(plan, output_dir, resume=True, max_tasks=args.max_tasks)
    print(
        f"attempted={result['requests_attempted']} "
        f"skipped={result['requests_skipped']} "
        f"rows={result['rows_written']} "
        f"errors={len(result['errors'])}"
    )
    for error in result["errors"][:10]:
        print(f"  ERROR {error['task_id']}: {error['error']}")
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
