#!/usr/bin/env python3
"""Download the 12-month, 8-site, 3-model European verification dataset.

Answers the procurement questions the README poses: whether one provider is
best, whether a blend beats it, and how far ahead either answer holds. That
needs a lead-time axis, so forecasts come from the previous-runs API
(`previous_dayN` = issued N days before the valid time) rather than the
stitched historical-forecast endpoint, which does not preserve issue time.

Sites are individual points, not a grid. Each response carries the model's
own grid coordinate and elevation, so the forecast-vs-observation grid
mismatch is recorded per site rather than silently absorbed.

The plan is checkpointed: re-running resumes and skips completed tasks.

Usage:
    python examples/europe_multimodel/build_dataset.py            # full year
    python examples/europe_multimodel/build_dataset.py --smoke    # one month
"""

from __future__ import annotations

import argparse
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

OUTPUT_DIR = REPO_ROOT / "examples/europe_multimodel/data"

# Coordinates verified to land on land-valid grid cells (non-zero elevation)
# for ecmwf_ifs025. Stockholm is nudged inland from the city centre, which
# snaps to a Baltic water cell at elevation 0.
LOCATIONS = [
    LocationSpec("Copenhagen", 55.676, 12.568),
    LocationSpec("Oslo", 59.913, 10.752),
    LocationSpec("Stockholm", 59.350, 17.950),
    LocationSpec("Bergen", 60.393, 5.325),
    LocationSpec("Berlin", 52.520, 13.405),
    LocationSpec("Paris", 48.857, 2.352),
    LocationSpec("Zurich", 47.377, 8.542),
    LocationSpec("Lisbon", 38.722, -9.139),
]

MODELS = ("ecmwf_ifs025", "icon_seamless", "gfs_seamless")
VARIABLES = ("temperature_2m", "wind_speed_10m", "precipitation")

# previous_day1 is the shortest lead this API exposes (~24-48 h). Sub-24 h
# leads are not available here, so this dataset complements the existing
# 48-hour Copenhagen sample rather than replacing it.
LEAD_DAYS = (1, 2, 3, 4, 5, 6, 7)

FULL_START, FULL_END = "2025-07-22", "2026-07-21"
SMOKE_START, SMOKE_END = "2026-03-01", "2026-03-31"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one month, one site, for testing")
    parser.add_argument("--plan-only", action="store_true", help="write manifest, do not download")
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    start, end = (SMOKE_START, SMOKE_END) if args.smoke else (FULL_START, FULL_END)
    locations = LOCATIONS[:1] if args.smoke else LOCATIONS
    output_dir = OUTPUT_DIR.with_name("data_smoke") if args.smoke else OUTPUT_DIR

    plan = build_download_plan(
        locations=locations,
        start_date=start,
        end_date=end,
        variables=VARIABLES,
        models=MODELS,
        kinds=("observations", "previous_runs"),
        lead_days=LEAD_DAYS,
        chunk_months=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = plan.write_manifest(output_dir / "metadata" / "manifest.json")
    print(f"Planned {plan.request_count} requests -> {manifest}")

    if args.plan_only:
        return 0

    result = execute_download_plan(
        plan,
        output_dir,
        resume=True,
        max_tasks=args.max_tasks,
    )

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
