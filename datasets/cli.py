"""CLI integration for curated research dataset preparation.

Adds the ``prepare-dataset`` command to the weather-analyzer CLI.

Usage (CLI)
-----------
# Full preparation with preset location
$ weather-analyzer prepare-dataset \\
    --location copenhagen \\
    --years 2 \\
    --variables all \\
    --build-all

# Custom location and date range
$ weather-analyzer prepare-dataset \\
    --lat 55.605 --lon 12.574 \\
    --start 2024-01-01 --end 2025-12-31 \\
    --variables temperature_2m,precipitation,wind_speed_10m \\
    --model icon_eu \\
    --build-metrics \\
    --output-dir ./datasets

# Show existing dataset info
$ weather-analyzer dataset-info ./datasets/copenhagen_2y
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def add_prepare_dataset_command(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Add the prepare-dataset subcommand to an argument parser.

    Args:
        subparsers: Subparser group from parent ArgumentParser.

    Returns:
        The prepared subparser for configure.
    """
    parser = subparsers.add_parser(
        "prepare-dataset",
        description="Prepare a curated research dataset for a location and time range.",
        help="Prepare a curated research dataset",
    )

    # Location options
    loc_group = parser.add_argument_group("Location")
    loc_group.add_argument(
        "--location", "-l",
        type=str,
        choices=["copenhagen", "stockholm", "berlin"],
        help="Preset location (overrides --lat/--lon)",
    )
    loc_group.add_argument(
        "--lat",
        type=float,
        help="Latitude (required if --location not specified)",
    )
    loc_group.add_argument(
        "--lon",
        type=float,
        help="Longitude (required if --location not specified)",
    )

    # Time range options
    time_group = parser.add_argument_group("Time range")
    time_group.add_argument(
        "--years", "-y",
        type=int,
        default=2,
        help="Number of years of past data to include (default: 2)",
    )
    time_group.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD, overrides --years)",
    )
    time_group.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD, defaults to today)",
    )

    # Variables and models
    var_group = parser.add_argument_group("Variables & models")
    var_group.add_argument(
        "--variables", "-v",
        type=str,
        default="all",
        help="Comma-separated variable names (or 'all', 'standard', 'minimal')",
    )
    var_group.add_argument(
        "--models", "-m",
        type=str,
        default=None,
        help="Comma-separated model names (or 'all' for all defaults)",
    )

    # Pipeline options
    pipe_group = parser.add_argument_group("Pipeline options")
    pipe_group.add_argument(
        "--build-metrics", "-M",
        action="store_true",
        help="Precompute metrics at multiple granularities",
    )
    pipe_group.add_argument(
        "--build-events", "-E",
        action="store_true",
        help="Precompute event datasets",
    )
    pipe_group.add_argument(
        "--build-aggregations", "-A",
        action="store_true",
        help="Compute lead-time degradation and seasonal summaries",
    )
    pipe_group.add_argument(
        "--build-all", "-B",
        action="store_true",
        help="Enable all build options (metrics, events, aggregations)",
    )
    pipe_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing data if available",
    )
    pipe_group.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    pipe_group.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./datasets",
        help="Output directory for datasets (default: ./datasets)",
    )

    # Dataset info subcommand
    info_parser = subparsers.add_parser(
        "dataset-info",
        description="Display information about an existing dataset.",
        help="Show dataset information",
    )
    info_parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset directory",
    )

    parser.set_defaults(handler=_handle_prepare_dataset)
    return parser


def _handle_prepare_dataset(args: argparse.Namespace) -> int:
    """Handler for the prepare-dataset subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    try:
        # Resolve location
        if args.location:
            from datasets.config import LOCATION_PRESETS
            location = LOCATION_PRESETS[args.location]
        elif args.lat is not None and args.lon is not None:
            location = {
                "name": f"custom_{args.lat}_{args.lon}",
                "lat": args.lat,
                "lon": args.lon,
            }
        else:
            print("Error: Must specify --location or both --lat and --lon", file=sys.stderr)
            return 1

        # Resolve time range
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        if args.start and args.end:
            start_date = args.start
            end_date = args.end
        else:
            end_date = now.strftime("%Y-%m-%d")
            start = now.replace(year=now.year - args.years)
            start_date = start.strftime("%Y-%m-%d")

        # Resolve variables
        from datasets.config import VARIABLE_SETS
        if args.variables == "all":
            variables = VARIABLE_SETS["all"]
        elif args.variables in VARIABLE_SETS:
            variables = VARIABLE_SETS[args.variables]
        else:
            variables = tuple(v.strip() for v in args.variables.split(","))

        # Resolve models
        if args.models:
            if args.models == "all":
                from datasets.config import DEFAULT_MODELS
                models = list(DEFAULT_MODELS.get(location["name"].lower(), ["icon_eu"]))
            else:
                models = [m.strip() for m in args.models.split(",")]
        else:
            from datasets.config import DEFAULT_MODELS
            models = list(DEFAULT_MODELS.get(location["name"].lower(), ["icon_eu"]))

        # Resolve build options
        build_metrics = args.build_metrics or args.build_all
        build_events = args.build_events or args.build_all
        build_aggregations = args.build_aggregations or args.build_all

        # Resolve output directory
        output_dir = args.output_dir if args.output_dir else "./datasets"

        # Configure logging
        log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        # Run pipeline
        from datasets.prepare import run_pipeline

        print()
        print(f"Preparing dataset: {location['name']} ({location['lat']}, {location['lon']})")
        print(f"Time range: {start_date} → {end_date}")
        print(f"Variables: {', '.join(variables)}")
        print(f"Models: {', '.join(models)}")
        print(f"Build metrics: {build_metrics}")
        print(f"Build events: {build_events}")
        print(f"Build aggregations: {build_aggregations}")
        print()

        dataset_path = run_pipeline(
            location=location,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            models=models,
            build_metrics=build_metrics,
            build_events=build_events,
            build_aggregations=build_aggregations,
            output_dir=output_dir,
        )

        print()
        print(f"Dataset created: {dataset_path}")
        return 0

    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _handle_dataset_info(args: argparse.Namespace) -> int:
    """Handler for the dataset-info subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    try:
        from datasets.manifest import read_manifest, validate_manifest
        from datasets.progress import ProgressReporter

        progress = ProgressReporter()
        dataset_dir = Path(args.dataset_path)

        if not dataset_dir.exists():
            print(f"Error: Dataset directory not found: {dataset_dir}", file=sys.stderr)
            return 1

        manifest = read_manifest(dataset_dir)
        if manifest is None:
            print(f"Error: No manifest found in {dataset_dir}/manifests/", file=sys.stderr)
            return 1

        # Validate
        issues = validate_manifest(manifest)
        if issues:
            progress.warning("Validation issues")
            for issue in issues:
                progress.warning(issue.message, f"[{issue.check}]")

        # Display info
        progress.section("Dataset Information")
        progress.summary(
            Name=manifest.get("dataset_name", "unknown"),
            Version=manifest.get("version", "N/A"),
            Location=f"{manifest.get('location', {}).get('name', 'N/A')} "
                     f"({manifest.get('location', {}).get('latitude', '?')}, "
                     f"{manifest.get('location', {}).get('longitude', '?')})",
            TimeRange=f"{manifest.get('time_range', {}).get('start_date', '?')} → "
                      f"{manifest.get('time_range', {}).get('end_date', '?')}",
            Variables=", ".join(manifest.get("variables", [])),
            Models=", ".join(manifest.get("models", [])),
            Generated=manifest.get("generated_at", "?"),
            PipelineSteps=", ".join(manifest.get("pipeline", {}).get("steps_completed", [])),
        )

        # Statistics
        stats = manifest.get("statistics", {})
        progress.section("Statistics")
        progress.summary(
            ForecastRows=f"{stats.get('forecast_rows', 0):,}",
            ObservationRows=f"{stats.get('observation_rows', 0):,}",
            VerificationRows=f"{stats.get('verification_rows', 0):,}",
            EventDatasets=", ".join(stats.get("event_datasets", [])),
            MetricGroups=", ".join(stats.get("metric_groups", [])),
        )

        return 0

    except Exception as e:
        logger.exception("dataset-info failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1
