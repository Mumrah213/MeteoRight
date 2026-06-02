"""Verification Builder CLI.

Builds a canonical weather forecast verification dataset by:
  1. Loading forecast and observation parquet files (partition-aware)
  2. LEFT JOINing forecasts to observations on target_time == observation_time
  3. Preserving forecast provenance (issue_time, target_time, model)
  4. Computing error columns (forecast - observed)
  5. Running validation checks
  6. Saving the canonical verification dataset

Usage:
    python -m weather_verifier.cli \
        --data-dir ./data \
        --model ecmwf_ifs_hres \
        --location 55.605,13.003 \
        --start 2025-01-01 \
        --end 2025-03-01 \
        --variables temperature_2m,precipitation \
        --output ./data/verification
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console

from . import __version__
from .aligner import align_forecasts_with_observations
from .errors import compute_error_columns
from .loaders import load_forecasts, load_observations
from .provenance import verify_provenance_integrity
from .schema import canonical_column_order
from .validator import VerificationValidator, save_validation_report

console = Console()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a canonical weather forecast verification dataset.",
        epilog=(
            "Example:\n"
            "  python -m weather_verifier.cli \\\n"
            "    --data-dir ./data \\\n"
            "    --model ecmwf_ifs_hres \\\n"
            "    --location 55.605,13.003 \\\n"
            "    --start 2025-01-01 \\\n"
            "    --end 2025-03-01 \\\n"
            "    --output ./data/verification\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing forecasts/ and observations/ partitions",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g. ecmwf_ifs_hres)",
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="Latitude,Longitude (e.g. 55.605,13.003)",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default="temperature_2m,precipitation",
        help=("Comma-separated variable names (default: temperature_2m,precipitation)"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for verification dataset",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    # Parse location
    try:
        lat_str, lon_str = args.location.split(",")
        args.latitude = float(lat_str.strip())
        args.longitude = float(lon_str.strip())
    except (ValueError, AttributeError):
        parser.error(f"Invalid --location format: {args.location} — expected Latitude,Longitude")

    # Parse variables
    args.variables = tuple(v.strip() for v in args.variables.split(","))

    # Validate dates
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        parser.error("Dates must be in YYYY-MM-DD format")

    return args


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full verification pipeline.

    Pipeline:
      1. Load forecasts (partition-aware, date-filtered)
      2. Load observations (partition-aware, date-filtered)
      3. Validate locations match
      4. LEFT JOIN forecasts to observations on target_time == observation_time
      5. Verify provenance integrity
      6. Build canonical column schema
      7. Compute error columns
      8. Run validation checks
      9. Save verification dataset + validation report
    """
    data_dir: Path = args.data_dir
    model: str = args.model
    lat: float = args.latitude
    lon: float = args.longitude
    start: str = args.start
    end: str = args.end
    variables: tuple[str, ...] = args.variables
    output_dir: Path = args.output

    console.print(f"[bold]Weather Forecast Verification Tool v{__version__}[/bold]")
    console.print()

    # Step 1: Load forecasts
    console.rule("[blue]Step 1: Loading forecasts[/blue]")
    forecasts = load_forecasts(data_dir, model, start, end)
    if forecasts.empty:
        console.print("[red]No forecast data found. Exiting.[/red]")
        sys.exit(1)

    # Filter to requested location
    loc_match = (forecasts["latitude"] == lat) & (forecasts["longitude"] == lon)
    forecasts = forecasts[loc_match].reset_index(drop=True)
    console.print(f"  Location ({lat}, {lon}): {len(forecasts)} rows")

    # Step 2: Load observations
    console.rule("[blue]Step 2: Loading observations[/blue]")
    observations = load_observations(data_dir, start, end)
    if observations.empty:
        console.print("[red]No observation data found. Exiting.[/red]")
        sys.exit(1)

    # Filter to requested location
    loc_match = (observations["latitude"] == lat) & (observations["longitude"] == lon)
    observations = observations[loc_match].reset_index(drop=True)
    console.print(f"  Location ({lat}, {lon}): {len(observations)} rows")

    # Step 3: Align (LEFT JOIN)
    console.rule("[blue]Step 3: Aligning forecasts with observations[/blue]")
    aligned = align_forecasts_with_observations(forecasts, observations, variables)
    matched = aligned["observation_time"].notna().sum()
    console.print(f"  Aligned: {len(aligned)} rows, {matched} with observations")

    # Step 4: Verify provenance
    console.rule("[blue]Step 4: Verifying provenance integrity[/blue]")
    verify_provenance_integrity(aligned)
    console.print("  [green]Provenance OK[/green]")

    # Step 5: Build canonical column order
    console.rule("[blue]Step 5: Building canonical schema[/blue]")
    expected_cols = canonical_column_order(variables)
    # Ensure all expected columns exist (add missing ones as NaN)
    for col in expected_cols:
        if col not in aligned.columns:
            if "forecast" in col or "observed" in col or "error" in col:
                aligned[col] = pd.NA
    aligned = aligned[expected_cols]
    console.print(f"  Columns: {len(expected_cols)}")

    # Step 6: Compute error columns
    console.rule("[blue]Step 6: Computing error columns[/blue]")
    aligned = compute_error_columns(aligned, variables)

    # Step 7: Validation
    console.rule("[blue]Step 7: Running validation checks[/blue]")
    validator = VerificationValidator(variables)
    issues = validator.validate(aligned)
    for issue in issues:
        icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[issue.severity]
        console.print(f"  {icon} [{issue.severity:>6}] {issue.check}: {issue.message}")

    # Step 8: Save output
    console.rule("[blue]Step 8: Saving output[/blue]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save verification dataset
    date_slug = f"{start}_to_{end}"
    output_path = output_dir / f"verification_{date_slug}.parquet"
    aligned.to_parquet(output_path, engine="pyarrow", index=False)
    console.print(f"  Saved: {output_path} ({len(aligned)} rows)")

    # Save validation report
    report_path = output_dir / "validation_report.json"
    save_validation_report(issues, report_path)
    console.print(f"  Saved: {report_path}")

    console.print()
    console.print("[bold green]Verification dataset built successfully.[/bold green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        run_pipeline(args)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
