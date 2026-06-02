"""CLI for forecast verification metrics computation.

Usage:
    python -m weather_metrics.cli \
        --verification ./data/verification/*.parquet \
        --group-by lead_hours \
        --metrics mae,rmse,bias \
        --variables temperature_2m \
        --output ./data/metrics

Example aggregations:
    # By lead time
    python -m weather_metrics.cli \
        --verification ./data/verification/*.parquet \
        --group-by lead_hours \
        --metrics mae,rmse,bias \
        --variables temperature_2m,precipitation \
        --output ./data/metrics

    # By model and lead time
    python -m weather_metrics.cli \
        --verification ./data/verification/*.parquet \
        --group-by model,lead_hours \
        --metrics mae,rmse,bias,std \
        --variables temperature_2m \
        --output ./data/metrics

    # By season
    python -m weather_metrics.cli \
        --verification ./data/verification/*.parquet \
        --group-by season \
        --metrics mae,rmse,bias \
        --variables temperature_2m \
        --output ./data/metrics
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .io import export_csv, read_verification, write_metrics
from .schemas import METRIC_NAME_COL
from .validation import validate_metrics

console = Console()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_VALID_METRICS = {"mae", "rmse", "bias", "std"}
_VALID_DIMENSIONS = {"lead_hours", "model", "month", "season"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute forecast verification metrics from canonical verification parquet.",
        epilog=(
            "Examples:\n"
            "  python -m weather_metrics.cli \\\n"
            "    --verification ./data/verification/*.parquet \\\n"
            "    --group-by lead_hours \\\n"
            "    --output ./data/metrics\n"
            "\n"
            "  python -m weather_metrics.cli \\\n"
            "    --verification ./data/verification/*.parquet \\\n"
            "    --group-by model,lead_hours \\\n"
            "    --metrics mae,rmse,bias,std \\\n"
            "    --variables temperature_2m,precipitation \\\n"
            "    --output ./data/metrics\n"
        ),
    )

    parser.add_argument(
        "--verification",
        type=str,
        required=True,
        help=(
            "Glob pattern or path to verification parquet files "
            "(e.g. 'data/verification/verification_*.parquet')"
        ),
    )

    parser.add_argument(
        "--group-by",
        type=str,
        required=True,
        help=(f"Comma-separated groupby dimensions. Valid: {', '.join(sorted(_VALID_DIMENSIONS))}"),
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="mae,rmse,bias",
        help="Comma-separated metric names (default: mae,rmse,bias)",
    )

    parser.add_argument(
        "--variables",
        type=str,
        default="temperature_2m,precipitation",
        help="Comma-separated variable names (default: temperature_2m,precipitation)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for metrics parquet files",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    # Parse group-by
    args.group_by = [g.strip() for g in args.group_by.split(",")]

    # Parse metrics
    args.metrics = [m.strip() for m in args.metrics.split(",")]

    # Parse variables
    args.variables = [v.strip() for v in args.variables.split(",")]

    # Validate group-by dimensions
    for dim in args.group_by:
        if dim not in _VALID_DIMENSIONS:
            parser.error(
                f"Invalid group-by dimension '{dim}'. "
                f"Valid dimensions: {', '.join(sorted(_VALID_DIMENSIONS))}"
            )

    # Validate metrics
    for m in args.metrics:
        if m not in _VALID_METRICS:
            parser.error(
                f"Unknown metric '{m}'. Valid metrics: {', '.join(sorted(_VALID_METRICS))}"
            )

    return args


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the metrics computation pipeline.

    Pipeline:
      1. Read verification parquet files (glob-aware)
      2. Compute aggregated metrics
      3. Validate metric outputs
      4. Write metrics to parquet
      5. Print summary
    """
    from .aggregations import aggregate_metrics

    verification_paths: str = args.verification
    groupby: list[str] = args.group_by
    metrics: list[str] = args.metrics
    variables: list[str] = args.variables
    output_dir: Path = args.output

    console.print(f"[bold]Weather Forecast Verification Metrics v{__version__}[/bold]")
    console.print()

    # Step 1: Read verification data
    console.rule("[blue]Step 1: Reading verification data[/blue]")
    try:
        df = read_verification(verification_paths)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(f"  Loaded {len(df)} rows from verification files")
    console.print(f"  Variables: {', '.join(df.columns)}")

    # Step 2: Compute aggregated metrics
    console.rule("[blue]Step 2: Computing metrics[/blue]")
    console.print(f"  Group by: {', '.join(groupby)}")
    console.print(f"  Metrics: {', '.join(metrics)}")
    console.print(f"  Variables: {', '.join(variables)}")

    result = aggregate_metrics(
        df,
        groupby=groupby,
        metrics=metrics,
        variables=variables,
    )

    console.print(f"  Computed {len(result)} metric rows")

    # Step 3: Validate outputs
    console.rule("[blue]Step 3: Validating outputs[/blue]")
    issues = validate_metrics(result, variables=variables)

    for issue in issues:
        icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[issue.severity]
        console.print(f"  {icon} [{issue.severity:>6}] {issue.check}: {issue.message}")

    # Check for errors
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        console.print()
        console.print("[bold red]Validation errors found. Review the report.[/bold red]")

    # Step 4: Write outputs
    console.rule("[blue]Step 4: Writing outputs[/blue]")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write per-variable parquet files
    for var in variables:
        var_result = result[result["variable"] == var]
        if var_result.empty:
            console.print(f"  [yellow]No data for variable '{var}'[/yellow]")
            continue

        # Build dimension string for filename
        dim_str = "_".join(groupby)
        metric_str = "_".join(metrics)
        filename = f"{var}_{metric_str}_by_{dim_str}.parquet"
        output_path = output_dir / filename

        write_metrics(
            var_result,
            output_path,
            metadata={
                "groupby": groupby,
                "metrics": metrics,
                "variable": var,
                "source_verification": verification_paths,
            },
        )

        # Also export CSV for quick inspection
        csv_path = output_dir / f"{var}_{metric_str}_by_{dim_str}.csv"
        export_csv(var_result, csv_path)

        console.print(f"  ✓ {filename} ({len(var_result)} rows)")

    # Step 5: Summary
    console.rule("[green]Summary[/green]")
    console.print()

    # Print a readable summary table
    if not result.empty:
        # Pivot for readability
        pivot_cols = [c for c in groupby if c in result.columns]
        if pivot_cols and "lead_hours" in result.columns:
            pivot = result.pivot_table(
                index="lead_hours",
                columns=["variable", METRIC_NAME_COL],
                values="metric_value",
                aggfunc="first",
            )
            console.print(pivot)

    console.print()
    console.print("[bold green]Metrics computation complete.[/bold green]")


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
