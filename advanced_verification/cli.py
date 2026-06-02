"""CLI commands for Advanced Verification & Forecast Skill Analysis.

Commands:
    events    Generate binary event flags from verification data
    skill     Compute confusion matrix and skill scores
    compare   Compare skill scores between models for the same event

Usage
-----
# Generate frost events and compute skill by lead time
weather-analyzer advanced events \
    --verification data/verification/verification.parquet \
    --event frost \
    --group-by lead_hours \
    --output data/advanced_verification

# Compute confusion matrix and skill scores for heavy precipitation
weather-analyzer advanced skill \
    --verification data/verification/verification.parquet \
    --event heavy_precip \
    --group-by lead_hours,model \
    --output data/advanced_verification

# Compare CSI between models for frost events
weather-analyzer advanced compare \
    --verification data/verification/verification.parquet \
    --event frost \
    --models icon_eu,gfs \
    --group-by lead_hours \
    --metric csi \
    --output data/advanced_verification
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from rich.console import Console

# Import phase 5 modules
from .confusion import compute_confusion_matrix
from .events import generate_events
from .io import load_verification, write_confusion, write_skill_scores
from .schema import ALL_SKILL_METRICS, BUILTIN_EVENTS
from .skill_scores import compute_skill_scores
from .validation import validate_confusion, validate_skill_scores

console = Console()
logger = logging.getLogger("weather_analyzer")


def cmd_advanced_events(args: argparse.Namespace) -> int:
    """Generate binary event flags from a verification dataset.

    Args:
        args: Parsed arguments with:
            --verification: Path to verification parquet
            --event: Built-in event name (frost, heavy_precip, hot_day, no_rain)
            --variable: Variable name (for custom events)
            --threshold: Threshold value (for custom events)
            --operator: Comparison operator (for custom events)
            --group-by: Dimensions for output grouping
            --output: Output directory

    Returns:
        0 on success, 1 on error.
    """
    console.print("[bold]Generating events[/bold]")

    # Load verification data
    verification_df = load_verification(args.verification)
    console.print(f"  Loaded {len(verification_df)} verification rows")

    # Resolve event definition
    event_name = args.event
    variable = args.variable
    threshold = args.threshold
    operator = args.operator

    # Generate event columns
    events_df = generate_events(
        verification_df,
        event_name=event_name,
        variable=variable,
        threshold=threshold,
        operator=operator,
    )

    # Save enriched DataFrame
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    event_col_fcst = f"event_forecast_{event_name}"
    event_col_obs = f"event_observed_{event_name}"
    fcst_count = int(events_df[event_col_fcst].sum())
    obs_count = int(events_df[event_col_obs].sum())

    path = output_dir / f"{event_name}_events.parquet"
    events_df.to_parquet(path, index=False)
    console.print(f"  [green]✓[/green] Wrote {path}")
    console.print(f"  Forecast event: {fcst_count}/{len(events_df)}")
    console.print(f"  Observed event:  {obs_count}/{len(events_df)}")

    # Also write confusion and skill if requested
    groupby = [d.strip() for d in args.group_by.split(",")] if args.group_by else None

    # Confusion matrix
    console.rule("[blue]Confusion Matrix[/blue]")
    confusion_df = compute_confusion_matrix(events_df, event_name=event_name, groupby=groupby)

    # Validate
    issues = validate_confusion(confusion_df)
    for issue in issues:
        if issue.severity == "error":
            console.print(f"  [red]✗ {issue.check}: {issue.message}[/red]")
        elif issue.severity == "warning":
            console.print(f"  [yellow]⚠ {issue.check}: {issue.message}[/yellow]")

    path = output_dir / f"{event_name}_confusion.parquet"
    write_confusion(confusion_df, path, event_name=event_name)
    console.print(f"  [green]✓[/green] {path}")
    console.print(f"  {len(confusion_df)} group(s)")

    # Skill scores
    console.rule("[blue]Skill Scores[/blue]")
    skill_df = compute_skill_scores(confusion_df)

    # Validate
    issues = validate_skill_scores(skill_df)
    for issue in issues:
        if issue.severity == "info":
            console.print(f"  [dim]i {issue.check}: {issue.message}[/dim]")

    path = output_dir / f"{event_name}_skill_scores.parquet"
    write_skill_scores(skill_df, path, event_name=event_name)
    console.print(f"  [green]✓[/green] {path}")
    console.print(f"  {len(skill_df)} metric rows")

    # Print summary table
    console.print()
    for metric in ["hit_rate", "false_alarm_rate", "precision", "csi"]:
        metric_df = skill_df[skill_df["metric_name"] == metric]
        if metric_df.empty:
            continue
        console.print(f"  [bold]{metric}:[/bold]")
        for _, row in metric_df.iterrows():
            val = row["metric_value"]
            display = f"{val:.4f}" if pd.notna(val) else "NaN"
            groups = " ".join(
                f"{col}={row[col]}" for col in groupby if col in row.index and pd.notna(row[col])
            )
            console.print(f"    {display:>8}  {groups}")

    return 0


def cmd_advanced_skill(args: argparse.Namespace) -> int:
    """Compute confusion matrix and skill scores.

    Similar to events but skips event generation (assumes events are pre-computed).

    Args:
        args: Parsed arguments with:
            --verification: Path to verification parquet
            --event: Built-in event name
            --group-by: Dimensions for grouping
            --output: Output directory
            --variables: Variables to include (for event generation)
            --threshold: Custom threshold (for custom events)
            --operator: Custom operator (for custom events)

    Returns:
        0 on success, 1 on error.
    """
    console.print("[bold]Computing skill scores[/bold]")

    verification_df = load_verification(args.verification)
    console.print(f"  Loaded {len(verification_df)} verification rows")

    # Generate events if needed
    event_name = args.event
    variable = args.variable
    threshold = args.threshold
    operator = args.operator

    events_df = generate_events(
        verification_df,
        event_name=event_name,
        variable=variable,
        threshold=threshold,
        operator=operator,
    )

    # Groupby dimensions
    groupby = [d.strip() for d in args.group_by.split(",")] if args.group_by else None

    # Confusion matrix
    console.rule("[blue]Confusion Matrix[/blue]")
    confusion_df = compute_confusion_matrix(events_df, event_name=event_name, groupby=groupby)
    issues = validate_confusion(confusion_df)
    for issue in issues:
        if issue.severity == "error":
            console.print(f"  [red]✗ {issue.check}: {issue.message}[/red]")
        elif issue.severity == "warning":
            console.print(f"  [yellow]⚠ {issue.check}: {issue.message}[/yellow]")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{event_name}_confusion.parquet"
    write_confusion(confusion_df, path, event_name=event_name)
    console.print(f"  [green]✓[/green] {path}")

    # Skill scores
    console.rule("[blue]Skill Scores[/blue]")
    skill_df = compute_skill_scores(confusion_df)

    issues = validate_skill_scores(skill_df)
    for issue in issues:
        if issue.severity == "info":
            console.print(f"  [dim]i {issue.check}: {issue.message}[/dim]")

    path = output_dir / f"{event_name}_skill_scores.parquet"
    write_skill_scores(skill_df, path, event_name=event_name)
    console.print(f"  [green]✓[/green] {path}")
    console.print(f"  {len(skill_df)} metric rows")

    return 0


def cmd_advanced_compare(args: argparse.Namespace) -> int:
    """Compare skill scores between models for the same event.

    Args:
        args: Parsed arguments with:
            --verification: Path to verification parquet
            --event: Built-in event name
            --models: Comma-separated model names (e.g., "icon_eu,gfs")
            --group-by: Dimensions for grouping
            --metric: Metric to compare (csi, hit_rate, etc.)
            --output: Output directory

    Returns:
        0 on success, 1 on error.
    """
    console.print("[bold]Comparing model skill[/bold]")

    verification_df = load_verification(args.verification)
    models = [m.strip() for m in args.models.split(",")]
    event_name = args.event
    metric = args.metric
    groupby = [d.strip() for d in args.group_by.split(",")] if args.group_by else None

    console.print(f"  Event: {event_name}")
    console.print(f"  Models: {', '.join(models)}")
    console.print(f"  Metric: {metric}")
    console.print(f"  Group by: {groupby}")

    # Compute skill for each model
    comparison_rows = []
    for model in models:
        # Filter to this model
        model_df = verification_df[verification_df["model"] == model]
        if model_df.empty:
            console.print(f"  [yellow]⚠ No data for model '{model}'[/yellow]")
            continue

        # Generate events
        events_df = generate_events(model_df, event_name=event_name)

        # Confusion matrix (grouped by model + other dims)
        model_groupby = groupby + ["model"] if groupby else ["model"]
        confusion_df = compute_confusion_matrix(
            events_df, event_name=event_name, groupby=model_groupby
        )

        # Skill scores
        skill_df = compute_skill_scores(confusion_df)
        metric_df = skill_df[skill_df["metric_name"] == metric]

        for _, row in metric_df.iterrows():
            comparison_rows.append(
                {
                    "model": model,
                    "event_name": event_name,
                    "metric_name": metric,
                    "metric_value": row["metric_value"],
                    **{col: row[col] for col in groupby if col in row.index},
                }
            )

    if not comparison_rows:
        console.print("[red]No comparison data produced.[/red]")
        return 1

    comparison_df = pd.DataFrame(comparison_rows)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"compare_{event_name}_{metric}.parquet"
    comparison_df.to_parquet(path, index=False)
    console.print(f"  [green]✓[/green] {path}")

    # Print comparison table
    console.print()
    console.print(f"  [bold]{metric} comparison:[/bold]")
    for _, row in comparison_df.iterrows():
        val = row["metric_value"]
        display = f"{val:.4f}" if pd.notna(val) else "NaN"
        groups = " ".join(
            f"{col}={row[col]}" for col in groupby if col in row.index and pd.notna(row[col])
        )
        console.print(f"    {row['model']:12s}  {display:>8}  {groups}")

    return 0


def register_advanced_subcommands(parser: argparse.ArgumentParser) -> None:
    """Register advanced verification subcommands on the main argument parser.

    Args:
        parser: The main ArgumentParser instance.
    """
    advanced = parser.add_subparsers(dest="command")

    # 'advanced events' command
    ae = advanced.add_parser("events", help="Generate events and compute skill scores")
    ae.add_argument("--verification", required=True, help="Verification parquet path")
    ae.add_argument("--event", choices=BUILTIN_EVENTS, help="Built-in event name")
    ae.add_argument("--variable", help="Variable name (for custom events)")
    ae.add_argument("--threshold", type=float, help="Threshold value (for custom events)")
    ae.add_argument(
        "--operator",
        choices=["le", "lt", "ge", "gt", "eq", "ne"],
        help="Comparison operator (for custom events)",
    )
    ae.add_argument("--group-by", help="Groupby dimensions (comma-separated)")
    ae.add_argument("--output", required=True, help="Output directory")

    # 'advanced skill' command
    asub = advanced.add_parser("skill", help="Compute confusion matrix and skill scores")
    asub.add_argument("--verification", required=True, help="Verification parquet path")
    asub.add_argument("--event", choices=BUILTIN_EVENTS, help="Built-in event name")
    asub.add_argument("--variable", help="Variable name (for custom events)")
    asub.add_argument("--threshold", type=float, help="Threshold value (for custom events)")
    asub.add_argument(
        "--operator",
        choices=["le", "lt", "ge", "gt", "eq", "ne"],
        help="Comparison operator (for custom events)",
    )
    asub.add_argument("--group-by", help="Groupby dimensions (comma-separated)")
    asub.add_argument("--output", required=True, help="Output directory")

    # 'advanced compare' command
    ac = advanced.add_parser("compare", help="Compare model skill scores")
    ac.add_argument("--verification", required=True, help="Verification parquet path")
    ac.add_argument("--event", choices=BUILTIN_EVENTS, help="Built-in event name")
    ac.add_argument("--models", required=True, help="Model names (comma-separated)")
    ac.add_argument("--group-by", help="Groupby dimensions (comma-separated)")
    ac.add_argument("--metric", default="csi", choices=ALL_SKILL_METRICS, help="Metric to compare")
    ac.add_argument("--output", required=True, help="Output directory")
