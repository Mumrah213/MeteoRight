"""`meteoright advanced` — event verification, skill scores and model comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ._shared import console


def cmd_advanced_events(args: argparse.Namespace) -> int:
    """Generate events and compute skill scores."""
    from advanced_verification.confusion import compute_confusion_matrix
    from advanced_verification.events import generate_events
    from advanced_verification.io import load_verification, write_confusion, write_skill_scores
    from advanced_verification.skill_scores import compute_skill_scores
    from advanced_verification.validation import validate_confusion, validate_skill_scores

    # Load verification data
    verification_df = load_verification(args.verification)
    console.print(f"  Loaded {len(verification_df)} verification rows")

    # Generate events
    events_df = generate_events(
        verification_df,
        event_name=args.event,
        variable=args.variable,
        threshold=args.threshold,
        operator=args.operator,
    )

    # Groupby
    groupby = [d.strip() for d in args.group_by.split(",")] if args.group_by else None

    # Confusion matrix
    confusion_df = compute_confusion_matrix(events_df, event_name=args.event, groupby=groupby)
    issues = validate_confusion(confusion_df)
    for issue in issues:
        if issue.severity == "error":
            console.print(f"  [red]✗ {issue.check}: {issue.message}[/red]")
        elif issue.severity == "warning":
            console.print(f"  [yellow]⚠ {issue.check}: {issue.message}[/yellow]")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{args.event}_confusion.parquet"
    write_confusion(confusion_df, path, event_name=args.event)
    console.print(f"  [green]✓[/green] {path}")

    # Skill scores
    skill_df = compute_skill_scores(confusion_df)
    issues = validate_skill_scores(skill_df)
    for issue in issues:
        if issue.severity == "info":
            console.print(f"  [dim]i {issue.check}: {issue.message}[/dim]")

    path = output_dir / f"{args.event}_skill_scores.parquet"
    write_skill_scores(skill_df, path, event_name=args.event)
    console.print(f"  [green]✓[/green] {path}")

    # Print summary
    console.print()
    for metric in ["hit_rate", "false_alarm_rate", "precision", "csi"]:
        metric_df = skill_df[skill_df["metric_name"] == metric]
        if metric_df.empty:
            continue
        console.print(f"  [bold]{metric}:[/bold]")
        for _, row in metric_df.iterrows():
            val = row["metric_value"]
            display = f"{val:.4f}" if pd.notna(val) else "NaN"
            console.print(f"    {display:>8}")

    return 0


def cmd_advanced_skill(args: argparse.Namespace) -> int:
    """Compute confusion matrix and skill scores."""
    from advanced_verification.confusion import compute_confusion_matrix
    from advanced_verification.events import generate_events
    from advanced_verification.io import load_verification, write_confusion, write_skill_scores
    from advanced_verification.skill_scores import compute_skill_scores
    from advanced_verification.validation import validate_confusion, validate_skill_scores

    verification_df = load_verification(args.verification)
    console.print(f"  Loaded {len(verification_df)} verification rows")

    events_df = generate_events(
        verification_df,
        event_name=args.event,
        variable=args.variable,
        threshold=args.threshold,
        operator=args.operator,
    )

    groupby = [d.strip() for d in args.group_by.split(",")] if args.group_by else None

    confusion_df = compute_confusion_matrix(events_df, event_name=args.event, groupby=groupby)
    issues = validate_confusion(confusion_df)
    for issue in issues:
        if issue.severity == "error":
            console.print(f"  [red]✗ {issue.check}: {issue.message}[/red]")
        elif issue.severity == "warning":
            console.print(f"  [yellow]⚠ {issue.check}: {issue.message}[/yellow]")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{args.event}_confusion.parquet"
    write_confusion(confusion_df, path, event_name=args.event)
    console.print(f"  [green]✓[/green] {path}")

    skill_df = compute_skill_scores(confusion_df)
    issues = validate_skill_scores(skill_df)
    for issue in issues:
        if issue.severity == "info":
            console.print(f"  [dim]i {issue.check}: {issue.message}[/dim]")

    path = output_dir / f"{args.event}_skill_scores.parquet"
    write_skill_scores(skill_df, path, event_name=args.event)
    console.print(f"  [green]✓[/green] {path}")

    return 0


def cmd_advanced_compare(args: argparse.Namespace) -> int:
    """Compare model skill scores."""
    from advanced_verification.confusion import compute_confusion_matrix
    from advanced_verification.events import generate_events
    from advanced_verification.io import load_verification
    from advanced_verification.skill_scores import compute_skill_scores

    verification_df = load_verification(args.verification)
    models = [m.strip() for m in args.models.split(",")]
    event_name = args.event
    metric = args.metric
    groupby = [d.strip() for d in args.group_by.split(",")] if args.group_by else None

    console.print(f"  Event: {event_name}")
    console.print(f"  Models: {', '.join(models)}")
    console.print(f"  Metric: {metric}")

    comparison_rows = []
    for model in models:
        model_df = verification_df[verification_df["model"] == model]
        if model_df.empty:
            console.print(f"  [yellow]⚠ No data for model '{model}'[/yellow]")
            continue

        events_df = generate_events(model_df, event_name=event_name)
        model_groupby = groupby + ["model"] if groupby else ["model"]
        confusion_df = compute_confusion_matrix(
            events_df, event_name=event_name, groupby=model_groupby
        )
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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"compare_{event_name}_{metric}.parquet"
    comparison_df.to_parquet(path, index=False)
    console.print(f"  [green]✓[/green] {path}")

    console.print()
    console.print(f"  [bold]{metric} comparison:[/bold]")
    for _, row in comparison_df.iterrows():
        val = row["metric_value"]
        display = f"{val:.4f}" if pd.notna(val) else "NaN"
        console.print(f"    {row['model']:12s}  {display:>8}")

    return 0
