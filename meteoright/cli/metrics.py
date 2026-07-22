"""`meteoright metrics` — aggregate verification rows into metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._shared import console


def cmd_metrics(args: argparse.Namespace) -> int:
    """Compute aggregated metrics."""
    from metrics.aggregations import aggregate_metrics
    from metrics.io import read_verification, write_metrics
    from metrics.validation import validate_metrics

    console.print("[bold]Computing metrics[/bold]")

    verification_df = read_verification(args.verification)
    console.print(f"  Loaded {len(verification_df)} verification rows")

    group_dims = [d.strip() for d in args.group_by.split(",")]
    metric_names = [m.strip() for m in args.metrics.split(",")]
    variables = [v.strip() for v in args.variables.split(",")]

    console.print(f"  Group by: {group_dims}")
    console.print(f"  Metrics: {metric_names}")
    console.print(f"  Variables: {variables}")

    metrics_df = aggregate_metrics(
        verification_df,
        groupby=group_dims,
        metrics=metric_names,
        variables=variables,
    )

    console.print(f"  Computed {len(metrics_df)} metric rows")

    # Validate
    issues = validate_metrics(metrics_df)
    for issue in issues:
        if issue.severity == "error":
            console.print(f"  [red]✗ {issue.check}: {issue.message}[/red]")
        elif issue.severity == "warning":
            console.print(f"  [yellow]⚠ {issue.check}: {issue.message}[/yellow]")

    # Write output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for var in variables:
        var_df = metrics_df[metrics_df["variable"] == var]
        if var_df.empty:
            continue
        filename = f"{var}_{'_'.join(metric_names)}_by_{'_'.join(group_dims)}.parquet"
        path = output_dir / filename
        write_metrics(var_df, path)
        console.print(f"  [green]✓[/green] {filename}")

    return 0
