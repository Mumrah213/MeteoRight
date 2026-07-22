"""`meteoright pipeline` — run download, verify, metrics and analysis end to end."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._shared import console
from .analyze import cmd_analyze
from .download import cmd_download
from .metrics import cmd_metrics
from .verify import cmd_verify


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run the full pipeline."""
    console.print("[bold]Running full pipeline[/bold]")

    data_dir = Path(args.output) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download observations
    console.rule("[blue]Step 1: Download observations[/blue]")
    obs_args = argparse.Namespace(
        lat=args.lat,
        lon=args.lon,
        start=args.start,
        end=args.end,
        type="observations",
        variables=args.variables,
        model=None,
        output=str(data_dir),
    )
    if cmd_download(obs_args) != 0:
        return 1

    # Step 2: Download forecasts
    console.rule("[blue]Step 2: Download forecasts[/blue]")
    fcst_args = argparse.Namespace(
        lat=args.lat,
        lon=args.lon,
        start=args.start,
        end=args.end,
        type="forecast",
        variables=args.variables,
        model=args.model,
        output=str(data_dir),
    )
    if cmd_download(fcst_args) != 0:
        return 1

    # Step 3: Build verification
    console.rule("[blue]Step 3: Build verification[/blue]")
    verify_args = argparse.Namespace(
        data_dir=str(data_dir),
        variables=args.variables,
        output=str(data_dir / "verification"),
    )
    if cmd_verify(verify_args) != 0:
        return 1

    # Step 4: Compute metrics
    console.rule("[blue]Step 4: Compute metrics[/blue]")
    verification_path = str(data_dir / "verification" / "verification.parquet")
    metrics_dir = data_dir / "metrics"

    for group_by in ["lead_hours", "season", "month"]:
        metrics_args = argparse.Namespace(
            verification=verification_path,
            group_by=group_by,
            metrics="mae,rmse,bias",
            variables=args.variables,
            output=str(metrics_dir),
        )
        if cmd_metrics(metrics_args) != 0:
            return 1

    # Step 5: Generate analysis
    console.rule("[blue]Step 5: Generate analysis[/blue]")
    analyze_args = argparse.Namespace(
        metrics_path=str(metrics_dir / "*.parquet"),
        verification=verification_path,
        output=str(args.output),
        variables=args.variables,
        metrics="mae,rmse,bias",
        plots="all",
        generate_tables=True,
    )
    if cmd_analyze(analyze_args) != 0:
        return 1

    console.print("\n[bold green]Pipeline complete![/bold green]")
    console.print(f"Output: {args.output}")
    return 0
