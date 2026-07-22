"""`meteoright analyze` — generate plots and summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._shared import console


def cmd_analyze(args: argparse.Namespace) -> int:
    """Generate analysis plots and tables."""
    from analysis.extremes import compute_error_quantiles, find_top_errors
    from data.loaders import filter_metrics, load_metrics, load_verification
    from plots.distributions import save_error_boxplot, save_error_histogram
    from plots.lead_time import save_lead_time_all_metrics, save_lead_time_degradation
    from plots.seasonal import save_monthly_metrics, save_seasonal_metrics

    console.print("[bold]Generating analysis[/bold]")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = [v.strip() for v in args.variables.split(",")]
    metric_names = [m.strip() for m in args.metrics.split(",")]
    plot_families = [p.strip() for p in args.plots.split(",")]

    # Load data
    metrics_df = None
    if args.metrics_path:
        metrics_df = load_metrics(args.metrics_path)
        metrics_df = filter_metrics(metrics_df, variables=variables, metrics=metric_names)
        console.print(f"  Loaded {len(metrics_df)} metric rows")

    verification_df = None
    if args.verification:
        verification_df = load_verification(args.verification)
        console.print(f"  Loaded {len(verification_df)} verification rows")

    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"

    # Generate plots
    if "seasonal" in plot_families or "all" in plot_families:
        if metrics_df is not None:
            console.rule("[blue]Seasonal Plots[/blue]")
            seasonal_dir = fig_dir / "seasonal"
            seasonal_dir.mkdir(parents=True, exist_ok=True)
            for var in variables:
                for metric in metric_names:
                    save_seasonal_metrics(metrics_df, var, metric, seasonal_dir)
                    save_monthly_metrics(metrics_df, var, metric, seasonal_dir)
                    console.print(f"  ✓ {metric}_by_season_{var}.png")

    if "lead_time" in plot_families or "all" in plot_families:
        if metrics_df is not None:
            console.rule("[blue]Lead Time Plots[/blue]")
            lt_dir = fig_dir / "lead_time"
            lt_dir.mkdir(parents=True, exist_ok=True)
            for var in variables:
                for metric in metric_names:
                    save_lead_time_degradation(metrics_df, var, metric, lt_dir)
                    console.print(f"  ✓ {metric}_by_lead_time_{var}.png")
                save_lead_time_all_metrics(metrics_df, var, lt_dir)

    if "distributions" in plot_families or "all" in plot_families:
        if verification_df is not None:
            console.rule("[blue]Distribution Plots[/blue]")
            dist_dir = fig_dir / "distributions"
            dist_dir.mkdir(parents=True, exist_ok=True)
            lead_times = sorted(verification_df["lead_hours"].unique())[:5]
            for var in variables:
                for lt in lead_times:
                    save_error_histogram(verification_df, var, lt, dist_dir)
                save_error_boxplot(verification_df, var, dist_dir)
                console.print(f"  ✓ {var} distributions")

    # Generate tables
    if args.generate_tables and verification_df is not None:
        console.rule("[blue]Analysis Tables[/blue]")
        table_dir.mkdir(parents=True, exist_ok=True)

        for var in variables:
            error_col = f"{var}_error"
            if error_col not in verification_df.columns:
                continue

            # Top errors
            top_errors = find_top_errors(verification_df, var, n=100)
            top_errors.to_csv(table_dir / f"{var}_top_errors.csv", index=False)

            # Error quantiles
            quantiles = compute_error_quantiles(verification_df, var)
            quantiles.to_csv(table_dir / f"{var}_error_quantiles.csv", index=False)

            console.print(f"  ✓ {var} tables")

    console.print(f"\n[green]Output:[/green] {output_dir}")
    return 0
