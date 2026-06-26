#!/usr/bin/env python3
"""Unified CLI for weather forecast analysis.

A complete pipeline for downloading, verifying, computing metrics, and
visualizing weather forecast data.

Commands:
    download    Download forecasts and observations from Open-Meteo
    verify      Build verification dataset (join forecasts with observations)
    metrics     Compute aggregated metrics from verification data
    analyze     Generate plots and analysis tables
    pipeline    Run the full pipeline end-to-end
    advanced    Event-based skill analysis
    prepare-dataset  Prepare a local analysis dataset
    dataset-info   Display information about an existing dataset

Examples:
    # Download data
    weather-analyzer download \\
        --lat 55.605 --lon 13.003 \\
        --start 2025-01-01 --end 2025-01-07 \\
        --type forecast --model icon_eu \\
        --output ./data

    # Build verification dataset
    weather-analyzer verify \\
        --data-dir ./data \\
        --output ./data/verification

    # Compute metrics
    weather-analyzer metrics \\
        --verification ./data/verification/*.parquet \\
        --group-by season \\
        --output ./data/metrics

    # Generate analysis
    weather-analyzer analyze \\
        --metrics ./data/metrics/*.parquet \\
        --verification ./data/verification/*.parquet \\
        --output ./outputs

    # Full pipeline. Download, Verify, Analyze+Basic plots
    weather-analyzer pipeline \\
        --lat 55.605 --lon 13.003 \\
        --start 2025-01-01 --end 2025-01-07 \\
        --model icon_eu \\
        --output ./outputs
"""


import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from rich.console import Console

from util.circular import circular_error_series, is_circular

console = Console()
logger = logging.getLogger("weather_analyzer")


def cmd_download(args: argparse.Namespace) -> int:
    """Download forecasts or observations."""
    import time

    from downloader.api import fetch_observations, fetch_single_run
    from downloader.config import DownloadConfig
    from downloader.constants import DEFAULT_API_SLEEP, MODEL_FORECAST_HOURS, MODEL_RUN_CYCLES
    from downloader.normalize import json_to_forecast_df, json_to_observation_df
    from downloader.storage import (
        write_attribution,
        write_download_log,
        write_forecasts,
        write_observations,
    )

    output_dir = Path(args.output).resolve()
    variables = tuple(v.strip() for v in args.variables.split(","))
    api_key = args.api_key
    archive_url = args.open_meteo_archive_url
    single_runs_url = args.open_meteo_single_runs_url
    if args.open_meteo_base_url:
        base_url = args.open_meteo_base_url.rstrip("/")
        archive_url = archive_url or f"{base_url}/v1/archive"
        single_runs_url = single_runs_url or f"{base_url}/v1/forecast"

    config = DownloadConfig(
        latitude=args.lat,
        longitude=args.lon,
        start_date=args.start,
        end_date=args.end,
        variables=variables,
        model=args.model,
        output_dir=output_dir,
    )

    if args.type == "observations":
        console.print(f"[bold]Downloading observations[/bold]: {args.start} to {args.end}")
        response = fetch_observations(
            config.latitude,
            config.longitude,
            config.start_date,
            config.end_date,
            config.variables,
            config.max_retries,
            config.retry_delay,
            api_key=api_key,
            url=archive_url,
        )
        df = json_to_observation_df(response, config.variables)
        rows = write_observations(df, config.output_dir)
        write_download_log(
            config.output_dir,
            "observations",
            None,
            config.start_date,
            config.end_date,
            rows_downloaded=rows,
        )
        write_attribution(
            config.output_dir,
            "observations",
            "open-meteo-archive",
            latitude=config.latitude,
            longitude=config.longitude,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        console.print(f"[green]Done.[/green] Wrote {rows} observation rows.")
        console.print("[dim]Data attribution: Open-Meteo.com (CC BY 4.0)[/dim]")

    else:  # forecast
        if not args.model:
            console.print("[red]Error: --model required for forecast download[/red]")
            return 1

        console.print(
            f"[bold]Downloading forecasts[/bold]: model={args.model}, {args.start} to {args.end}"
        )

        # Generate run times
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC) + timedelta(days=1)
        cycles = MODEL_RUN_CYCLES.get(args.model, [0, 6, 12, 18])
        forecast_hours = MODEL_FORECAST_HOURS.get(args.model, 240)
        forecast_delta = timedelta(hours=forecast_hours)

        run_times = []
        current = start
        while current < end + forecast_delta:
            for hour in cycles:
                run_time = current.replace(hour=hour, minute=0, second=0)
                if run_time < start - forecast_delta:
                    continue
                if run_time >= end + timedelta(hours=24):
                    continue
                run_times.append(run_time)
            current += timedelta(days=1)

        console.print(f"  Fetching {len(run_times)} forecast runs...")

        all_dfs = []
        skipped = 0
        for i, run_time in enumerate(run_times):
            try:
                response = fetch_single_run(
                    config.latitude,
                    config.longitude,
                    run_time,
                    config.variables,
                    model=args.model,
                    max_retries=config.max_retries,
                    retry_delay=config.retry_delay,
                    api_key=api_key,
                    url=single_runs_url,
                )
                df = json_to_forecast_df(response, run_time, args.model, config.variables)
                all_dfs.append(df)
                if i < len(run_times) - 1:
                    time.sleep(DEFAULT_API_SLEEP)
            except Exception as e:
                logger.warning("Failed to fetch run %s: %s", run_time.isoformat(), e)
                skipped += 1

        if not all_dfs:
            console.print("[red]No forecast data downloaded.[/red]")
            return 1

        all_columns = sorted(set().union(*(set(df.columns) for df in all_dfs)))
        for df in all_dfs:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = None
        combined = pd.concat(all_dfs, ignore_index=True)[all_columns]
        rows = write_forecasts(combined, config.output_dir)
        write_download_log(
            config.output_dir,
            "forecast",
            args.model,
            config.start_date,
            config.end_date,
            rows_downloaded=rows,
            rows_skipped=skipped,
        )
        write_attribution(
            config.output_dir,
            "forecast",
            "open-meteo-single-runs",
            latitude=config.latitude,
            longitude=config.longitude,
            model=args.model,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        console.print(f"[green]Done.[/green] Wrote {rows} forecast rows ({skipped} runs failed).")
        console.print("[dim]Data attribution: Open-Meteo.com (CC BY 4.0)[/dim]")

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Build verification dataset by joining forecasts with observations."""
    from pathlib import Path

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    variables = [v.strip() for v in args.variables.split(",")]

    console.print("[bold]Building verification dataset[/bold]")

    # Load forecasts
    fcst_files = list(data_dir.glob("forecasts/model=*/year=*/month=*/data.parquet"))
    if not fcst_files:
        console.print("[red]No forecast files found[/red]")
        return 1

    forecasts = pd.concat([pd.read_parquet(f) for f in fcst_files], ignore_index=True)
    console.print(f"  Loaded {len(forecasts)} forecast rows from {len(fcst_files)} files")

    # Load observations
    obs_files = list(data_dir.glob("observations/year=*/month=*/data.parquet"))
    if not obs_files:
        console.print("[red]No observation files found[/red]")
        return 1

    observations = pd.concat([pd.read_parquet(f) for f in obs_files], ignore_index=True)
    console.print(f"  Loaded {len(observations)} observation rows from {len(obs_files)} files")

    # Prepare columns
    obs_rename = {v: f"observed_{v}" for v in variables if v in observations.columns}
    obs_df = observations[["observation_time"] + list(obs_rename.keys())].copy()
    obs_df = obs_df.rename(columns=obs_rename)

    fcst_cols = [
        "forecast_issue_time",
        "forecast_target_time",
        "lead_hours",
        "model",
        "latitude",
        "longitude",
        "elevation",
    ] + [v for v in variables if v in forecasts.columns]
    fcst_rename = {v: f"forecast_{v}" for v in variables if v in forecasts.columns}
    fcst_df = forecasts[fcst_cols].copy()
    fcst_df = fcst_df.rename(columns=fcst_rename)

    # Join on time
    merged = pd.merge(
        fcst_df, obs_df, left_on="forecast_target_time", right_on="observation_time", how="left"
    )

    # Compute errors
    for var in variables:
        fcst_col = f"forecast_{var}"
        obs_col = f"observed_{var}"
        if fcst_col in merged.columns and obs_col in merged.columns:
            if is_circular(var):
                merged[f"{var}_error"] = circular_error_series(merged[fcst_col], merged[obs_col])
            else:
                merged[f"{var}_error"] = merged[fcst_col] - merged[obs_col]

    # Add time dimensions
    merged["month"] = merged["forecast_target_time"].dt.month
    month_to_season = {
        12: "DJF",
        1: "DJF",
        2: "DJF",
        3: "MAM",
        4: "MAM",
        5: "MAM",
        6: "JJA",
        7: "JJA",
        8: "JJA",
        9: "SON",
        10: "SON",
        11: "SON",
    }
    merged["season"] = merged["month"].map(month_to_season)

    matched = merged["observation_time"].notna().sum()
    console.print(f"  Merged: {len(merged)} rows, {matched} with observations")

    # Save
    output_path = output_dir / "verification.parquet"
    merged.to_parquet(output_path)
    console.print(f"[green]Saved:[/green] {output_path}")

    return 0


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


def cmd_analyze(args: argparse.Namespace) -> int:
    """Generate analysis plots and tables."""
    from data.loaders import filter_metrics, load_metrics, load_verification

    from analysis.extremes import compute_error_quantiles, find_top_errors
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


# Bundled sample shipped with the package, resolved relative to this file so it
# works from any working directory and from an installed wheel.
_SAMPLE_DIR = Path(__file__).resolve().parent / "examples" / "copenhagen_sample"
_SAMPLE_VERIFICATION = _SAMPLE_DIR / "verification" / "verification.parquet"
_SAMPLE_VARIABLES = "temperature_2m,wind_speed_10m,precipitation"


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the offline demo against the bundled Copenhagen sample dataset.

    No network and no backend required: it computes lead-time metrics from the
    pre-built verification table, then renders lead-time and error-distribution
    plots plus summary tables. This is the "it works" smoke test for a fresh
    install.
    """
    if not _SAMPLE_VERIFICATION.exists():
        console.print(
            f"[red]Bundled sample not found at {_SAMPLE_VERIFICATION}[/red]\n"
            "The package may be installed without its example data."
        )
        return 1

    output = Path(getattr(args, "output", None) or (_SAMPLE_DIR / "out"))
    metrics_dir = output / "metrics"

    console.print("[bold]MeteoRight demo[/bold] — bundled Copenhagen sample (offline)")
    console.print(f"  Verification: {_SAMPLE_VERIFICATION}")

    # Step 1: metrics by lead time (drives the lead-time plots).
    metrics_args = argparse.Namespace(
        verification=str(_SAMPLE_VERIFICATION),
        group_by="lead_hours",
        metrics="mae,rmse,bias",
        variables=_SAMPLE_VARIABLES,
        output=str(metrics_dir),
    )
    if cmd_metrics(metrics_args) != 0:
        return 1

    # Step 2: plots + tables (lead-time from metrics, distributions from rows).
    analyze_args = argparse.Namespace(
        metrics_path=str(metrics_dir / "*.parquet"),
        verification=str(_SAMPLE_VERIFICATION),
        output=str(output),
        variables=_SAMPLE_VARIABLES,
        metrics="mae,rmse,bias",
        plots="lead_time,distributions",
        generate_tables=True,
    )
    if cmd_analyze(analyze_args) != 0:
        return 1

    console.print(f"\n[green]Demo complete.[/green] See figures and tables under: {output}")
    return 0


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
        api_key=getattr(args, "api_key", None),
        open_meteo_base_url=getattr(args, "open_meteo_base_url", None),
        open_meteo_archive_url=getattr(args, "open_meteo_archive_url", None),
        open_meteo_single_runs_url=getattr(args, "open_meteo_single_runs_url", None),
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
        api_key=getattr(args, "api_key", None),
        open_meteo_base_url=getattr(args, "open_meteo_base_url", None),
        open_meteo_archive_url=getattr(args, "open_meteo_archive_url", None),
        open_meteo_single_runs_url=getattr(args, "open_meteo_single_runs_url", None),
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


def cmd_advanced_events(args: argparse.Namespace) -> int:
    """Generate events and compute skill scores."""
    from verification.confusion import compute_confusion_matrix
    from verification.event_io import load_verification, write_confusion, write_skill_scores
    from verification.event_validation import validate_confusion, validate_skill_scores
    from verification.events import generate_events
    from verification.skill_scores import compute_skill_scores

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
    from verification.confusion import compute_confusion_matrix
    from verification.event_io import load_verification, write_confusion, write_skill_scores
    from verification.event_validation import validate_confusion, validate_skill_scores
    from verification.events import generate_events
    from verification.skill_scores import compute_skill_scores

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
    from verification.confusion import compute_confusion_matrix
    from verification.event_io import load_verification
    from verification.events import generate_events
    from verification.skill_scores import compute_skill_scores

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


def cmd_grid_points(args: argparse.Namespace) -> int:
    """Find model grid points within a square region."""
    from forecast.grid_points import find_grid_points, summarize_grid_points

    console = Console()

    if args.summary:
        summary = summarize_grid_points(args.lat, args.lon, args.radius_km)
        console.print("[bold]Grid Points Summary[/bold]")
        console.print(f"  Center:     ({summary['center']['lat']}, {summary['center']['lon']})")
        console.print(f"  Radius:     {summary['radius_km']} km (square)")
        console.print(f"  Total pts:  {summary['total_points']} (IFS={summary['ifs_points']}, ERA5={summary['era5_points']})")
        console.print(f"  Unique pos: {summary['unique_positions']}")
        console.print(f"  Lat range:  {summary['lat_range']['min']} to {summary['lat_range']['max']}")
        console.print(f"  Lon range:  {summary['lon_range']['min']} to {summary['lon_range']['max']}")
        console.print(f"  Distance:   {summary['min_distance_km']:.1f} to {summary['max_distance_km']:.1f} km")
    else:
        models = None
        if args.models:
            models = tuple(m.strip() for m in args.models.split(","))
        points = find_grid_points(args.lat, args.lon, args.radius_km, models=models)
        console.print(f"[bold]Grid Points ({len(points)} total)[/bold]")
        for p in points:
            console.print(
                f"  {p.latitude:8.3f}°N  {p.longitude:9.3f}°E  [{p.model:>4}]  {p.distance_km:6.1f} km"
            )
    return 0


# ---------------------------------------------------------------------------
# Agent-facing tool commands (machine-readable JSON output)
# ---------------------------------------------------------------------------


_AGENT_COMMANDS = {
    "ask",
    "agent",
    "chat",
    "compare-models",
    "describe-backend",
    "geocode",
    "resolve-area",
    "list-models",
    "list-variables",
}


def _agent_moved(command: str) -> int:
    """Explain that the agent commands now live in the meteoright-agent package."""
    console.print(
        f"[yellow]'{command}' moved to the separate meteoright-agent package.[/yellow]\n"
        "Install it and use the 'meteoright-agent' command:\n\n"
        "    uv tool install meteoright-agent   # or: pip install meteoright-agent\n"
        f"    meteoright-agent {command} ...\n\n"
        "See https://github.com/Mumrah213/meteoright-agent"
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    # Agent commands moved to the separate meteoright-agent package. Intercept
    # them before argparse so the user gets a clear pointer instead of an
    # "invalid choice" / "unrecognized arguments" error (these commands carry
    # options like --graph that the core parser no longer knows about).
    _args = sys.argv[1:] if argv is None else argv
    if _args and _args[0] in _AGENT_COMMANDS:
        return _agent_moved(_args[0])

    parser = argparse.ArgumentParser(
        description="MeteoRight: local Open-Meteo weather analysis laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # download
    dl = subparsers.add_parser("download", help="Download forecasts or observations")
    dl.add_argument("--lat", type=float, required=True, help="Latitude")
    dl.add_argument("--lon", type=float, required=True, help="Longitude")
    dl.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    dl.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    dl.add_argument("--type", choices=["forecast", "observations"], required=True)
    dl.add_argument("--model", help="Model name (required for forecasts)")
    dl.add_argument("--variables", default="temperature_2m,precipitation")
    dl.add_argument("--output", default="./data", help="Output directory")
    dl.add_argument(
        "--open-meteo-base-url",
        help="Self-hosted Open-Meteo base URL, for example http://localhost:8080",
    )
    dl.add_argument(
        "--open-meteo-archive-url",
        help="Override the observation/archive endpoint URL",
    )
    dl.add_argument(
        "--open-meteo-single-runs-url",
        help="Override the archived forecast/single-runs endpoint URL",
    )
    dl.add_argument(
        "--api-key",
        help="Open-Meteo API key; env fallback is METEORIGHT_OPEN_METEO_API_KEY",
    )

    # verify
    vf = subparsers.add_parser("verify", help="Build verification dataset")
    vf.add_argument("--data-dir", required=True, help="Directory with forecasts/ and observations/")
    vf.add_argument("--variables", default="temperature_2m,precipitation")
    vf.add_argument("--output", required=True, help="Output directory")

    # metrics
    mt = subparsers.add_parser("metrics", help="Compute aggregated metrics")
    mt.add_argument("--verification", required=True, help="Verification parquet path/glob")
    mt.add_argument("--group-by", required=True, help="Groupby dimensions (comma-separated)")
    mt.add_argument("--metrics", default="mae,rmse,bias")
    mt.add_argument("--variables", default="temperature_2m,precipitation")
    mt.add_argument("--output", required=True, help="Output directory")

    # analyze
    an = subparsers.add_parser("analyze", help="Generate plots and tables")
    an.add_argument("--metrics", dest="metrics_path", help="Metrics parquet path/glob")
    an.add_argument("--verification", help="Verification parquet path/glob")
    an.add_argument("--output", required=True, help="Output directory")
    an.add_argument("--variables", default="temperature_2m,precipitation")
    an.add_argument("--metrics-list", dest="metrics", default="mae,rmse,bias")
    an.add_argument(
        "--plots", default="all", help="Plot families: seasonal,lead_time,distributions,all"
    )
    an.add_argument("--generate-tables", action="store_true")

    # demo (offline, bundled sample)
    demo_parser = subparsers.add_parser(
        "demo",
        help="Render plots from the bundled sample dataset (offline, no backend)",
    )
    demo_parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: examples/copenhagen_sample/out)",
    )

    # pipeline
    pl = subparsers.add_parser("pipeline", help="Run full pipeline")
    pl.add_argument("--lat", type=float, required=True, help="Latitude")
    pl.add_argument("--lon", type=float, required=True, help="Longitude")
    pl.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    pl.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    pl.add_argument("--model", required=True, help="Model name")
    pl.add_argument("--variables", default="temperature_2m,precipitation")
    pl.add_argument("--output", default="./outputs", help="Output directory")
    pl.add_argument(
        "--open-meteo-base-url",
        help="Self-hosted Open-Meteo base URL, for example http://localhost:8080",
    )
    pl.add_argument(
        "--open-meteo-archive-url",
        help="Override the observation/archive endpoint URL",
    )
    pl.add_argument(
        "--open-meteo-single-runs-url",
        help="Override the archived forecast/single-runs endpoint URL",
    )
    pl.add_argument(
        "--api-key",
        help="Open-Meteo API key; env fallback is METEORIGHT_OPEN_METEO_API_KEY",
    )

    # Event-based skill analysis.
    adv = subparsers.add_parser(
        "advanced", help="Event verification: events, skill scores, model comparison"
    )
    adv_subparsers = adv.add_subparsers(dest="advanced_command", help="Advanced sub-commands")

    # 'advanced events'
    ae = adv_subparsers.add_parser("events", help="Generate events and compute skill scores")
    ae.add_argument("--verification", required=True, help="Verification parquet path")
    ae.add_argument(
        "--event",
        choices=["frost", "heavy_precip", "hot_day", "no_rain"],
        help="Built-in event name",
    )
    ae.add_argument("--variable", help="Variable name (for custom events)")
    ae.add_argument("--threshold", type=float, help="Threshold value (for custom events)")
    ae.add_argument(
        "--operator", choices=["le", "lt", "ge", "gt", "eq", "ne"], help="Comparison operator"
    )
    ae.add_argument("--group-by", help="Groupby dimensions (comma-separated)")
    ae.add_argument("--output", required=True, help="Output directory")

    # 'advanced skill'
    asub = adv_subparsers.add_parser("skill", help="Compute confusion matrix and skill scores")
    asub.add_argument("--verification", required=True, help="Verification parquet path")
    asub.add_argument(
        "--event",
        choices=["frost", "heavy_precip", "hot_day", "no_rain"],
        help="Built-in event name",
    )
    asub.add_argument("--variable", help="Variable name (for custom events)")
    asub.add_argument("--threshold", type=float, help="Threshold value (for custom events)")
    asub.add_argument(
        "--operator", choices=["le", "lt", "ge", "gt", "eq", "ne"], help="Comparison operator"
    )
    asub.add_argument("--group-by", help="Groupby dimensions (comma-separated)")
    asub.add_argument("--output", required=True, help="Output directory")

    # 'advanced compare'
    ac = adv_subparsers.add_parser("compare", help="Compare model skill scores")
    ac.add_argument("--verification", required=True, help="Verification parquet path")
    ac.add_argument(
        "--event",
        choices=["frost", "heavy_precip", "hot_day", "no_rain"],
        help="Built-in event name",
    )
    ac.add_argument("--models", required=True, help="Model names (comma-separated)")
    ac.add_argument("--group-by", help="Groupby dimensions (comma-separated)")
    ac.add_argument(
        "--metric", default="csi", choices=["hit_rate", "false_alarm_rate", "precision", "csi"]
    )
    ac.add_argument("--output", required=True, help="Output directory")

    # Grid point finder
    gp_parser = subparsers.add_parser(
        "grid-points",
        description="Find model grid points within a square region.",
        help="Find IFS/ERA5 grid points in a square region",
    )
    gp_parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    gp_parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    gp_parser.add_argument(
        "--radius-km", "-r", type=float, default=50.0,
        help="Half-side of square region in km (default: 50)",
    )
    gp_parser.add_argument(
        "--models", "-m", type=str, default=None,
        help='Models to include: "ifs", "era5", or comma-separated (default: both)',
    )
    gp_parser.add_argument("--summary", "-s", action="store_true", help="Show summary only")
    gp_parser.set_defaults(handler=cmd_grid_points)

    # Local dataset preparation.
    pd_parser = subparsers.add_parser(
        "prepare-dataset",
        description="Prepare a curated research dataset for a location and time range.",
        help="Prepare a local analysis dataset",
    )
    # Location
    pd_parser.add_argument(
        "--location", "-l", choices=["copenhagen", "malmo", "stockholm", "berlin"],
        help="Preset location"
    )
    pd_parser.add_argument("--lat", type=float, help="Latitude")
    pd_parser.add_argument("--lon", type=float, help="Longitude")
    # Time
    pd_parser.add_argument("--years", "-y", type=int, default=2, help="Years of data (default: 2)")
    pd_parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    pd_parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    # Variables & models
    pd_parser.add_argument(
        "--variables",
        "-v",
        default="all",
        help="Variables: all, standard, minimal, or comma-separated",
    )
    pd_parser.add_argument("--models", "-m", default=None, help="Models (comma-separated)")
    # Pipeline options
    pd_parser.add_argument("--build-metrics", "-M", action="store_true", help="Precompute metrics")
    pd_parser.add_argument("--build-events", "-E", action="store_true", help="Precompute events")
    pd_parser.add_argument(
        "--build-aggregations", "-A", action="store_true", help="Compute aggregations"
    )
    pd_parser.add_argument(
        "--build-all", "-B", action="store_true", help="Enable all build options"
    )
    pd_parser.add_argument("--resume", action="store_true", help="Resume from existing data")
    pd_parser.add_argument("--verbose", "-V", action="store_true", help="Verbose logging")
    pd_parser.add_argument("--output-dir", "-o", default="./datasets", help="Output directory")
    pd_parser.set_defaults(handler="prepare-dataset")

    # Dataset inspection.
    di_parser = subparsers.add_parser(
        "dataset-info",
        description="Display information about an existing dataset.",
        help="Show dataset information",
    )
    di_parser.add_argument("dataset_path", type=str, help="Path to dataset directory")
    di_parser.set_defaults(handler="dataset-info")

    # (Agent commands — ask/agent/chat/compare-models/etc. — are intercepted at
    # the top of main() and routed to the moved-package message.)

    args = parser.parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        # Quiet noisy HTTP client logs (httpx logs every request at INFO); they
        # clutter JSON output and the agent flow trace. Re-enabled by --verbose.
        for noisy in ("httpx", "httpcore", "openai", "langchain", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    if not args.command:
        parser.print_help()
        return 0

    # Agent-tool commands register a callable handler directly.
    handler = getattr(args, "handler", None)
    if callable(handler):
        return handler(args)

    commands = {
        "download": cmd_download,
        "verify": cmd_verify,
        "metrics": cmd_metrics,
        "analyze": cmd_analyze,
        "demo": cmd_demo,
        "pipeline": cmd_pipeline,
    }

    # Dispatch event-verification commands.
    if args.command == "advanced":
        adv_commands = {
            "events": cmd_advanced_events,
            "skill": cmd_advanced_skill,
            "compare": cmd_advanced_compare,
        }
        handler = adv_commands.get(args.advanced_command)
        if handler:
            return handler(args)
        adv_parser = subparsers.choices["advanced"]
        adv_parser.print_help()
        return 1

    # Dispatch dataset commands.
    if args.command == "prepare-dataset":
        from datasets.cli import _handle_prepare_dataset

        return _handle_prepare_dataset(args)

    if args.command == "dataset-info":
        from datasets.cli import _handle_dataset_info

        return _handle_dataset_info(args)

    # Grid point finder
    if args.command == "grid-points":
        return cmd_grid_points(args)

    # Standard commands
    cmd = commands.get(args.command)
    if cmd:
        return cmd(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
