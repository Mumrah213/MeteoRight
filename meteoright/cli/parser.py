"""Argument parser for the MeteoRight CLI.

Kept separate from dispatch so the full command surface can be inspected
(and tested) without importing every command implementation.
"""

from __future__ import annotations

import argparse

from .grid import cmd_grid_points


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level parser with every subcommand attached.

    The subparsers action is stored on the parser as ``_meteoright_subparsers``
    so dispatch can print a subcommand's help without reaching into argparse
    internals.
    """
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
    vf.add_argument(
        "--interpolate",
        choices=["none", "nearest", "bilinear", "idw"],
        default="none",
        help=(
            "Interpolate forecasts onto the observation location before "
            "computing errors (default: none, i.e. native grid point)"
        ),
    )

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

    # pipeline
    pl = subparsers.add_parser("pipeline", help="Run full pipeline")
    pl.add_argument("--lat", type=float, required=True, help="Latitude")
    pl.add_argument("--lon", type=float, required=True, help="Longitude")
    pl.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    pl.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    pl.add_argument("--model", required=True, help="Model name")
    pl.add_argument("--variables", default="temperature_2m,precipitation")
    pl.add_argument("--output", default="./outputs", help="Output directory")

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

    # blend
    bl = subparsers.add_parser(
        "blend",
        description=(
            "Blend multiple models on shared verification targets using adaptive "
            "skill-based weights, then report whether the blend beat the best single model."
        ),
        help="Adaptively blend models and evaluate the result",
    )
    bl.add_argument("--verification", required=True, help="Verification parquet path/glob")
    bl.add_argument("--variable", default="temperature_2m", help="Variable to blend")
    bl.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
        help="Fraction of rows used to learn skill; the rest is held out (default: 0.5)",
    )
    bl.add_argument(
        "--split",
        choices=("temporal", "positional"),
        default="temporal",
        help=(
            "How to divide train from test. 'temporal' (default) learns on the earliest "
            "issue times and evaluates on the latest, so the result is a claim about "
            "future performance. 'positional' splits on row order, which is only "
            "chronological for single-location data (default: temporal)"
        ),
    )
    bl.add_argument("--output", help="Optional JSON path for the blend report")

    # Grid point finder
    gp_parser = subparsers.add_parser(
        "grid-points",
        description="Find model grid points within a square region.",
        help="Find IFS/ERA5 grid points in a square region",
    )
    gp_parser.add_argument("--lat", type=float, required=True, help="Center latitude")
    gp_parser.add_argument("--lon", type=float, required=True, help="Center longitude")
    gp_parser.add_argument(
        "--radius-km",
        "-r",
        type=float,
        default=50.0,
        help="Half-side of square region in km (default: 50)",
    )
    gp_parser.add_argument(
        "--models",
        "-m",
        type=str,
        default=None,
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
        "--location", "-l", choices=["copenhagen", "stockholm", "berlin"], help="Preset location"
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

    parser._meteoright_subparsers = subparsers
    return parser
