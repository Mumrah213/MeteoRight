"""`meteoright grid-points` — locate model grid points around a location."""

from __future__ import annotations

import argparse

from rich.console import Console


def cmd_grid_points(args: argparse.Namespace) -> int:
    """Find model grid points within a square region."""
    from src.forecast.grid_points import find_grid_points, summarize_grid_points

    console = Console()

    if args.summary:
        summary = summarize_grid_points(args.lat, args.lon, args.radius_km)
        console.print("[bold]Grid Points Summary[/bold]")
        console.print(f"  Center:     ({summary['center']['lat']}, {summary['center']['lon']})")
        console.print(f"  Radius:     {summary['radius_km']} km (square)")
        console.print(
            f"  Total pts:  {summary['total_points']} (IFS={summary['ifs_points']}, ERA5={summary['era5_points']})"
        )
        console.print(f"  Unique pos: {summary['unique_positions']}")
        console.print(
            f"  Lat range:  {summary['lat_range']['min']} to {summary['lat_range']['max']}"
        )
        console.print(
            f"  Lon range:  {summary['lon_range']['min']} to {summary['lon_range']['max']}"
        )
        console.print(
            f"  Distance:   {summary['min_distance_km']:.1f} to {summary['max_distance_km']:.1f} km"
        )
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
