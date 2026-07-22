#!/usr/bin/env python3
"""Generate the interpolation-effect showcase figure.

Compares native-grid verification (forecast taken at the nearest model grid
point) against verification after the forecast field is interpolated onto the
observation location, across 23 observation sites around Copenhagen.

The headline is not the average improvement but *why* it happens: the gain
tracks the distance between the grid point and the observation, which is the
error interpolation is supposed to remove.

Usage:
    python showcase/make_interpolation_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from verification.interpolate import haversine_km
from verification.interpolate_frames import interpolate_forecasts_to_observations

SAMPLE = REPO_ROOT / "examples/copenhagen_grid_sample"
OUTPUT = REPO_ROOT / "showcase/interpolation_effect.png"

VARIABLES = [
    ("wind_speed_10m", "Wind speed 10 m", "km/h", False),
    ("wind_gusts_10m", "Wind gusts 10 m", "km/h", False),
    ("temperature_2m", "Temperature 2 m", "°C", False),
    ("wind_direction_10m", "Wind direction 10 m", "°", True),
]

# Validated tokens (light mode, surface #fcfcfb).
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"
DEEMPHASIS = "#898781"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def _mae(forecast: np.ndarray, observed: np.ndarray, circular: bool) -> float:
    valid = np.isfinite(forecast) & np.isfinite(observed)
    if not valid.any():
        return float("nan")
    if circular:
        return float(np.abs((forecast[valid] - observed[valid] + 180) % 360 - 180).mean())
    return float(np.abs(forecast[valid] - observed[valid]).mean())


def evaluate() -> pd.DataFrame:
    """Per-site native and interpolated MAE for every variable."""
    forecasts = pd.read_parquet(SAMPLE / "forecasts.parquet")
    observations = pd.read_parquet(SAMPLE / "observations.parquet")
    grid = forecasts[["latitude", "longitude"]].drop_duplicates()
    names = [name for name, _, _, _ in VARIABLES]

    records = []
    for (obs_lat, obs_lon), site in observations.groupby(["latitude", "longitude"]):
        nearest = min(
            grid.itertuples(),
            key=lambda row: haversine_km(obs_lat, obs_lon, row.latitude, row.longitude),
        )
        distance = haversine_km(obs_lat, obs_lon, nearest.latitude, nearest.longitude)

        interpolated, _ = interpolate_forecasts_to_observations(
            forecasts, site, names, method="bilinear"
        )
        native = forecasts[
            (forecasts.latitude == nearest.latitude) & (forecasts.longitude == nearest.longitude)
        ]

        joined_interp = interpolated.merge(
            site, left_on="forecast_target_time", right_on="observation_time", suffixes=("", "_obs")
        )
        joined_native = native.merge(
            site, left_on="forecast_target_time", right_on="observation_time", suffixes=("", "_obs")
        )

        record = {"distance_km": distance}
        for name, _, _, circular in VARIABLES:
            record[f"{name}_native"] = _mae(
                joined_native[name].to_numpy(), joined_native[f"{name}_obs"].to_numpy(), circular
            )
            record[f"{name}_interp"] = _mae(
                joined_interp[name].to_numpy(), joined_interp[f"{name}_obs"].to_numpy(), circular
            )
        records.append(record)

    return pd.DataFrame(records)


def main() -> int:
    if not (SAMPLE / "forecasts.parquet").exists():
        print(f"Missing grid sample: {SAMPLE}", file=sys.stderr)
        print("Build it with: python showcase/make_grid_sample.py", file=sys.stderr)
        return 1

    results = evaluate()

    fig = plt.figure(figsize=(12.5, 5.6))
    fig.patch.set_facecolor(SURFACE)
    # Explicit margins: tight_layout cannot see the figure-level title and
    # footnote, and silently overlaps them with the axes.
    grid_spec = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.05, 1.0],
        wspace=0.30,
        left=0.135,
        right=0.985,
        top=0.76,
        bottom=0.22,
    )

    # ── Left: percentage change per variable ─────────────────────────────
    # Percent, not raw MAE: these variables are km/h, degrees Celsius and
    # degrees of angle, so a shared absolute axis would be meaningless.
    ax = fig.add_subplot(grid_spec[0, 0])
    ax.set_facecolor(SURFACE)

    labels, changes = [], []
    for name, title, _, _ in VARIABLES:
        native = results[f"{name}_native"].mean()
        interp = results[f"{name}_interp"].mean()
        labels.append(title)
        changes.append(100.0 * (native - interp) / native)

    positions = np.arange(len(labels))
    ax.barh(positions, changes, height=0.30, color=ACCENT, zorder=3)
    for y, value in zip(positions, changes, strict=True):
        ax.text(
            value + 0.15,
            y,
            f"{value:.1f}%",
            va="center",
            fontsize=9.5,
            color=INK_PRIMARY,
            fontweight="600",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK_SECONDARY)
    ax.invert_yaxis()
    ax.set_xlabel("MAE reduction from interpolation (%)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_xlim(0, max(changes) * 1.25)
    ax.set_title(
        "Interpolation reduces error for every variable",
        fontsize=11,
        color=INK_PRIMARY,
        fontweight="600",
        loc="left",
        pad=10,
    )
    ax.grid(axis="x", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)

    # ── Right: why — improvement against grid-to-observation distance ────
    ax2 = fig.add_subplot(grid_spec[0, 1])
    ax2.set_facecolor(SURFACE)

    improvement = results["wind_speed_10m_native"] - results["wind_speed_10m_interp"]
    ax2.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax2.scatter(
        results["distance_km"],
        improvement,
        s=52,
        color=DEEMPHASIS,
        edgecolors=SURFACE,
        linewidths=2,
        zorder=3,
        label="One observation site",
    )

    # Binned means: with one site far from the rest, the raw scatter reads as
    # flat. The binned trend is what the eye should follow.
    bins = [0, 3, 6, 10, 20]
    binned = pd.DataFrame({"distance": results["distance_km"], "improvement": improvement})
    binned["bin"] = pd.cut(binned["distance"], bins)
    summary = binned.groupby("bin", observed=True).agg(
        centre=("distance", "mean"), mean_improvement=("improvement", "mean")
    )
    ax2.plot(
        summary["centre"],
        summary["mean_improvement"],
        color=ACCENT,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=4,
        label="Mean by distance band",
    )
    legend = ax2.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK_SECONDARY)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    ax2.set_xlabel("Distance, grid point to observation (km)", fontsize=9.5, color=INK_SECONDARY)
    ax2.set_ylabel("Wind-speed MAE reduction (km/h)", fontsize=9.5, color=INK_SECONDARY)
    ax2.set_title(
        "…and the gain grows with the spatial mismatch",
        fontsize=11,
        color=INK_PRIMARY,
        fontweight="600",
        loc="left",
        pad=10,
    )
    ax2.grid(color=GRIDLINE, linewidth=1, zorder=0)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.spines["left"].set_color(BASELINE)
    ax2.spines["bottom"].set_color(BASELINE)
    ax2.tick_params(colors=INK_MUTED, labelsize=9, length=0)

    improved = int((improvement > 0).sum())
    fig.suptitle(
        "Interpolating forecasts onto observation locations — Copenhagen, one year",
        fontsize=13.5,
        color=INK_PRIMARY,
        fontweight="600",
        x=0.006,
        ha="left",
        y=0.965,
    )
    fig.text(
        0.006,
        0.885,
        f"{len(results)} observation sites verified against a 24-point ECMWF IFS 0.25° grid. "
        f"Wind speed improves at {improved} of {len(results)} sites.",
        fontsize=9.5,
        color=INK_SECONDARY,
        ha="left",
    )
    fig.text(
        0.006,
        0.035,
        "Native verification compares the observation with the nearest grid point, so it charges the "
        "model for a spatial mismatch it never had a chance to get right.\n"
        "Data: Open-Meteo (CC BY 4.0). Reproduce with: python showcase/make_interpolation_figure.py",
        fontsize=8.5,
        color=INK_MUTED,
        ha="left",
    )

    fig.savefig(OUTPUT, dpi=160, facecolor=SURFACE)
    print(f"Wrote {OUTPUT}")
    for (name, title, unit, _), change in zip(VARIABLES, changes, strict=True):
        native = results[f"{name}_native"].mean()
        interp = results[f"{name}_interp"].mean()
        print(f"  {title:<22} {native:7.4f} -> {interp:7.4f} {unit:<5} {change:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
