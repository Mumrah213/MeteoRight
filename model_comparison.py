#!/usr/bin/env python3
"""Comparison plots for model exploration: temperature and wind speed vs lead time.

Uses coarse resolution (every 6 hours) to avoid timeouts.

Usage:
    python model_comparison.py [--data docs/model-exploration-data.json]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
import numpy as np


# Color palette — distinct but not too many colors
MODEL_COLORS = plt.cm.tab20(np.linspace(0, 1, 33))


def load_data(data_path: str) -> dict:
    """Load exploration data."""
    path = Path(data_path)
    if not path.exists():
        print(f"Error: {data_path} not found. Run model_exploration.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def plot_temperature_by_lead_time(data: dict, output_dir: Path) -> None:
    """Plot temperature vs lead time for all models."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (model, model_data) in enumerate(sorted(data.items())):
        temps = model_data.get("temperature_data", [])
        if not temps or not model_data.get("available_variables"):
            continue

        lead_hours = [t["lead_hours"] for t in temps]
        temperatures = [t.get("temperature") for t in temps]

        # Filter out None values
        valid = [(h, t) for h, t in zip(lead_hours, temperatures) if t is not None]
        if not valid:
            continue

        leads, temps_valid = zip(*valid)
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        ax.plot(leads, temps_valid, color=color, alpha=0.6, linewidth=1.0, label=model)

    ax.set_xlabel("Lead Time (hours)", fontsize=12)
    ax.set_ylabel("Temperature 2m (°C)", fontsize=12)
    ax.set_title("Temperature Forecast vs Lead Time — All Models", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=2,
        framealpha=0.9, edgecolor="lightgray"
    )
    ax.set_xlim(0, max(max(d.get("forecast_hours", 0) for d in data.values()), 200))

    plt.tight_layout()
    path = output_dir / "temperature_by_lead_time.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_wind_speed_by_lead_time(data: dict, output_dir: Path) -> None:
    """Plot wind speed vs lead time for all models."""
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (model, model_data) in enumerate(sorted(data.items())):
        winds = model_data.get("wind_speed_data", [])
        if not winds or not model_data.get("available_variables"):
            continue

        lead_hours = [w["lead_hours"] for w in winds]
        wind_speeds = [w.get("wind_speed") for w in winds]

        # Filter out None values
        valid = [(h, w) for h, w in zip(lead_hours, wind_speeds) if w is not None]
        if not valid:
            continue

        leads, winds_valid = zip(*valid)
        color = MODEL_COLORS[i % len(MODEL_COLORS)]

        ax.plot(leads, winds_valid, color=color, alpha=0.6, linewidth=1.0, label=model)

    ax.set_xlabel("Lead Time (hours)", fontsize=12)
    ax.set_ylabel("Wind Speed 10m (m/s)", fontsize=12)
    ax.set_title("Wind Speed Forecast vs Lead Time — All Models", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, ncol=2,
        framealpha=0.9, edgecolor="lightgray"
    )
    ax.set_xlim(0, max(max(d.get("forecast_hours", 0) for d in data.values()), 200))

    plt.tight_layout()
    path = output_dir / "wind_speed_by_lead_time.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_at_key_lead_times(data: dict, output_dir: Path, value_key: str, unit_label: str, title_prefix: str) -> None:
    """Plot values at specific lead times with per-subplot y-axis scaling.

    Args:
        data: Model exploration data dict.
        output_dir: Output directory for plots.
        value_key: Key in data for the values to plot ("temperature" or "wind_speed").
        unit_label: Unit label for the y-axis (e.g. "Temp (°C)").
        title_prefix: Title prefix (e.g. "Temperature").
    """
    lead_times = [0, 24, 48, 72, 168]
    models = sorted(data.keys())

    fig, axes = plt.subplots(1, len(lead_times), figsize=(20, 5))
    if len(lead_times) == 1:
        axes = [axes]

    for ax_idx, lt in enumerate(lead_times):
        ax = axes[ax_idx]
        values_by_model = []
        colors = []

        for i, model in enumerate(models):
            model_data = data[model]
            entries = model_data.get("temperature_data", []) if value_key == "temperature" else model_data.get("wind_speed_data", [])
            for entry in entries:
                if entry["lead_hours"] == lt:
                    values_by_model.append(entry.get(value_key))
                    colors.append(MODEL_COLORS[i % len(MODEL_COLORS)])
                    break

        valid = [(m, v) for m, v in zip(models, values_by_model) if v is not None]
        if not valid:
            continue

        models_valid, values_valid = zip(*valid)
        x_positions = list(range(len(models_valid)))
        ax.scatter(x_positions, values_valid, c=colors, s=30, alpha=0.7)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models_valid, rotation=45, ha="right", fontsize=6)
        ax.set_title(f"Lead: {lt}h", fontsize=11)
        ax.set_ylabel(unit_label)
        ax.grid(True, alpha=0.3, axis="y")

        # Scale y-axis to fit the data with small padding
        vmin = min(values_valid)
        vmax = max(values_valid)
        y_range = vmax - vmin if vmax != vmin else 1.0
        ax.set_ylim(vmin - y_range * 0.1, vmax + y_range * 0.1)

    fig.suptitle(f"{title_prefix} at Key Lead Times — All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = output_dir / f"{value_key}_at_lead_times.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    data_path = "docs/model-exploration-data.json"
    output_dir = Path("docs") / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading exploration data...")
    data = load_data(data_path)
    print(f"  Loaded {len(data)} models")

    print("\nGenerating plots...")
    plot_temperature_by_lead_time(data, output_dir)
    plot_wind_speed_by_lead_time(data, output_dir)
    plot_at_key_lead_times(data, output_dir, "temperature", "Temp (°C)", "Temperature")
    plot_at_key_lead_times(data, output_dir, "wind_speed", "Wind (m/s)", "Wind Speed")

    print("\nDone!")


if __name__ == "__main__":
    main()
