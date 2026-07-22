#!/usr/bin/env python3
"""Generate the blended-vs-best-single showcase figure.

Reads the bundled Copenhagen verification sample, runs the same adaptive
blend the `meteoright blend` command performs (learn skill on a training
split, weight, score the held-out half), and plots held-out MAE per model
against the blend.

Usage:
    python showcase/make_blend_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.forecast.meta_forecasting import DynamicWeightingEngine, SkillProfileStore
from src.forecast.meta_forecasting.models import SkillProfileEntry

VERIFICATION = REPO_ROOT / "examples/copenhagen_sample/verification/verification.parquet"
OUTPUT = REPO_ROOT / "showcase/blend_vs_best_single.png"

VARIABLES = [
    ("temperature_2m", "Temperature 2 m", "°C"),
    ("wind_speed_10m", "Wind speed 10 m", "m/s"),
    ("precipitation", "Precipitation", "mm"),
]
TRAIN_FRACTION = 0.5

# Validated tokens (light mode, surface #fcfcfb). The blend is the subject,
# the individual models are context: accent hue + de-emphasis gray.
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"
DEEMPHASIS = "#898781"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

BAR_WIDTH = 0.40  # thin marks: leave air in the band rather than filling it


def _blend_one(df: pd.DataFrame, variable: str) -> tuple[dict[str, float], float, int]:
    """Return (per-model held-out MAE, blended held-out MAE, n_test)."""
    fcst_col, obs_col = f"forecast_{variable}", f"observed_{variable}"
    data = df.dropna(subset=[fcst_col, obs_col])
    models = sorted(data["model"].unique())

    keys = ["forecast_issue_time", "forecast_target_time", "latitude", "longitude", "lead_hours"]
    keys = [k for k in keys if k in data.columns]

    wide = data.pivot_table(
        index=keys, columns="model", values=fcst_col, aggfunc="first"
    ).reset_index()
    observed = data.groupby(keys, as_index=False)[obs_col].mean()
    wide = wide.merge(observed, on=keys, how="inner").dropna(subset=[*models, obs_col])

    split = int(len(wide) * TRAIN_FRACTION)
    train, test = wide.iloc[:split], wide.iloc[split:]

    store = SkillProfileStore()
    for model in models:
        store.add_model(model)
        store.get_profile(model).add_entry(
            variable,
            None,
            "",
            "",
            SkillProfileEntry(
                model_name=model,
                variable=variable,
                metric_type="mae",
                value=float((train[model] - train[obs_col]).abs().mean()),
                sample_size=len(train),
                is_stable=True,
            ),
        )

    weights = {
        w.model_name: w.weight
        for w in DynamicWeightingEngine(skill_store=store).compute_weights(
            models, variable=variable
        )
    }
    blended = sum(test[m] * weights[m] for m in models)

    per_model = {m: float((test[m] - test[obs_col]).abs().mean()) for m in models}
    return per_model, float((blended - test[obs_col]).abs().mean()), len(test)


def main() -> int:
    if not VERIFICATION.exists():
        print(f"Missing verification sample: {VERIFICATION}", file=sys.stderr)
        return 1

    df = pd.read_parquet(VERIFICATION)
    results = {var: _blend_one(df, var) for var, _, _ in VARIABLES}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0))
    fig.patch.set_facecolor(SURFACE)

    for ax, (variable, title, unit) in zip(axes, VARIABLES, strict=True):
        per_model, blend_mae, n_test = results[variable]
        labels = [*per_model.keys(), "Blend"]
        values = [*per_model.values(), blend_mae]
        colors = [DEEMPHASIS] * len(per_model) + [ACCENT]

        ax.set_facecolor(SURFACE)
        bars = ax.bar(
            range(len(values)),
            values,
            width=BAR_WIDTH,
            color=colors,
            zorder=3,
        )
        # 4px-equivalent rounded data-end, square at the baseline.
        for bar in bars:
            bar.set_joinstyle("round")

        best_single = min(per_model.values())
        change_pct = (blend_mae - best_single) / best_single * 100

        # Label selectively: every bar carries its value because there are only
        # four, and the comparison IS the chart's subject.
        for i, (value, color) in enumerate(zip(values, colors, strict=True)):
            ax.text(
                i,
                value + max(values) * 0.03,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK_PRIMARY if color == ACCENT else INK_SECONDARY,
                fontweight="600" if color == ACCENT else "normal",
            )

        ax.set_title(
            f"{title}",
            fontsize=11,
            color=INK_PRIMARY,
            fontweight="600",
            pad=26,
            loc="left",
        )
        ax.text(
            0.0,
            1.035,
            f"{change_pct:+.1f}% vs best single model",
            transform=ax.transAxes,
            fontsize=9.5,
            color=ACCENT if change_pct < -1 else INK_MUTED,
            fontweight="600" if change_pct < -1 else "normal",
        )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, color=INK_SECONDARY, rotation=20, ha="right")
        ax.set_ylabel(f"Held-out MAE ({unit})", fontsize=9.5, color=INK_SECONDARY)
        ax.set_ylim(0, max(values) * 1.22)

        ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)

    n_test = results[VARIABLES[0][0]][2]
    fig.suptitle(
        "Adaptive blending vs. the best single model — Copenhagen, May 2026",
        fontsize=13.5,
        color=INK_PRIMARY,
        fontweight="600",
        x=0.006,
        ha="left",
        y=0.972,
    )
    fig.text(
        0.006,
        0.906,
        f"Mean absolute error on {n_test} held-out targets never seen when learning model skill. "
        "Lower is better.",
        fontsize=9.5,
        color=INK_SECONDARY,
        ha="left",
    )
    fig.text(
        0.006,
        0.018,
        "Blending helps where model errors are partly independent (temperature, wind) and does "
        "nothing where they are correlated (precipitation).\n"
        "Data: Open-Meteo (CC BY 4.0). Models: dwd_icon, ecmwf_ifs025, gfs_seamless.",
        fontsize=8.5,
        color=INK_MUTED,
        ha="left",
    )

    fig.tight_layout(rect=(0, 0.07, 1, 0.90))
    fig.savefig(OUTPUT, dpi=160, facecolor=SURFACE)
    print(f"Wrote {OUTPUT}")
    for variable, title, _ in VARIABLES:
        per_model, blend_mae, _ = results[variable]
        best = min(per_model.values())
        print(
            f"  {title:<20} blend={blend_mae:.4f}  best_single={best:.4f}  "
            f"{(blend_mae - best) / best * 100:+.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
