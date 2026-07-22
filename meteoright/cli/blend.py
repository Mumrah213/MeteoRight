"""`meteoright blend` — adaptive multi-model blending and its evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._shared import console


def cmd_blend(args: argparse.Namespace) -> int:
    """Blend multiple models on shared targets and report whether it helped."""
    from metrics.io import read_verification
    from src.forecast.meta_forecasting import (
        DynamicWeightingEngine,
        MetaEvaluator,
        SkillProfileStore,
    )
    from src.forecast.meta_forecasting.models import SkillProfileEntry
    from verification.schema import forecast_var, observed_var

    console.print("[bold]Adaptive model blending[/bold]")

    variable = args.variable
    fcst_col, obs_col = forecast_var(variable), observed_var(variable)

    df = read_verification(args.verification)
    missing = {c for c in (fcst_col, obs_col, "model", "lead_hours") if c not in df.columns}
    if missing:
        console.print(f"[red]Verification data missing columns: {sorted(missing)}[/red]")
        return 1

    df = df.dropna(subset=[fcst_col, obs_col])
    models = sorted(df["model"].unique())
    if len(models) < 2:
        console.print(f"[red]Blending needs >= 2 models, found: {models or 'none'}[/red]")
        return 1

    # Pivot to shared targets: one row per (issue, target, location) with a
    # column per model. Only targets every model forecast are comparable.
    keys = [
        k
        for k in ["forecast_issue_time", "forecast_target_time", "latitude", "longitude"]
        if k in df.columns
    ]
    # The observation belongs to the target, not to a model. Rows for the same
    # target can still carry slightly different observed values (different
    # interpolation or nearest-station choices per model), so aggregate the
    # observation per target rather than indexing on it — otherwise every model
    # lands on its own row and nothing aligns.
    index_keys = keys + ["lead_hours"]
    wide = df.pivot_table(
        index=index_keys, columns="model", values=fcst_col, aggfunc="first"
    ).reset_index()
    observed = df.groupby(index_keys, as_index=False)[obs_col].mean()
    wide = wide.merge(observed, on=index_keys, how="inner")
    wide = wide.dropna(subset=[*models, obs_col])
    if wide.empty:
        console.print("[red]No targets are shared across all models.[/red]")
        return 1
    console.print(f"  {len(wide)} shared targets across {len(models)} models: {', '.join(models)}")

    # ── Learn skill from a held-out training split ────────────────────────
    # The blend is evaluated only on rows whose skill it never saw, so the
    # weights cannot be informed by the observations they are scored against.
    split = int(len(wide) * args.train_fraction)
    train, test = wide.iloc[:split], wide.iloc[split:]
    if train.empty or test.empty:
        console.print("[red]Not enough rows to split into train and test.[/red]")
        return 1

    store = SkillProfileStore()
    for model in models:
        mae = float((train[model] - train[obs_col]).abs().mean())
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
                value=mae,
                sample_size=len(train),
                is_stable=True,
            ),
        )
        console.print(f"    {model:<24} train MAE = {mae:.3f}")

    weights = DynamicWeightingEngine(skill_store=store).compute_weights(models, variable=variable)
    weight_by_model = {w.model_name: w.weight for w in weights}

    console.rule("[blue]Weights[/blue]")
    for w in sorted(weights, key=lambda x: -x.weight):
        console.print(f"  {w.model_name:<24} {w.weight:6.3f}   {w.rationale}")

    # ── Blend the held-out rows and evaluate ──────────────────────────────
    blended = sum(test[m] * weight_by_model[m] for m in models)
    pairs = lambda values: [  # noqa: E731
        {"value": float(v), "observed": float(o)}
        for v, o in zip(values, test[obs_col], strict=True)
    ]
    result = MetaEvaluator().evaluate(
        blended_forecasts=pairs(blended),
        single_model_forecasts={m: pairs(test[m]) for m in models},
        observations=[{"observed": float(o)} for o in test[obs_col]],
        variable=variable,
        metric="mae",
    )

    console.rule("[blue]Held-out evaluation[/blue]")
    console.print(f"  Blended MAE      : {result.blended_mae:.4f}")
    console.print(f"  Best single MAE  : {result.best_single_mae:.4f} ({result.best_single_model})")
    if result.improvement_over_best:
        console.print(
            f"  [green]Blending helped: {abs(result.mae_improvement_pct):.1f}% lower error[/green]"
        )
    else:
        console.print(
            f"  [yellow]Blending did not help: {result.mae_improvement_pct:+.1f}% error[/yellow]"
        )
    console.print(f"  Evaluated on {result.sample_size} held-out targets")

    if args.output:
        import json

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "variable": variable,
                    "models": models,
                    "weights": weight_by_model,
                    "train_rows": len(train),
                    "blended_mae": result.blended_mae,
                    "best_single_mae": result.best_single_mae,
                    "best_single_model": result.best_single_model,
                    "mae_improvement_pct": result.mae_improvement_pct,
                    "improvement_over_best": result.improvement_over_best,
                    "sample_size": result.sample_size,
                    "rationale": result.rationale,
                },
                indent=2,
            )
        )
        console.print(f"  Wrote {out}")

    return 0
