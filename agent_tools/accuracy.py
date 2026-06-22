"""High-level forecast-accuracy orchestrator.

``forecast_accuracy`` answers questions like "how accurate is the wind direction
forecast for the Malmö-Copenhagen area?" by composing the smaller tools:

    describe_backend -> resolve_area -> resolve_model_alias
      -> fetch forecast + ERA5 truth for the point over an overlapping window
      -> circular-aware error -> MAE / RMSE / bias
      -> single structured answer

Backend specifics (which models exist, what dates overlap with truth) are
discovered at runtime, so the default date window is the forecast/observation
overlap rather than a hardcoded range.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from agent_tools.areas import resolve_area
from agent_tools.backend import describe_backend
from agent_tools.catalog import variable_units
from agent_tools.download import (
    archive_url_for,
    fetch_forecast_series,
    fetch_observation_series,
)
from agent_tools.models import resolve_model_alias
from metrics.metrics import compute_bias, compute_mae, compute_rmse
from util.circular import circular_error_series, is_circular

_METRIC_FUNCS = {"mae": compute_mae, "rmse": compute_rmse, "bias": compute_bias}


def _err(exc_type: str, message: str) -> dict[str, Any]:
    return {"type": exc_type, "message": message}


def _overlap(
    a: dict[str, str] | None, b: dict[str, str] | None
) -> dict[str, str] | None:
    """Intersect two {'start','end'} ISO-date ranges, or None if disjoint."""
    if not a or not b:
        return None
    start = max(a["start"], b["start"])
    end = min(a["end"], b["end"])
    if start > end:
        return None
    return {"start": start, "end": end}


def forecast_accuracy(
    variable: str,
    area: str | list[str],
    *,
    model: str = "ecmwf",
    start: str | None = None,
    end: str | None = None,
    metrics: tuple[str, ...] = ("mae", "rmse", "bias"),
    backend_base_url: str = "http://127.0.0.1:8080/v1/forecast",
    backend_desc: dict[str, Any] | None = None,
    use_network_geocode: bool = True,
) -> dict[str, Any]:
    """Compute forecast accuracy for one variable over an area.

    Args:
        variable: e.g. ``"wind_direction_10m"`` or ``"temperature_2m"``.
        area: A place, list of places, or "Malmö-Copenhagen"-style string.
        model: Friendly model name (mapped to a backend domain).
        start, end: ISO dates. Default: the forecast/observation overlap window.
        metrics: Which metrics to compute (subset of mae/rmse/bias).
        backend_base_url: Forecast endpoint of the backend.
        backend_desc: A prior describe_backend() result to reuse.
        use_network_geocode: Allow Open-Meteo geocoding fallback for place names.

    Returns:
        A structured answer dict (see module docstring); ``error`` is None on
        success or ``{"type","message"}`` on failure.
    """
    question = {"variable": variable, "area": area, "model": model}
    out: dict[str, Any] = {
        "question": question,
        "resolved": None,
        "answer": None,
        "notes": [],
        "error": None,
    }

    # 1. Backend capabilities (models + coverage + observation coverage).
    if backend_desc is None:
        backend_desc = describe_backend(backend_base_url)
    if not backend_desc.get("reachable"):
        out["error"] = backend_desc.get("error") or _err(
            "BackendUnreachable", f"Backend not reachable at {backend_base_url}"
        )
        return out

    # 2. Resolve the model to a backend domain and confirm it has data.
    domain = resolve_model_alias(model)
    model_entry = next(
        (m for m in backend_desc["models"] if m["backend_name"] == domain), None
    )
    if model_entry is None:
        available = [m["backend_name"] for m in backend_desc["models"]]
        out["error"] = _err(
            "ModelUnavailable",
            f"Model '{model}' -> '{domain}' has no data on backend. Available: {available}",
        )
        return out

    # 3. Resolve the area to a single representative grid point.
    area_res = resolve_area(area, use_network=use_network_geocode)
    if area_res["error"] or area_res["grid_point"] is None:
        out["error"] = area_res["error"] or _err("AreaUnresolved", "No grid point for area")
        return out
    point = area_res["grid_point"]

    # 4. Pick a date window: explicit, else the forecast/observation overlap.
    obs_info = backend_desc.get("observations")
    obs_cov = obs_info["coverage"] if obs_info else None
    if start and end:
        window = {"start": start, "end": end}
    else:
        window = _overlap(model_entry.get("coverage"), obs_cov)
        if window is None:
            out["error"] = _err(
                "NoOverlap",
                "Forecast and observation coverage do not overlap; pass start/end explicitly. "
                f"forecast={model_entry.get('coverage')} observations={obs_cov}",
            )
            return out
        out["notes"].append(
            f"Date window defaulted to forecast/observation overlap {window['start']}..{window['end']}."
        )

    out["resolved"] = {
        "model": model,
        "model_backend_name": domain,
        "grid_point": point,
        "center": area_res["center"],
        "places": [p["name"] for p in area_res["places"]],
        "date_range": window,
        "spatial_aggregation": "single_nearest_grid_point",
    }

    # 5. Fetch forecast + truth and join on target time.
    variables = (variable,)
    try:
        fcst = fetch_forecast_series(
            point["lat"], point["lon"], domain, variables,
            window["start"], window["end"], base_url=backend_base_url,
        )
        obs = fetch_observation_series(
            point["lat"], point["lon"], variables,
            window["start"], window["end"], archive_url=archive_url_for(backend_base_url),
        )
    except Exception as exc:  # network/parse failure -> structured error
        out["error"] = _err(type(exc).__name__, str(exc))
        return out

    merged = pd.merge(
        fcst.rename(columns={variable: "forecast"}),
        obs.rename(columns={variable: "observed"}),
        on="time",
        how="inner",
    )

    # 6. Circular-aware error, then metrics.
    if is_circular(variable):
        merged["error"] = circular_error_series(merged["forecast"], merged["observed"])
        out["notes"].append("Wind direction uses shortest-angle (circular) error.")
    else:
        merged["error"] = merged["forecast"] - merged["observed"]

    valid = merged["error"].dropna()
    if valid.empty:
        # Distinguish "variable not available" from "times didn't line up" so the
        # agent gets an actionable error rather than a bare empty result.
        fcst_has = fcst[variable].notna().any()
        obs_has = obs[variable].notna().any()
        if not fcst_has or not obs_has:
            missing = []
            if not fcst_has:
                missing.append(f"forecast model '{domain}'")
            if not obs_has:
                missing.append("observation archive")
            out["error"] = _err(
                "VariableUnavailable",
                f"'{variable}' has no data in {' and '.join(missing)} for this "
                f"point/window on the backend (only some variables are synced).",
            )
        else:
            out["error"] = _err(
                "NoMatchedData",
                "Forecast and observation series did not overlap in time.",
            )
        return out

    computed = {m: _METRIC_FUNCS[m](merged["error"]) for m in metrics if m in _METRIC_FUNCS}
    out["answer"] = {
        "variable": variable,
        "is_circular": is_circular(variable),
        "metrics": {k: round(v, 4) for k, v in computed.items()},
        "units": variable_units(variable),
        "sample_size": int(len(valid)),
        "missing_count": int(len(merged) - len(valid)),
    }
    out["notes"].append("Area resolved to nearest grid point (no spatial pooling).")
    return out


def compare_models(
    variable: str,
    area: str | list[str],
    *,
    models: list[str] | None = None,
    metric: str = "mae",
    start: str | None = None,
    end: str | None = None,
    backend_base_url: str = "http://127.0.0.1:8080/v1/forecast",
    backend_desc: dict[str, Any] | None = None,
    use_network_geocode: bool = True,
) -> dict[str, Any]:
    """Rank models by accuracy for one variable over an area.

    Runs :func:`forecast_accuracy` for each model and sorts the successful ones
    by ``metric`` (ascending — lower error is better). Models that cannot be
    evaluated (no overlap, variable unavailable, ...) are reported separately
    with their error, so the comparison stays honest about what was skipped.

    Args:
        models: Friendly model names. Default: every model the backend serves.
        metric: Ranking metric (one of mae/rmse/bias; |bias| is used to rank).
        Other args mirror :func:`forecast_accuracy`.

    Returns:
        ``{"question", "variable", "metric", "ranking": [...],
           "skipped": [{"model", "error"}], "error": None|{...}}``
    """
    if backend_desc is None:
        backend_desc = describe_backend(backend_base_url)
    if not backend_desc.get("reachable"):
        return {
            "question": {"variable": variable, "area": area},
            "ranking": [], "skipped": [],
            "error": backend_desc.get("error")
            or _err("BackendUnreachable", f"Backend not reachable at {backend_base_url}"),
        }

    if models is None:
        # Reverse the alias table so each backend domain gets a friendly name.
        from agent_tools.models import MODEL_ALIASES

        friendly_for = {d: a for a, d in MODEL_ALIASES.items() if a != d}
        models = [
            friendly_for.get(m["backend_name"], m["backend_name"])
            for m in backend_desc["models"]
        ]

    ranking: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for model in models:
        r = forecast_accuracy(
            variable, area, model=model, start=start, end=end,
            metrics=(metric,) if metric != "bias" else ("bias",),
            backend_base_url=backend_base_url, backend_desc=backend_desc,
            use_network_geocode=use_network_geocode,
        )
        if r["answer"] is None:
            skipped.append({"model": model, "error": r["error"]})
            continue
        ranking.append({
            "model": model,
            "model_backend_name": r["resolved"]["model_backend_name"],
            "value": r["answer"]["metrics"].get(metric),
            "sample_size": r["answer"]["sample_size"],
            "date_range": r["resolved"]["date_range"],
        })

    # Lower is better for mae/rmse; for bias, closest to zero.
    ranking.sort(key=lambda e: abs(e["value"]) if e["value"] is not None else float("inf"))

    return {
        "question": {"variable": variable, "area": area},
        "variable": variable,
        "metric": metric,
        "units": variable_units(variable),
        "ranking": ranking,
        "skipped": skipped,
        "error": None if ranking else _err("NoModelsEvaluated", "No model could be evaluated"),
    }
