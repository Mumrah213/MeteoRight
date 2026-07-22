# Public API surface

This document maps the **blessed, importable functions** of the `meteoright`
package — the surface a separate consumer (notably a future MCP server) should
call. It also flags the parts of the codebase that look like public API but are
not safe to depend on yet.

The repo uses a **flat namespace**: the installed package is a set of sibling
top-level packages (`forecast`, `downloader`, `data`, `metrics`, `explore`, …),
not a single `meteoright` umbrella. Import from those package names directly.

## Blessed surface

### Fetching forecasts / observations (raw JSON)

| Function | Import | Endpoint | Notes |
|---|---|---|---|
| `fetch_forecast` | `from forecast import fetch_forecast` | `/v1/forecast` | Operational forecast. Returns raw Open-Meteo JSON. `requests`, no retries. |
| `fetch_history` | `from forecast import fetch_history` | `/v1/archive` | Historical reanalysis/archive. |
| `fetch_climate_normals` | `from forecast import fetch_climate_normals` | `/v1/climate` | Requires climate data synced in the backend. |
| `fetch_multiple_locations` | `from forecast import fetch_multiple_locations` | `/v1/forecast` | Batch points in one query. |
| `fetch_observations` | `from downloader.api import fetch_observations` | Archive API | Pipeline-grade: `httpx`, retries w/ backoff, typed `ApiError`. |
| `fetch_single_run` | `from downloader.api import fetch_single_run` | Single Runs API | One model init time. Public API ignores `model` (always ECMWF IFS). |

Backend host resolution:
- `forecast.fetch.*` → `METEORIGHT_OPEN_METEO_BASE_URL` (default `https://api.open-meteo.com`).
- `downloader.api.*` → per-endpoint env vars in `downloader/constants.py`
  (`METEORIGHT_OPEN_METEO_ARCHIVE_API`, `…_SINGLE_RUNS_API`, etc.).

**Which to wrap in MCP?** For a "just give me the forecast JSON" tool, the
`forecast.fetch.*` wrappers are minimal and predictable. For **typed, validated**
results prefer the provider layer below — that is the recommended MCP surface.
Use `downloader.api.*` when you specifically need the archive/single-run pipeline
semantics (retries, rate-limit handling, per-endpoint hosts).

### Forecast providers (typed + validated — recommended for MCP)

| Class | Import | Convenience method | Returns |
|---|---|---|---|
| `OpenMeteoForecastProvider` | `from forecast import OpenMeteoForecastProvider` | `fetch_forecast(lat, lon, models?, hourly?, daily?, past_days=0, timezone="UTC")` | `RawForecastResponse` |
| `OpenMeteoArchiveProvider` | `from forecast import OpenMeteoArchiveProvider` | `fetch_archive(lat, lon, start_date, end_date, models?, hourly?, timezone="UTC")` | `RawForecastResponse` |
| `OpenMeteoSingleRunsProvider` | `from forecast import OpenMeteoSingleRunsProvider` | `fetch_single_run(lat, lon, date, hourly?)` | `RawForecastResponse` |

Typed support classes (all re-exported from `forecast`): `RawForecastResponse`,
`MeteoProviderError`, `ProviderError`, `ProviderErrorKind`.

- **Success**: `RawForecastResponse` — `.locations[].hourly`/`hourly_units`,
  `.provider`, `.endpoint`, `.request_params`, `.is_valid`, `.raw_json`.
- **Errors**: providers raise `MeteoProviderError`; catch it and surface
  `e.error.model_dump()` (`kind`, `message`, `http_status_code`,
  `null_classification`).
- **Config**: providers use `HttpClient`, which resolves
  `METEORIGHT_OPEN_METEO_BASE_URL` (default public Open-Meteo) and optional
  `METEORIGHT_OPEN_METEO_API_KEY`. Inject a pre-configured `HttpClient` for
  custom backends.

Caveats to handle in a wrapper:
- **Providers target a self-hosted Open-Meteo backend, not the public API.** All
  three send a `models` query param for multi-domain selection, which
  `api.open-meteo.com` rejects with HTTP 400 (`Cannot initialize MultiDomains…`),
  even for a single model. To use the provider layer, point
  `METEORIGHT_OPEN_METEO_BASE_URL` at a self-hosted instance. For the *public*
  API, use the `forecast.fetch.*` wrappers instead (they don't send `models` the
  same way). This is verified in `tests/test_forecast_providers.py`
  (`test_provider_public_api_rejects_models_param`).
- **Single-runs: HTTP 200 ≠ valid data.** Requesting unsupported models/variables
  returns 200 with empty/all-null arrays. `fetch()` discards the `ValidationResult`
  that flags this, so check `response.is_valid` / `response.null_classification`
  yourself and surface an error when invalid.
- **`save_raw_response` filename length.** Provider `fetch()` persists each raw
  response to disk with a filename built from all params; with many `hourly`
  variables this can exceed the OS 255-char limit (`OSError: File name too long`).
  Pass a short `hourly` list, or point `raw_dir` somewhere tolerant — a known
  rough edge in `forecast/base_provider._params_to_path`.

### Loading verification / metrics artifacts (DataFrames)

| Function | Import |
|---|---|
| `load_verification`, `load_metrics`, `filter_metrics`, `pivot_metrics` | `from data.loaders import …` |

> These are **not** re-exported from `data/__init__.py` — import the
> `data.loaders` module path. Candidate for a curated re-export later.

### Metrics computation (already curated — copy this pattern)

`import metrics` exposes a clean `__all__`: `compute_mae`, `compute_rmse`,
`compute_bias`, `compute_std`, `aggregate_metrics`, `read_verification`,
`write_metrics`, `export_csv`, and the `summary_by_*` helpers.

### Interactive exploration (headless-capable)

`from explore import dispatch, run_script, ExploreSession, Selection, Result` —
string-in → `Result`-out, no UI dependency. Usable programmatically.

## Fixed: provider layer now imports

The `forecast/providers/` layer previously failed to import in the flat layout
(multi-dot relative imports + missing `from __future__ import annotations` on
self-referential pydantic models). Both defects are fixed: all `forecast/`
modules now import, the provider classes are re-exported from `forecast`
(see "Forecast providers" above), and `tests/test_forecast_providers.py`
smoke-tests them with a stubbed client.

## Notes / overlapping layers

There are three forecast-fetch implementations with distinct intents:
1. `forecast/fetch.py` — thin `requests` wrappers (blessed above).
2. `downloader/api.py` — `httpx` pipeline client (blessed above).
3. `forecast/providers/openmeteo/*` — the typed, validated provider layer
   (blessed above; recommended for MCP).

The `requests` (#1) and `httpx` (#2/#3) split is why both are project
dependencies. Consolidating onto one HTTP client is a possible future cleanup,
not required for the MCP work.

Minor known issue (not blocking): `BaseProvider.validate()` semantic checks
append to `response.issues`, but `RawForecastResponse` has no `issues` field —
that path would `AttributeError`. In practice providers validate in
`_parse_response` against a local `ValidationResult`, so it is not hit.

## Deferred to the MCP repo (do NOT build in core)

- An umbrella `meteoright` namespace package.
- Tool registries / JSON-schema generation.
- LLM/agent dependencies of any kind.

The core stays a weather/forecasting library + CLI; the MCP server is a separate
consumer that imports the blessed surface above.
