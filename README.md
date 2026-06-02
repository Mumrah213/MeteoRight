# MeteoRight

A small weather analysis laboratory built on Open-Meteo.

MeteoRight helps you ask a weather question, pull historical forecasts and
observations into local files, and get to analyzing quickly. It is
focused on forecast evolution, lead-time degradation, grid verification and interpolation, event
skill, and model comparison.

It is not a forecasting model or a general weather API wrapper. It is a
practical workspace for investigating how forecast accuracy depends on lead time, using Open Meteo's open API.

## Why This Exists

Forecasts are most useful when you can ask specific questions:

- How did the forecast evolve before a storm?
- Which lead times were still useful?
- Were forecast misses caused by timing, spatial mismatch, or model bias?
- Did a predicted rain or frost event actually happen?
- Do the same locations fail across multiple variables?

Open-Meteo makes the raw forecast and observation data accessible. This project
turns that access into local analysis tables, verification metrics, and curated
figures that make iteration fast.

The goal is simple: ask a weather question, get to a plot with verified data quickly.

## Showcase

The `showcase/` folder contains the curated figures to showcase what can be achieved by a single CLI command.

E.g., it is straightforward to generate the following plot showing the accuracy of wind-related variables in the Malmö/Copenhagen area.
![Wind forecast verification grid analysis](showcase/wind_composite_interpolated_linear_nearest.png)

A cornerstone of the MeteoRight analysis scheme is the grid interpolation, which minimizes the errors from the mismatch between the forecast and the observation grid
![Interpolation effect on wind verification](showcase/grid_interpolation_schematic.png)
Since most available observations are on land, whereas forecasts are everywhere, the interpolation
can have significant effects on coastal areas
![Interpolation effect on wind verification](showcase/wind_interpolation_effect.png)

Studying e.g., the mean absolute error (MAE) as a factor of lead time - averaged across an entire year - the decay in forecast accuracy with lead time is clear
![Precipitation event verification](showcase/wind_error_quantiles_interpolated.png)
and to answer questions like ***are there months which are better/worse in terms of forecast accuracy for a given variable?*** one can generate heat maps like this
![Precipitation event verification](showcase/monthly_lead_mae_heatmap.png)

Event-based verification turns weather questions into yes/no outcomes: forecast
rain, observed rain, missed event, false alarm.

![Precipitation event verification](showcase/precipitation_event_confusion.png)

While Copenhagen was chosen as a demo area, the methodology can be applied anywhere on the globe.

## What You Can Do

- Download observations and archived forecast runs from Open-Meteo.
- Preserve forecast issue time, target time, model, location, and lead time.
- Compare forecasts with observed weather.
- Measure MAE, RMSE, bias, event skill, and lead-time degradation.
- Study forecast evolution before specific events.
- Run local grid investigations around a city or point of interest.
- Compare native grid verification with interpolated verification.
- Build reproducible weather investigations from local Parquet data.

## Open-Meteo Backend

MeteoRight works with the public Open-Meteo APIs by default. That is the easiest
way to try the project: no local backend is required for
non-commercial use. This method is however, strongly rate-limited.

For larger investigations, the fast path is a local Open-Meteo backend. The
upstream server is open source at
[open-meteo/open-meteo](https://github.com/open-meteo/open-meteo), and their
README links to Docker-based self-hosting instructions. Once your local server
is running, point MeteoRight at it:

```bash
export METEORIGHT_OPEN_METEO_BASE_URL="http://localhost:8080"
```

You can also pass the backend directly to a download command:

```bash
meteoright download \
  --lat 55.605 --lon 12.574 \
  --start 2026-05-20 --end 2026-05-22 \
  --type observations \
  --variables temperature_2m,precipitation,wind_speed_10m \
  --open-meteo-base-url http://localhost:8080 \
  --output data/copenhagen_demo
```

Commercial Open-Meteo users are can use their API key from the
[Open-Meteo pricing page](https://open-meteo.com/en/pricing). MeteoRight sends
that key as the Open-Meteo `apikey` query parameter:

```bash
export METEORIGHT_OPEN_METEO_API_KEY="your-key"
```

Endpoint overrides are available when a hosted or self-hosted deployment uses
separate domains:

```bash
export METEORIGHT_OPEN_METEO_ARCHIVE_API="https://archive-api.open-meteo.com/v1/archive"
export METEORIGHT_OPEN_METEO_SINGLE_RUNS_API="https://single-runs-api.open-meteo.com/v1/forecast"
export METEORIGHT_OPEN_METEO_PREVIOUS_RUNS_API="https://previous-runs-api.open-meteo.com/v1/forecast"
export METEORIGHT_OPEN_METEO_HISTORICAL_FORECAST_API="https://historical-forecast-api.open-meteo.com/v1/forecast"
```

## Quick Start

Requires Python 3.11+.

```bash
git clone <repo-url>
cd meteoright

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Download a small Copenhagen example:

```bash
meteoright download \
  --lat 55.605 --lon 12.574 \
  --start 2026-05-20 --end 2026-05-22 \
  --type observations \
  --variables temperature_2m,precipitation,wind_speed_10m \
  --output data/copenhagen_demo

meteoright download \
  --lat 55.605 --lon 12.574 \
  --start 2026-05-20 --end 2026-05-22 \
  --type forecast \
  --model ecmwf_ifs_single \
  --variables temperature_2m,precipitation,wind_speed_10m \
  --output data/copenhagen_demo
```

Build verification rows and first metrics:

```bash
meteoright verify \
  --data-dir data/copenhagen_demo \
  --variables temperature_2m,precipitation,wind_speed_10m \
  --output data/copenhagen_demo/verification

meteoright metrics \
  --verification "data/copenhagen_demo/verification/*.parquet" \
  --group-by lead_hours,model \
  --variables temperature_2m,precipitation,wind_speed_10m \
  --output data/copenhagen_demo/metrics
```

Create the first figures:

```bash
meteoright analyze \
  --metrics "data/copenhagen_demo/metrics/*.parquet" \
  --verification "data/copenhagen_demo/verification/*.parquet" \
  --variables temperature_2m,precipitation,wind_speed_10m \
  --plots lead_time,distributions \
  --generate-tables \
  --output data/copenhagen_demo/analysis
```

This writes lead-time plots, error-distribution plots, and small CSV summary
tables under `data/copenhagen_demo/analysis/`.

For grid-based investigations around a location:

```bash
meteoright grid-points \
  --lat 55.605 --lon 12.574 \
  --radius-km 50 \
  --summary
```

Then open `showcase/README.md` for examples of the analysis outputs this repo is shaped around.

## Python Examples

Fetch observations and archived forecast runs:

```python
from src.core.pipeline import download_forecast, download_historical

location = {"name": "Copenhagen", "lat": 55.605, "lon": 12.574}
variables = ("temperature_2m", "precipitation", "wind_speed_10m")

download_historical(location, "2026-05-20", "2026-05-22", variables, "data/copenhagen_demo")
download_forecast(
    location,
    "2026-05-20",
    "2026-05-22",
    variables,
    ["ecmwf_ifs_single"],
    "data/copenhagen_demo",
)
```

Align forecasts with observations and calculate lead-time error:

```python
from pathlib import Path

from src.historical.verification.aligner import align_forecasts_with_observations
from src.historical.verification.loaders import load_forecasts, load_observations

data_dir = Path("data/copenhagen_demo")
variables = ("temperature_2m", "wind_speed_10m")

forecast = load_forecasts(data_dir, "ecmwf_ifs_single", "2026-05-20", "2026-05-22")
truth = load_observations(data_dir, "2026-05-20", "2026-05-22")
aligned = align_forecasts_with_observations(forecast, truth, variables)

aligned["temperature_error"] = (
    aligned["forecast_temperature_2m"] - aligned["observed_temperature_2m"]
)
mae_by_lead = aligned.groupby("lead_hours")["temperature_error"].apply(lambda s: s.abs().mean())
print(mae_by_lead.head())
```

Use a local Open-Meteo backend directly:

```python
from src.forecast.fetch import fetch_forecast

data = fetch_forecast(
    55.605,
    12.574,
    models=["ecmwf_ifs_hres"],
    hourly=["temperature_2m", "wind_speed_10m"],
    past_days=2,
)
```

## Example Investigations

- **Forecast evolution**: inspect how forecasts for the same event changed
  across issue times.
- **Lead-time degradation**: measure where forecast skill starts to decay.
- **Grid interpolation**: compare native grid-point verification with forecasts
  interpolated onto observation points.
- **Rain and frost events**: ask whether predicted threshold events actually
  happened.
- **Wind reliability**: compare wind speed, gust, and direction error by lead
  window.
- **Spatial error structure**: find whether hard locations are shared across
  variables.
- **Model comparison**: compare model behavior on shared targets rather than
  isolated downloads.

## Project Structure

```text
meteoright/
|-- cli.py                  # Command-line workflow for download, verify, metrics, analysis
|-- src/
|   |-- core/               # High-level pipeline helpers
|   |-- forecast/           # Forecast clients, canonical records, local fetch wrappers
|   `-- historical/         # Historical download, storage, verification, plotting
|-- advanced_verification/  # Event verification, confusion matrices, skill scores
|-- datasets/               # Curated dataset preparation helpers used by the CLI/tests
|-- metrics/, plots/        # Top-level analysis helpers used by the current CLI
|-- showcase/               # Curated publication figures
|-- docs/images/            # Supporting README/documentation figures
|-- tests/                  # Unit and integration tests
`-- data/                   # Local generated datasets, ignored by Git
```

## Data Model

The key invariant is forecast provenance:

```text
forecast_issue_time + forecast_target_time + model + location + variable
```

## Future Directions

- More polished forecast-evolution investigations.
- More operational event investigations.
- Probabilistic and ensemble-style verification.
- Cleaner local dashboards over the same analysis stores.


## Data Attribution

Weather data is provided by [Open-Meteo](https://open-meteo.com/) under the
[CC BY 4.0 license](https://open-meteo.com/en/license).
