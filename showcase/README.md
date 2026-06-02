# Showcase Figures

Curated publication figures for MeteoRight.

These images are selected to show the current analysis direction: grid
verification, interpolation, event verification, lead-time degradation, and
cross-variable error structure. They are not meant to include every generated
plot.

## Core Figures

- `wind_composite_interpolated_linear_nearest.png` shows wind-speed, gust, and
  direction verification across lead time and space after grid interpolation.
- `wind_composite_forecast_observation_grid.png` shows the forecast and
  observation grid points used before interpolation.
- `wind_interpolation_effect.png` compares native grid-point verification with
  interpolated verification.
- `wind_interpolation_schematic.png` explains why interpolation changes the
  verification question.

### Reproduce the Basic Lead-Time Plot

Start with a small local download, build verification rows, compute metrics,
then ask the CLI for lead-time and distribution plots:

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

meteoright verify \
  --data-dir data/copenhagen_demo \
  --variables temperature_2m,precipitation,wind_speed_10m \
  --output data/copenhagen_demo/verification

meteoright metrics \
  --verification "data/copenhagen_demo/verification/*.parquet" \
  --group-by lead_hours,model \
  --variables temperature_2m,precipitation,wind_speed_10m \
  --output data/copenhagen_demo/metrics

meteoright analyze \
  --metrics "data/copenhagen_demo/metrics/*.parquet" \
  --verification "data/copenhagen_demo/verification/*.parquet" \
  --variables temperature_2m,precipitation,wind_speed_10m \
  --plots lead_time,distributions \
  --generate-tables \
  --output data/copenhagen_demo/analysis
```

This creates standard lead-time and error-distribution figures. The polished
composite figures in this folder use the same verification idea on larger local
grid runs.

### Find Grid Points for a Spatial Run

Use this before a grid investigation to see which model and observation points
surround the target location:

```bash
meteoright grid-points \
  --lat 55.605 --lon 12.574 \
  --radius-km 50 \
  --summary
```

## Event Verification

- `precipitation_event_confusion.png` asks whether predicted rain events
  happened, with and without interpolation.
- `frost_event_confusion.png` applies the same event-verification framing to
  frost.
- `precipitation_event_investigation.png` shows a precipitation-focused event
  investigation.

### Reproduce an Event Verification Table

Once verification rows exist, event verification turns a forecast question into
hits, misses, false alarms, and correct negatives:

```bash
meteoright advanced events \
  --verification "data/copenhagen_demo/verification/*.parquet" \
  --event no_rain \
  --group-by lead_hours,model \
  --output data/copenhagen_demo/events
```

For frost, switch the event:

```bash
meteoright advanced events \
  --verification "data/copenhagen_demo/verification/*.parquet" \
  --event frost \
  --group-by lead_hours,model \
  --output data/copenhagen_demo/events
```

The showcase event figures build on this workflow by arranging the confusion
counts across lead windows and comparing native-grid verification with
interpolated verification.

## Error Structure

- `wind_error_percentile_heatmap_interpolated.png` shows the distribution of
  wind errors by percentile and lead time.
- `wind_error_quantiles_interpolated.png` summarizes wind-error tails across
  lead windows.
- `error_correlation_matrix_interpolated_composite.png` shows whether locations
  that are hard for one variable are also hard for others.

## Surface Variables

- `surface_composite_interpolated_linear_nearest.png` extends the grid-analysis
  framing to temperature, relative humidity, and precipitation.
- `surface_composite_forecast_observation_grid.png` shows the same variables
  before interpolation.

## Monthly Skill

- `monthly_lead_mae_heatmap_interpolated_linear_nearest.png` shows how
  lead-time error varies by month after interpolation.
- `monthly_lead_mae_heatmap.png` is the native-grid comparison.
