# Showcase Figures

Curated publication figures for MeteoRight.

These images are selected to show the current analysis direction: grid
verification, interpolation, event verification, lead-time degradation, and
cross-variable error structure. They are not meant to include every generated
plot.

## Core Figures

- `wind_composite.png` is the full verification composite: error against lead
  time for speed, gusts and direction; where wind-speed error sits at three
  lead bands on a shared scale; and how much each site's error depends on lead
  time. Regenerate with `python showcase/make_wind_composite_figure.py`.
- `interpolation_schematic.png` explains the geometry of bilinear
  interpolation on one idealised grid cell. It is a diagram rather than a
  measurement, and says so. Regenerate with
  `python showcase/make_interpolation_schematic.py`.
- `interpolation_map.png` is the spatial view: model grid points against the
  observation sites they are verified at, where the gain lands on the map, and
  what predicts it. Regenerate with
  `python showcase/make_interpolation_map_figure.py`.
- `interpolation_effect.png` compares native-grid verification against
  verification after forecasts are interpolated onto the observation location,
  across 23 sites. Regenerate with
  `python showcase/make_interpolation_figure.py` — it reads the bundled grid
  sample, so it needs no downloads.
- `blend_vs_best_single.png` compares an adaptively blended forecast against
  each single model on held-out targets, for three variables. Regenerate it
  with `python showcase/make_blend_figure.py` — it reads the bundled
  verification sample, so it needs no downloads.

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

This creates standard lead-time and error-distribution figures. The curated
figures in this folder apply the same verification idea to a larger local grid
run, bundled as `examples/copenhagen_grid_sample`.

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

- `event_verification.png` turns the forecast into a yes/no question — was the
  hour dry? — and reports the confusion matrix, per-model skill, and how both
  move with lead time. Regenerate with
  `python showcase/make_event_figure.py`.

  The bundled sample is May 2026, so it carries no frost: a frost figure would
  need a winter download. Switch `EVENT` in the script to `frost` once you
  have one.

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

- `wind_error_structure.png` combines three views of the same error field: the
  error distribution native vs interpolated, error percentiles by lead time,
  and whether sites that are hard for one variable are hard for others.
  Regenerate with `python showcase/make_error_structure_figure.py`.

## Monthly Skill

- `monthly_lead_skill.png` shows wind-speed MAE by target month and lead time,
  native and interpolated on a shared colour scale. Regenerate with
  `python showcase/make_monthly_heatmap_figure.py`.
