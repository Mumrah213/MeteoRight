# Case studies

Four decisions a forecast buyer actually faces, and where MeteoRight answers them.
Every number below was produced by the commands shown, on the bundled European
dataset: 8 sites, 3 models, leads 24–168 h, twelve months (2025-07-22 → 2026-07-21),
1,471,680 verification rows.

The models are `ecmwf_ifs025` (ECMWF IFS, 0.25°), `icon_seamless` (DWD ICON) and
`gfs_seamless` (NOAA GFS). Observations are ERA5 reanalysis via the Open-Meteo
archive.

---

## From raw API response to analysable table

The first difficulty is that forecast APIs do not return anything you can verify
against. Understanding the shape of that gap is most of the reason the storage
layer looks the way it does.

### What the API returns

A request for two lead times at one location, one model:

```json
{
  "latitude": 55.75,
  "longitude": 12.5,
  "elevation": 10.0,
  "hourly_units": {"temperature_2m_previous_day1": "°C"},
  "hourly": {
    "time": ["2026-03-01T00:00", "2026-03-01T01:00", "2026-03-01T02:00", "..."],
    "temperature_2m_previous_day1": [6.4, 6.2, 5.8, "..."],
    "temperature_2m_previous_day3": [2.2, 1.2, 0.5, "..."]
  }
}
```

Four things are wrong with this as an analysis input:

1. **It is column-oriented, not row-oriented.** Parallel arrays that must be
   zipped by position, with no key tying a value to its time.
2. **Lead time is encoded in the column name.** `previous_day3` means "issued
   three days before the valid time" — a string that has to be parsed into a
   number before it can be grouped or plotted.
3. **There is no issue time at all.** It is implied by the column name and the
   valid time, and must be reconstructed. Without it, no backtest is honest:
   a bid placed at 12:00 has to be judged against what was knowable at 12:00.
4. **The location silently moved.** The request asked for (55.676, 12.568);
   the response is for (55.75, 12.5) — the model's nearest grid cell, 9.3 km
   away. Nothing in the payload flags this.

Model identity is not in the payload either; it exists only in the request.

### What the store holds

`historical/normalize.py` turns the above into rows, and `historical/storage.py`
writes them as Parquet partitioned by model, year and month:

```
data/
├── forecasts/
│   ├── model=ecmwf_ifs025/year=2025/month=07/data.parquet
│   ├── model=icon_seamless/year=2025/month=07/data.parquet
│   └── model=gfs_seamless/…
├── observations/
│   └── year=2025/month=07/data.parquet
└── metadata/
    ├── manifest.json                 # what was planned
    ├── download_checkpoints.jsonl    # what succeeded, for resume
    └── attribution.json              # source and licence
```

A normalised forecast row:

```
      forecast_issue_time      forecast_target_time  lead_hours        model location_id  latitude  longitude  requested_latitude  requested_longitude  location_distance_km  temperature_2m
2025-07-21 00:00:00+00:00 2025-07-22 00:00:00+00:00          24 ecmwf_ifs025  copenhagen     55.75       12.5              55.676               12.568              9.265578            18.3
2025-07-20 00:00:00+00:00 2025-07-22 00:00:00+00:00          48 ecmwf_ifs025  copenhagen     55.75       12.5              55.676               12.568              9.265578            17.7
2025-07-19 00:00:00+00:00 2025-07-22 00:00:00+00:00          72 ecmwf_ifs025  copenhagen     55.75       12.5              55.676               12.568              9.265578            18.4
2025-07-18 00:00:00+00:00 2025-07-22 00:00:00+00:00          96 ecmwf_ifs025  copenhagen     55.75       12.5              55.676               12.568              9.265578            18.0
```

Every implicit thing is now explicit. Issue time and target time are separate
columns, so lead time is *derived* rather than assumed. Model and location are
carried on the row. And the grid displacement is recorded rather than discarded:
`latitude`/`longitude` are where the model actually answered,
`requested_latitude`/`requested_longitude` are where the question was asked, and
`location_distance_km` is the gap.

Observations are normalised the same way, and land on their own grid — note the
coordinates differ from every model's:

```
         observation_time location_id  latitude  longitude  elevation  location_distance_km  temperature_2m  wind_speed_10m  precipitation
2025-07-22 00:00:00+00:00  copenhagen 55.641476  12.596349       10.0              4.230763            18.9            13.6            0.1
2025-07-22 01:00:00+00:00  copenhagen 55.641476  12.596349       10.0              4.230763            18.5            11.3            0.1
2025-07-22 02:00:00+00:00  copenhagen 55.641476  12.596349       10.0              4.230763            18.6            12.6            0.0
```

For Copenhagen the four relevant points are all different: ECMWF answers at
(55.75, 12.50), GFS at (55.70, 12.54), ICON at (55.68, 12.56), and the
observation sits at (55.64, 12.60). Comparing them is a modelling choice, not a
lookup, which is what Case study 4 is about.

### What verification produces

`meteoright verify` joins forecasts to observations on **location and valid
time** and computes errors:

```
      forecast_issue_time      forecast_target_time  lead_hours        model location_id  latitude  longitude  forecast_temperature_2m  observed_temperature_2m  temperature_2m_error
2025-07-21 00:00:00+00:00 2025-07-22 00:00:00+00:00          24 ecmwf_ifs025  copenhagen     55.75       12.5                     18.3                     18.9                  -0.6
2025-07-20 00:00:00+00:00 2025-07-22 00:00:00+00:00          48 ecmwf_ifs025  copenhagen     55.75       12.5                     17.7                     18.9                  -1.2
2025-07-19 00:00:00+00:00 2025-07-22 00:00:00+00:00          72 ecmwf_ifs025  copenhagen     55.75       12.5                     18.4                     18.9                  -0.5
2025-07-18 00:00:00+00:00 2025-07-22 00:00:00+00:00          96 ecmwf_ifs025  copenhagen     55.75       12.5                     18.0                     18.9                  -0.9
```

One row per (issue time, valid time, model, location), forecast and observation
side by side. The four rows above are four *different forecasts of the same
hour*, issued one to four days ahead — which is what makes skill decay
measurable at all.

---

## Case study 1 — Frost warnings for winter road maintenance

**The decision.** A road authority treats a route when frost is forecast.
Treating unnecessarily wastes material and crew time; missing a frost event
risks accidents and liability. The question is not "what is the temperature
error" but "how often does the forecast catch real frost, and how often does it
cry wolf."

**Where it is answered.** `advanced_verification/events.py` converts continuous
forecasts into binary events at a threshold (frost = `temperature_2m ≤ 0 °C`),
`confusion.py` builds the contingency table, and `skill_scores.py` computes hit
rate, false alarm rate, precision and CSI.

```bash
meteoright advanced compare \
  --verification examples/europe_multimodel/verification/verification.parquet \
  --event frost --models ecmwf_ifs025,gfs_seamless,icon_seamless \
  --group-by lead_hours --metric hit_rate
```

**The finding — the models fail in opposite directions.**

| Lead | ECMWF hit rate | ICON hit rate | ECMWF precision | ICON precision |
|-----:|---------------:|--------------:|----------------:|---------------:|
|  24 h | **0.917** | 0.843 | 0.917 | **0.953** |
|  72 h | **0.894** | 0.842 | 0.880 | **0.917** |
| 120 h | **0.862** | 0.717 | 0.818 | **0.847** |

ECMWF catches more of the frost that happens. ICON is right more often when it
warns. These point at different providers, and the tie is broken by the cost
ratio rather than by the meteorology: if a missed frost costs more than a wasted
treatment — the usual case for road safety — ECMWF is correct despite lower
precision. If treatment is expensive and misses are tolerable, ICON is.

Overall skill (critical success index) favours ECMWF at every lead:

| Lead | ECMWF | ICON | GFS |
|-----:|------:|-----:|----:|
|  24 h | **0.846** | 0.809 | 0.776 |
|  72 h | **0.797** | 0.783 | 0.737 |
| 120 h | **0.723** | 0.635 | 0.651 |
| 168 h | **0.620** | n/a | 0.593 |

False alarm rates are low and similar across models (0.005–0.030), so the
practical trade-off is hit rate against precision, not against nuisance alerts.

This is the clearest case of verification answering a business question rather
than a meteorological one: the same measurements support opposite procurement
decisions, and the platform reports both sides instead of collapsing them into
a single score.

---

## Case study 2 — Day-ahead bidding and the value of a second provider

**The decision.** A bid placed at 12:00 for delivery the next day is settled
against what actually happened, and wind-speed error drives the imbalance cost.
Is one forecast contract enough, or does a second pay for itself?

**Where it is answered.** `meteoright/cli/blend.py` learns per-model skill on an
earlier period and scores the blend only on a later one, so the improvement is a
forward-looking claim rather than a description of the past.
`src/forecast/meta_forecasting/weighting.py` produces the weights, each with a
plain-language rationale.

```bash
meteoright blend \
  --verification examples/europe_multimodel/verification/verification.parquet \
  --variable wind_speed_10m --split temporal
```

**The finding — yes, but buy two, not three.** Weights were learned on
2025-07 → 2026-01 and evaluated on 2026-01 → 2026-07, roughly 210,000 held-out
targets per variable:

| Variable | Best single | Blend | Improvement | Weights |
|---|---|---|---|---|
| temperature_2m | 1.223 °C (ECMWF) | **1.119** | **−8.5 %** | ECMWF 0.756 / ICON 0.244 |
| wind_speed_10m | 3.295 m/s (ECMWF) | **3.168** | **−3.9 %** | ECMWF 0.892 / GFS 0.108 |
| precipitation | 0.113 mm (GFS) | **0.108** | **−4.5 %** | GFS 0.772 / ECMWF 0.228 |

In every case **one of the three models received zero weight**, and the
surviving pair changed by variable. The procurement answer is "buy two, and
which two depends on what you are forecasting" — not "buy everything and average
it." A third source added nothing the better two did not already provide.

A 3.9 % reduction in wind error is the number to weigh against the cost of a
second contract. Whether it clears that bar depends on the imbalance price and
the volume, which the platform does not know and deliberately does not guess.

---

## Case study 3 — How far ahead is the forecast worth acting on?

**The decision.** Planning horizons need a defensible cutoff. Past some lead
time a forecast stops informing the decision, and treating it as if it still
does is worse than admitting ignorance.

**Where it is answered.** `analysis/degradation.py` —
`compute_half_skill_lead_time()` finds where error has doubled from its minimum,
`compute_degradation_rate()` fits the decay slope. `advanced_verification/baselines.py`
scores forecasts against a climatology baseline.

```bash
meteoright metrics \
  --verification examples/europe_multimodel/verification/verification.parquet \
  --group-by model,lead_hours --metrics mae,rmse,bias \
  --variables temperature_2m,wind_speed_10m,precipitation \
  --output examples/europe_multimodel/metrics
```

**The finding — skill halves at six days, and the ranking is stable until it
does not.**

ECMWF's half-skill lead time is **144 hours** for both temperature and wind:
error doubles from its 24 h minimum by day six. Decay is close to linear, at
0.0096 °C/h for temperature and 0.021 m/s/h for wind.

| Lead | Temp MAE (ECMWF) | Wind MAE (ECMWF) |
|-----:|-----------------:|-----------------:|
|  24 h | 0.750 °C | 2.247 m/s |
|  72 h | 1.009 °C | 2.924 m/s |
| 120 h | 1.473 °C | 4.009 m/s |
| 168 h | 2.112 °C | 5.187 m/s |

For temperature and wind the ordering of the three models does not change across
the whole window, so the horizon does not affect the procurement answer.

**Precipitation is the exception, and it matters.** ECMWF leads to 96 h, after
which GFS takes over:

| Lead | ECMWF | GFS | Best |
|-----:|------:|----:|------|
|  24 h | **0.104** | 0.116 | ECMWF |
|  96 h | **0.142** | 0.145 | ECMWF |
| 120 h | 0.156 | **0.123** | GFS |
| 168 h | 0.166 | **0.123** | GFS |

A five-day precipitation decision and a one-day one should be sourced
differently. This is the one place in the dataset where "which provider is best"
genuinely depends on how far ahead you are looking.

---

## Case study 4 — Does the grid point you are given match the site you care about?

**The decision.** A forecast is bought for a specific location — a wind farm, a
depot, a field. The provider returns its nearest grid cell, which may be
kilometres away. Does that displacement matter enough to pay for higher
resolution?

**Where it is answered.** `verification/interpolate.py` implements nearest,
bilinear and inverse-distance weighting onto the observation location;
`interpolate_frames.py` applies it across a whole dataset, per model and per
site.

```bash
meteoright verify --data-dir examples/europe_multimodel/data \
  --variables temperature_2m,wind_speed_10m,precipitation \
  --interpolate bilinear --output examples/europe_multimodel/verification_bilinear
```

**What is established.** Grid displacement varies systematically by provider:

| Site | ECMWF | GFS | ICON |
|---|---:|---:|---:|
| Bergen | 12.59 km | 2.85 km | 2.12 km |
| Copenhagen | 9.27 km | 3.65 km | 0.67 km |
| Paris | 14.05 km | 5.91 km | 0.67 km |
| Zurich | 14.04 km | 1.46 km | 0.37 km |

ECMWF sits 6.8–14.1 km from the requested points across the eight sites, while
ICON is frequently under a kilometre. ECMWF nonetheless wins temperature and
wind at 7 of 8 sites — **despite carrying the largest handicap in the
comparison.** The two sites where ICON does win (Copenhagen temperature, Zurich
wind) are precisely where ICON's grid is closest to the target.

That suggests the native-grid comparison may understate ECMWF. Testing it needs
the four grid cells surrounding each site rather than the single cell each model
snaps to, so `find_grid_corners.py` locates them (probing the API to discover
each model's grid spacing) and `build_grid_dataset.py` fetches them — 94 corner
points, 1,222 requests. Interpolating all three models onto the observation
location and rescoring gives:

| Variable | Model | Native MAE | Bilinear MAE | Change |
|---|---|---:|---:|---:|
| temperature_2m | ECMWF | 1.321 | 1.362 | **+3.0 %** |
| | ICON | 1.432 | 1.378 | −3.8 % |
| | GFS | 1.727 | 1.731 | +0.2 % |
| wind_speed_10m | ECMWF | 3.578 | 3.400 | −5.0 % |
| | ICON | 3.981 | 3.682 | **−7.5 %** |
| | GFS | 4.196 | 4.143 | −1.2 % |
| precipitation | ECMWF | 0.140 | 0.138 | −1.3 % |
| | ICON | 0.141 | 0.140 | −0.3 % |
| | GFS | 0.127 | 0.127 | +0.0 % |

**The ranking does not change.** Every model holds its position on every
variable: ECMWF first on temperature and wind, GFS first on precipitation. The
hypothesis that ECMWF's win was a grid-resolution artifact is not supported —
correcting the handicap leaves the ordering intact, which makes the procurement
answer in Case study 1–3 more robust, not less.

Two secondary findings are worth noting. Interpolation helps wind most
(−1.2 % to −7.5 %), which fits: wind is the most spatially variable of the three
fields, so displacement costs the most. And it *hurts* ECMWF's temperature
slightly (+3.0 %) — averaging four cells across a 0.25° span smooths away real
local structure that the single nearest cell happened to capture. Interpolation
is not free, and it is not uniformly an improvement.

The practical answer to the buyer's question: for these sites and variables, the
grid displacement is worth roughly 1–7 % of error on wind and essentially
nothing on temperature and precipitation. That is the number to weigh against a
higher-resolution contract.

---

## Reproducing these numbers

```bash
# 1. Download the dataset (416 requests, checkpointed and resumable)
python examples/europe_multimodel/build_dataset.py

# 2. Build the verification table
meteoright verify --data-dir examples/europe_multimodel/data \
  --variables temperature_2m,wind_speed_10m,precipitation \
  --output examples/europe_multimodel/verification

# 3. Case study 1 — event verification
meteoright advanced compare --verification examples/europe_multimodel/verification/verification.parquet \
  --event frost --models ecmwf_ifs025,gfs_seamless,icon_seamless \
  --group-by lead_hours --metric csi --output out/frost

# 4. Case study 2 — blending, temporal split
meteoright blend --verification examples/europe_multimodel/verification/verification.parquet \
  --variable temperature_2m --split temporal

# 5. Case study 3 — skill decay
meteoright metrics --verification examples/europe_multimodel/verification/verification.parquet \
  --group-by model,lead_hours --metrics mae,rmse,bias \
  --variables temperature_2m,wind_speed_10m,precipitation --output examples/europe_multimodel/metrics
```

Data: Open-Meteo (CC BY 4.0).
