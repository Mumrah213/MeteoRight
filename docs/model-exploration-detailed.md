# Model Exploration — Detailed Results

> **Architecture note**: The public Open-Meteo Single Runs API returns the
> same forecast data (ECMWF IFS) for all model names. Values below are
> identical across models because the public endpoint only serves one model.
> Different models require a self-hosted Open-Meteo backend.

## access_g

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 240h (10 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 1336 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## aifs_0p25_single

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 360h (15 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 220 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## aigfs

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 360h (15 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 222 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## arome_france

- **Run cycles**: [0, 3, 6, 9, 12, 15, 18, 21]
- **Forecast horizon**: 90h (3 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 223 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## arome_france_hd

- **Run cycles**: [0, 3, 6, 9, 12, 15, 18, 21]
- **Forecast horizon**: 72h (3 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 225 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## arpge_europe

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 96h (4 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 223 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## arpge_world

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 96h (4 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 582 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## ecmwf_ifs_0p25

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 360h (15 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 222 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## ecmwf_ifs_0p4

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 360h (15 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 265 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## ecmwf_ifs_hres

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 240h (10 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 231 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## gem_global

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 240h (10 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 230 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## gem_regional

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 240h (10 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 578 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## gfs

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 384h (16 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 236 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## gfs_grapes

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 240h (10 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 2143 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## gfs_graphcast

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 360h (15 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 237 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## gsm

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 264h (11 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 237 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## harmonie_arome_dini

- **Run cycles**: [0, 3, 6, 9, 12, 15, 18, 21]
- **Forecast horizon**: 60h (2 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 227 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## harmonie_arome_europe

- **Run cycles**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
- **Forecast horizon**: 60h (2 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 458 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## harmonie_arome_netherlands

- **Run cycles**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
- **Forecast horizon**: 60h (2 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 226 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## hgefs

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 360h (15 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 237 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## hrdps_continental

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 144h (6 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 217 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## hrrr

- **Run cycles**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
- **Forecast horizon**: 42h (1 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 223 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## icon

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 180h (7 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 217 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## icon_ch1

- **Run cycles**: [0, 3, 6, 9, 12, 15, 18, 21]
- **Forecast horizon**: 120h (5 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 476 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## icon_ch2

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 144h (6 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 225 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## icon_d2

- **Run cycles**: [0, 3, 6, 9, 12, 15, 18, 21]
- **Forecast horizon**: 72h (3 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 223 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## icon_eu

- **Run cycles**: [0, 3, 6, 9, 12, 15, 18, 21]
- **Forecast horizon**: 120h (5 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 229 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## met_nordic

- **Run cycles**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
- **Forecast horizon**: 60h (2 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 247 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## msm

- **Run cycles**: [0, 3, 6, 9, 12, 15, 18, 21]
- **Forecast horizon**: 132h (5 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 2130 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## nam

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 144h (6 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 245 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## nbm

- **Run cycles**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
- **Forecast horizon**: 144h (6 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 1506 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## ukmo_global

- **Run cycles**: [0, 6, 12, 18]
- **Forecast horizon**: 168h (7 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 245 ms
- **Run times tested**: 2
- **Available run times**: 2

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---

## ukmo_ukv

- **Run cycles**: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
- **Forecast horizon**: 24h (1 days)
- **Total forecast hours**: 168
- **Available variables** (3): temperature_2m, precipitation, wind_speed_10m
- **Missing variables** (0): None
- **Data completeness**: 99.8%
- **API latency**: 477 ms
- **Run times tested**: 2
- **Available run times**: 1

### Available Variables

- ✅ temperature_2m
- ✅ precipitation
- ✅ wind_speed_10m

---
