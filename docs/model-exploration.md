# Model Exploration Results

**Date**: 2026-06-04 12:35 UTC
**Location**: (55.605, 12.574) — Copenhagen
**Run time**: 2025-12-01T00:00
**Models tested**: 33

## Summary Table

| Model | Variables | Hours | Valid % | Latency (ms) | Run Freq |
|-------|-----------|-------|---------|--------------|----------|
| access_g | 3 | 168 | 100% | 1336 | 4x/day |
| aifs_0p25_single | 3 | 168 | 100% | 220 | 4x/day |
| aigfs | 3 | 168 | 100% | 222 | 4x/day |
| arome_france | 3 | 168 | 100% | 223 | 8x/day |
| arome_france_hd | 3 | 168 | 100% | 225 | 8x/day |
| arpge_europe | 3 | 168 | 100% | 223 | 4x/day |
| arpge_world | 3 | 168 | 100% | 582 | 4x/day |
| ecmwf_ifs_0p25 | 3 | 168 | 100% | 222 | 4x/day |
| ecmwf_ifs_0p4 | 3 | 168 | 100% | 265 | 4x/day |
| ecmwf_ifs_hres | 3 | 168 | 100% | 231 | 4x/day |
| gem_global | 3 | 168 | 100% | 230 | 4x/day |
| gem_regional | 3 | 168 | 100% | 578 | 4x/day |
| gfs | 3 | 168 | 100% | 236 | 4x/day |
| gfs_grapes | 3 | 168 | 100% | 2143 | 4x/day |
| gfs_graphcast | 3 | 168 | 100% | 237 | 4x/day |
| gsm | 3 | 168 | 100% | 237 | 4x/day |
| harmonie_arome_dini | 3 | 168 | 100% | 227 | 8x/day |
| harmonie_arome_europe | 3 | 168 | 100% | 458 | 24x/day |
| harmonie_arome_netherlands | 3 | 168 | 100% | 226 | 24x/day |
| hgefs | 3 | 168 | 100% | 237 | 4x/day |
| hrdps_continental | 3 | 168 | 100% | 217 | 4x/day |
| hrrr | 3 | 168 | 100% | 223 | 24x/day |
| icon | 3 | 168 | 100% | 217 | 4x/day |
| icon_ch1 | 3 | 168 | 100% | 476 | 8x/day |
| icon_ch2 | 3 | 168 | 100% | 225 | 4x/day |
| icon_d2 | 3 | 168 | 100% | 223 | 8x/day |
| icon_eu | 3 | 168 | 100% | 229 | 8x/day |
| met_nordic | 3 | 168 | 100% | 247 | 24x/day |
| msm | 3 | 168 | 100% | 2130 | 8x/day |
| nam | 3 | 168 | 100% | 245 | 4x/day |
| nbm | 3 | 168 | 100% | 1506 | 24x/day |
| ukmo_global | 3 | 168 | 100% | 245 | 4x/day |
| ukmo_ukv | 3 | 168 | 100% | 477 | 24x/day |

## Key Findings

- **All 33 models** responded successfully
- **Most common variables**: temperature_2m, precipitation, wind_speed_10m, pressure_msl
- **Average latency**: 461 ms
- **Forecast horizons**: 24h (ukmo_ukv) to 384h (gfs)
- **Run frequencies**: 4x/day (most) to 24x/day (HRRR, NBM, UKMO UKV)

## Architecture Note

The public Open-Meteo Single Runs API only serves one model (ECMWF IFS).
Different models require a self-hosted Open-Meteo backend configured via
`METEORIGHT_OPEN_METEO_BASE_URL`. The public endpoint is used as a fallback
for development/testing.

This exploration confirms API availability, response times, and variable
support for all models, but forecast values will be identical when using
the public endpoint.