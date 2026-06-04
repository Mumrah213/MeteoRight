"""Constants for Open-Meteo API access."""

import os

# Public Open-Meteo endpoints used when no self-hosted backend is configured.
PUBLIC_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
PUBLIC_SINGLE_RUNS_API = "https://single-runs-api.open-meteo.com/v1/forecast"

# API base URLs.
#
# Set METEORIGHT_OPEN_METEO_BASE_URL to route all requests to a self-hosted
# Open-Meteo backend (e.g. http://localhost:8080 for local development).
# Per-endpoint variables (METEORIGHT_OPEN_METEO_ARCHIVE_API, etc.) take
# precedence when set.
OPEN_METEO_BASE_URL = os.getenv("METEORIGHT_OPEN_METEO_BASE_URL")
OPEN_METEO_API_KEY = os.getenv("METEORIGHT_OPEN_METEO_API_KEY")


def _endpoint_from_base(path: str, default: str) -> str:
    if not OPEN_METEO_BASE_URL:
        return default
    return f"{OPEN_METEO_BASE_URL.rstrip('/')}{path}"


ARCHIVE_API = os.getenv("METEORIGHT_OPEN_METEO_ARCHIVE_API") or _endpoint_from_base(
    "/v1/archive", PUBLIC_ARCHIVE_API
)
SINGLE_RUNS_API = os.getenv("METEORIGHT_OPEN_METEO_SINGLE_RUNS_API") or _endpoint_from_base(
    "/v1/forecast", PUBLIC_SINGLE_RUNS_API
)

# Default variables
DEFAULT_VARIABLES = ("temperature_2m", "precipitation")

# Known model run frequencies (hours between runs)
# Format: model_name -> list of UTC hours when runs are issued
MODEL_RUN_CYCLES: dict[str, list[int]] = {
    "ecmwf_ifs_hres": [0, 6, 12, 18],
    "ecmwf_ifs_0p25": [0, 6, 12, 18],
    "ecmwf_ifs_0p4": [0, 6, 12, 18],
    "gfs": [0, 6, 12, 18],
    "icon": [0, 6, 12, 18],
    "icon_eu": [0, 3, 6, 9, 12, 15, 18, 21],
    "icon_d2": [0, 3, 6, 9, 12, 15, 18, 21],
    "hrrr": list(range(24)),
    "nam": [0, 6, 12, 18],
    "nbm": list(range(24)),
    "arpge_world": [0, 6, 12, 18],
    "arpge_europe": [0, 6, 12, 18],
    "arome_france": [0, 3, 6, 9, 12, 15, 18, 21],
    "arome_france_hd": [0, 3, 6, 9, 12, 15, 18, 21],
    "ukmo_global": [0, 6, 12, 18],
    "ukmo_ukv": list(range(24)),
    "gsm": [0, 6, 12, 18],
    "msm": [0, 3, 6, 9, 12, 15, 18, 21],
    "met_nordic": list(range(24)),
    "gem_global": [0, 6, 12, 18],
    "gem_regional": [0, 6, 12, 18],
    "hrdps_continental": [0, 6, 12, 18],
    "gfs_grapes": [0, 6, 12, 18],
    "access_g": [0, 6, 12, 18],
    "harmonie_arome_europe": list(range(24)),
    "harmonie_arome_netherlands": list(range(24)),
    "harmonie_arome_dini": [0, 3, 6, 9, 12, 15, 18, 21],
    "aifs_0p25_single": [0, 6, 12, 18],
    "gfs_graphcast": [0, 6, 12, 18],
    "aigfs": [0, 6, 12, 18],
    "hgefs": [0, 6, 12, 18],
    "icon_ch1": [0, 3, 6, 9, 12, 15, 18, 21],
    "icon_ch2": [0, 6, 12, 18],
}

# Forecast horizon in hours per model (approximate)
MODEL_FORECAST_HOURS: dict[str, int] = {
    "ecmwf_ifs_hres": 240,  # 10 days
    "ecmwf_ifs_0p25": 360,  # 15 days
    "ecmwf_ifs_0p4": 360,
    "gfs": 384,  # 16 days
    "icon": 180,  # ~7.5 days
    "icon_eu": 120,  # 5 days
    "icon_d2": 72,  # 3 days
    "hrrr": 42,  # ~1.75 days
    "nam": 144,  # 6 days
    "nbm": 144,
    "arpge_world": 96,  # 4 days
    "arpge_europe": 96,
    "arome_france": 90,  # ~3.75 days
    "arome_france_hd": 72,
    "ukmo_global": 168,  # 7 days
    "ukmo_ukv": 24,
    "gsm": 264,  # 11 days
    "msm": 132,  # ~5.5 days
    "met_nordic": 60,  # 2.5 days
    "gem_global": 240,  # 10 days
    "gem_regional": 240,
    "hrdps_continental": 144,  # 6 days
    "gfs_grapes": 240,
    "access_g": 240,  # 10 days
    "harmonie_arome_europe": 60,  # 2.5 days
    "harmonie_arome_netherlands": 60,
    "harmonie_arome_dini": 60,
    "aifs_0p25_single": 360,
    "gfs_graphcast": 360,  # 6-hourly
    "aigfs": 360,  # 6-hourly
    "hgefs": 360,  # 6-hourly
    "icon_ch1": 120,  # 5 days
    "icon_ch2": 144,  # 6 days
}

# Default sleep between API requests (seconds)
DEFAULT_API_SLEEP = 0.5
