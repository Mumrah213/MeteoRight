"""Constants for Open-Meteo API access.

This module is a thin re-export of :mod:`downloader.constants`, which is the
single source of truth for endpoint resolution (public vs. self-hosted backend)
and model metadata. It exists for backwards compatibility with code that imports
``historical.constants``.
"""

from downloader.constants import (  # noqa: F401
    ARCHIVE_API,
    DEFAULT_API_SLEEP,
    DEFAULT_VARIABLES,
    HISTORICAL_FORECAST_API,
    MODEL_FORECAST_HOURS,
    MODEL_RUN_CYCLES,
    OPEN_METEO_API_KEY,
    OPEN_METEO_BASE_URL,
    PREVIOUS_RUNS_API,
    PUBLIC_ARCHIVE_API,
    PUBLIC_HISTORICAL_FORECAST_API,
    PUBLIC_PREVIOUS_RUNS_API,
    PUBLIC_SINGLE_RUNS_API,
    SINGLE_RUNS_API,
)
