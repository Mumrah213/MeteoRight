"""Smoke tests: verify the package imports cleanly and all public APIs are accessible."""


def test_imports():
    """All public modules import without errors."""
    import downloader
    from downloader.api import ApiError, fetch_observations, fetch_single_run
    from downloader.config import DownloadConfig
    from downloader.constants import (
        ARCHIVE_API,
        DEFAULT_VARIABLES,
        MODEL_FORECAST_HOURS,
        MODEL_RUN_CYCLES,
        SINGLE_RUNS_API,
    )
    from downloader.models import ForecastRecord, ObservationRecord
    from downloader.normalize import DataError, json_to_forecast_df, json_to_observation_df
    from downloader.storage import (
        list_downloaded,
        write_download_log,
        write_forecasts,
        write_observations,
    )

    # Every public symbol imported without error.
    public_symbols = [
        downloader, DownloadConfig, ForecastRecord, ObservationRecord,
        ApiError, fetch_observations, fetch_single_run,
        DataError, json_to_forecast_df, json_to_observation_df,
        list_downloaded, write_download_log, write_forecasts, write_observations,
        SINGLE_RUNS_API,
    ]
    assert all(symbol is not None for symbol in public_symbols)

    # Spot check a few constants so they can't silently change.
    assert "archive-api.open-meteo.com" in ARCHIVE_API
    assert "temperature_2m" in DEFAULT_VARIABLES
    assert "ecmwf_ifs_hres" in MODEL_RUN_CYCLES
    assert "ecmwf_ifs_hres" in MODEL_FORECAST_HOURS
