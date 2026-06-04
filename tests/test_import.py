"""Smoke tests: verify the package imports cleanly and all public APIs are accessible."""


def test_imports():
    """All public modules import without errors."""
    import downloader
    from downloader.config import DownloadConfig
    from downloader.models import ForecastRecord, ObservationRecord
    from downloader.api import ApiError, fetch_observations, fetch_single_run
    from downloader.normalize import DataError, json_to_forecast_df, json_to_observation_df
    from downloader.storage import (
        list_downloaded,
        write_download_log,
        write_forecasts,
        write_observations,
    )
    from downloader.constants import (
        ARCHIVE_API,
        DEFAULT_VARIABLES,
        MODEL_FORECAST_HOURS,
        MODEL_RUN_CYCLES,
        SINGLE_RUNS_API,
    )

    # Sanity: key symbols are actual objects
    assert DownloadConfig is not None
    assert ForecastRecord is not None
    assert ObservationRecord is not None
    assert fetch_observations is not None
    assert fetch_single_run is not None
    assert ApiError is not None
    assert json_to_observation_df is not None
    assert json_to_forecast_df is not None
    assert DataError is not None
    assert write_forecasts is not None
    assert write_observations is not None
    assert write_download_log is not None
    assert list_downloaded is not None
    assert "archive-api.open-meteo.com" in ARCHIVE_API
    assert "temperature_2m" in DEFAULT_VARIABLES
    assert "ecmwf_ifs_hres" in MODEL_RUN_CYCLES
    assert "ecmwf_ifs_hres" in MODEL_FORECAST_HOURS
