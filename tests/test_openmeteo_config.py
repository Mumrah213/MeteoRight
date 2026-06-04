"""Open-Meteo backend configuration behavior."""


import importlib

import pytest


@pytest.fixture(autouse=True)
def restore_open_meteo_constants(monkeypatch):
    """Leave endpoint constants in public-default state after each test."""
    yield
    monkeypatch.delenv("METEORIGHT_OPEN_METEO_BASE_URL", raising=False)
    monkeypatch.delenv("METEORIGHT_OPEN_METEO_ARCHIVE_API", raising=False)
    monkeypatch.delenv("METEORIGHT_OPEN_METEO_SINGLE_RUNS_API", raising=False)
    monkeypatch.delenv("METEORIGHT_OPEN_METEO_API_KEY", raising=False)

    import downloader.constants as downloader_constants
    import historical.constants as historical_constants

    importlib.reload(downloader_constants)
    importlib.reload(historical_constants)


def test_downloader_uses_public_endpoints_by_default(monkeypatch):
    """Without env overrides, downloads target the public Open-Meteo APIs."""
    monkeypatch.delenv("METEORIGHT_OPEN_METEO_BASE_URL", raising=False)
    monkeypatch.delenv("METEORIGHT_OPEN_METEO_ARCHIVE_API", raising=False)
    monkeypatch.delenv("METEORIGHT_OPEN_METEO_SINGLE_RUNS_API", raising=False)

    import downloader.constants as constants

    constants = importlib.reload(constants)

    assert constants.ARCHIVE_API == constants.PUBLIC_ARCHIVE_API
    assert constants.SINGLE_RUNS_API == constants.PUBLIC_SINGLE_RUNS_API


def test_downloader_can_use_self_hosted_base_url(monkeypatch):
    """A single base URL maps archive and single-runs paths to the local backend."""
    monkeypatch.setenv("METEORIGHT_OPEN_METEO_BASE_URL", "http://localhost:8080")
    monkeypatch.delenv("METEORIGHT_OPEN_METEO_ARCHIVE_API", raising=False)
    monkeypatch.delenv("METEORIGHT_OPEN_METEO_SINGLE_RUNS_API", raising=False)

    import downloader.constants as constants

    constants = importlib.reload(constants)

    assert constants.ARCHIVE_API == "http://localhost:8080/v1/archive"
    assert constants.SINGLE_RUNS_API == "http://localhost:8080/v1/forecast"


def test_downloader_endpoint_override_wins(monkeypatch):
    """Dedicated endpoint overrides win over the generic self-hosted base URL."""
    monkeypatch.setenv("METEORIGHT_OPEN_METEO_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("METEORIGHT_OPEN_METEO_ARCHIVE_API", "http://archive.local/v1/archive")

    import downloader.constants as constants

    constants = importlib.reload(constants)

    assert constants.ARCHIVE_API == "http://archive.local/v1/archive"


def test_api_key_is_added_as_open_meteo_apikey_parameter():
    """Open-Meteo expects the commercial API key under the apikey parameter."""
    from downloader.api import _build_params

    params = _build_params(
        55.605,
        12.574,
        ("temperature_2m",),
        extra={"start_date": "2026-05-20", "end_date": "2026-05-22"},
        api_key="secret",
    )

    assert params["apikey"] == "secret"


def test_forecast_http_client_uses_env_base_url_and_api_key(monkeypatch):
    """The provider transport follows the same backend and API-key env vars."""
    monkeypatch.setenv("METEORIGHT_OPEN_METEO_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("METEORIGHT_OPEN_METEO_API_KEY", "secret")

    from forecast.http_client import HttpClient

    client = HttpClient()
    url = client._build_url("/v1/forecast", {"latitude": 55.605})

    assert url.startswith("http://localhost:8080/v1/forecast?")
    assert "apikey=secret" in url
