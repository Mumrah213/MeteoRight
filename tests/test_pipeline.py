"""Pipeline tests — essential cases only."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import PipelineConfig
from src.core.pipeline import (
    download_forecast,
    download_historical,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def tmp_data(tmp_path):
    """Create a temporary data directory with expected structure."""
    data_dir = tmp_path / "data"
    for subdir in [
        "raw/observations",
        "raw/forecasts/model=icon_eu/year=2024/month=1",
        "intermediate",
        "cache",
    ]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    return data_dir


# ===========================================================================
# Test: Configuration
# ===========================================================================


def test_config_loads():
    """PipelineConfig loads with defaults."""
    config = PipelineConfig()
    assert config.get_location("copenhagen") is not None
    assert config.get_variables("minimal") == ("temperature_2m", "precipitation")
    assert config.http.base_url == "https://api.open-meteo.com"


def test_backward_compat():
    """Legacy settings still work."""
    from src.config import LOCATION_PRESETS, VARIABLE_SETS

    assert "copenhagen" in LOCATION_PRESETS
    assert "minimal" in VARIABLE_SETS


# ===========================================================================
# Test: Historical Ingestion
# ===========================================================================


@patch("src.core.pipeline.PipelineOrchestrator")
def test_download_historical(mock_orch_class, tmp_data):
    """Download historical observations works end-to-end."""
    mock_orch = MagicMock()
    mock_orch.run_ingestion.return_value = {
        "observation_rows": 100,
        "forecast_rows": 0,
        "models_ingested": [],
        "errors": [],
    }
    mock_orch_class.return_value = mock_orch

    result = download_historical(
        location={"name": "Copenhagen", "lat": 55.605, "lon": 12.574},
        start_date="2024-01-01",
        end_date="2024-01-01",
        output_dir=tmp_data,
    )

    assert result["observation_rows"] == 100
    mock_orch.run_ingestion.assert_called_once()


# ===========================================================================
# Test: Forecast Ingestion
# ===========================================================================


@patch("src.core.pipeline.PipelineOrchestrator")
def test_download_forecast(mock_orch_class, tmp_data):
    """Download forecast data works end-to-end."""
    mock_orch = MagicMock()
    mock_orch.run_ingestion.return_value = {
        "forecast_rows": 500,
        "observation_rows": 0,
        "models_ingested": ["ecmwf_ifs_single"],
        "errors": [],
    }
    mock_orch_class.return_value = mock_orch

    result = download_forecast(
        location={"name": "Copenhagen", "lat": 55.605, "lon": 12.574},
        start_date="2024-01-01",
        end_date="2024-01-01",
        output_dir=tmp_data,
    )

    assert result["forecast_rows"] == 500
    assert "ecmwf_ifs_single" in result["models_ingested"]
    mock_orch.run_ingestion.assert_called_once()
