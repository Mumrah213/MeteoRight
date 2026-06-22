"""Offline integration test for the forecast_accuracy orchestrator.

Mocks the backend HTTP (forecast + archive) via httpx.MockTransport and injects
a describe_backend() result, so the full chain — area resolution, window
selection, fetch, circular-aware error, metrics — runs without network.
"""

import httpx
import pytest

from agent_tools import accuracy, download


def _hourly_payload(times, **series):
    return {"hourly": {"time": list(times), **{k: list(v) for k, v in series.items()}}}


def _times(n):
    # n hourly ISO timestamps starting at a fixed instant.
    base = "2026-05-10T"
    return [f"{base}{h:02d}:00" for h in range(n)]


@pytest.fixture
def fake_backend_desc():
    return {
        "base_url": "http://test/v1/forecast",
        "reachable": True,
        "models": [
            {"backend_name": "ecmwf_ifs025", "has_data": True,
             "coverage": {"start": "2026-05-07", "end": "2026-05-18"}},
        ],
        "observations": {
            "source": "archive", "url": "http://test/v1/archive",
            "coverage": {"start": "2026-05-07", "end": "2026-05-18"},
        },
        "error": None,
    }


def _install_http(monkeypatch, forecast_series, obs_series, variable):
    times = _times(len(forecast_series))

    def handler(request: httpx.Request) -> httpx.Response:
        if "archive" in str(request.url):
            return httpx.Response(200, json=_hourly_payload(times, **{variable: obs_series}))
        return httpx.Response(200, json=_hourly_payload(times, **{variable: forecast_series}))

    monkeypatch.setattr(
        download, "_make_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_temperature_accuracy_offline(monkeypatch, fake_backend_desc):
    # forecast - observed = [1, -1, 2] -> MAE = 4/3, bias = 2/3
    _install_http(monkeypatch, [11.0, 9.0, 12.0], [10.0, 10.0, 10.0], "temperature_2m")
    r = accuracy.forecast_accuracy(
        "temperature_2m", ["Copenhagen", "Malmö"],
        backend_desc=fake_backend_desc, use_network_geocode=False,
    )
    assert r["error"] is None
    ans = r["answer"]
    assert ans["sample_size"] == 3
    assert ans["metrics"]["mae"] == pytest.approx(4 / 3, abs=1e-4)
    assert ans["metrics"]["bias"] == pytest.approx(2 / 3, abs=1e-4)
    assert ans["is_circular"] is False
    assert ans["units"] == "°C"


def test_wind_direction_uses_circular_error_offline(monkeypatch, fake_backend_desc):
    # Forecasts straddle the 0/360 seam vs observations. Shortest angles are all 20.
    _install_http(monkeypatch, [10.0, 350.0, 5.0], [350.0, 10.0, 345.0], "wind_direction_10m")
    r = accuracy.forecast_accuracy(
        "wind_direction_10m", "Malmö-Copenhagen",
        backend_desc=fake_backend_desc, use_network_geocode=False,
    )
    assert r["error"] is None
    ans = r["answer"]
    assert ans["is_circular"] is True
    # Plain subtraction would give |340|,|−340|,|−340| -> huge MAE. Circular -> 20.
    assert ans["metrics"]["mae"] == pytest.approx(20.0, abs=1e-4)
    assert ans["units"] == "degrees"


def test_defaults_to_overlap_window(monkeypatch, fake_backend_desc):
    _install_http(monkeypatch, [11.0], [10.0], "temperature_2m")
    r = accuracy.forecast_accuracy(
        "temperature_2m", "Copenhagen",
        backend_desc=fake_backend_desc, use_network_geocode=False,
    )
    assert r["resolved"]["date_range"] == {"start": "2026-05-07", "end": "2026-05-18"}


def test_unknown_model_is_structured_error(fake_backend_desc):
    r = accuracy.forecast_accuracy(
        "temperature_2m", "Copenhagen", model="nonexistent",
        backend_desc=fake_backend_desc, use_network_geocode=False,
    )
    assert r["answer"] is None
    assert r["error"]["type"] == "ModelUnavailable"


def test_variable_unavailable_offline(monkeypatch, fake_backend_desc):
    # Backend returns all-null for the variable -> actionable error.
    _install_http(monkeypatch, [None, None], [None, None], "wind_speed_10m")
    r = accuracy.forecast_accuracy(
        "wind_speed_10m", "Copenhagen",
        backend_desc=fake_backend_desc, use_network_geocode=False,
    )
    assert r["error"]["type"] == "VariableUnavailable"
