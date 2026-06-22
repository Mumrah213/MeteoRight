"""Live end-to-end test against a self-hosted Open-Meteo backend.

Marked ``network``; skips automatically when the backend is unreachable, so it
never breaks the default suite. Run explicitly with:

    pytest -m network tests/test_forecast_accuracy_live.py

Requires a backend with temperature_2m forecast + ERA5 archive data (e.g. the
local podman ``open-meteo-api`` container at 127.0.0.1:8080).
"""

import pytest

from agent_tools import describe_backend, forecast_accuracy

BACKEND = "http://127.0.0.1:8080/v1/forecast"

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def backend_desc():
    desc = describe_backend(BACKEND)
    if not desc.get("reachable"):
        pytest.skip(f"Backend not reachable at {BACKEND}")
    return desc


def test_temperature_accuracy_live(backend_desc):
    r = forecast_accuracy(
        "temperature_2m", ["Malmö", "Copenhagen"], model="ecmwf",
        backend_desc=backend_desc,
    )
    assert r["error"] is None, r["error"]
    ans = r["answer"]
    assert ans is not None
    assert ans["sample_size"] > 0
    assert ans["metrics"]["mae"] >= 0.0
    # Sanity: a reanalysis-vs-forecast temperature MAE should be small (< 10°C).
    assert ans["metrics"]["mae"] < 10.0


def test_resolved_metadata_live(backend_desc):
    r = forecast_accuracy(
        "temperature_2m", "Copenhagen", model="ecmwf", backend_desc=backend_desc
    )
    res = r["resolved"]
    assert res["model_backend_name"] == "ecmwf_ifs025"
    assert res["grid_point"]["lat"] is not None
    assert res["date_range"]["start"] <= res["date_range"]["end"]
