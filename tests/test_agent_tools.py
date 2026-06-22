"""Unit tests for the composable agent tools (no network)."""

import httpx
import pytest

from agent_tools import areas, catalog, geocoding, models

# --------------------------------------------------------------------------
# geocoding
# --------------------------------------------------------------------------


def test_geocode_preset_no_network():
    # Diacritic-insensitive: "Malmö" resolves via the "malmo" preset.
    r = geocoding.geocode_location("Malmö", use_network=False)
    assert r["error"] is None
    assert r["source"] == "preset"
    assert r["lat"] == 55.605 and r["lon"] == 13.003
    assert r["name"] == "Malmö"


def test_geocode_copenhagen_preset():
    r = geocoding.geocode_location("copenhagen", use_network=False)
    assert r["source"] == "preset"
    assert r["lat"] == 55.605 and r["lon"] == 12.574


def test_geocode_miss_offline_is_structured_error():
    r = geocoding.geocode_location("Nowhereville", use_network=False)
    assert r["lat"] is None
    assert r["error"]["type"] == "LocationNotFound"


def test_geocode_network_fallback(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("name") == "Reykjavik"
        return httpx.Response(
            200,
            json={"results": [{"name": "Reykjavík", "latitude": 64.15,
                               "longitude": -21.95, "country": "Iceland",
                               "timezone": "Atlantic/Reykjavik"}]},
        )

    monkeypatch.setattr(
        geocoding, "_make_client", lambda timeout: httpx.Client(transport=httpx.MockTransport(handler))
    )
    r = geocoding.geocode_location("Reykjavik")
    assert r["source"] == "open-meteo-geocoding"
    assert r["lat"] == 64.15 and r["country"] == "Iceland"


# --------------------------------------------------------------------------
# areas
# --------------------------------------------------------------------------


def test_resolve_area_two_cities_nearest_point():
    r = areas.resolve_area("Malmö-Copenhagen", use_network=False)
    assert r["error"] is None
    assert len(r["places"]) == 2
    # Midpoint sits between Malmö (13.003) and Copenhagen (12.574).
    assert 12.5 < r["center"]["lon"] < 13.1
    gp = r["grid_point"]
    assert gp is not None
    # Snapped to a 0.25° grid cell near the center.
    assert abs(gp["lat"] - 55.6) < 0.4 and abs(gp["lon"] - 12.8) < 0.4


def test_resolve_area_accepts_list():
    r = areas.resolve_area(["Copenhagen", "Malmö"], use_network=False)
    assert r["error"] is None
    assert r["grid_point"] is not None


def test_resolve_area_all_unresolved():
    r = areas.resolve_area("Atlantis, El Dorado", use_network=False)
    assert r["grid_point"] is None
    assert r["error"]["type"] == "LocationNotFound"


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ecmwf", "ecmwf_ifs025"),
        ("ECMWF_IFS_HRES", "ecmwf_ifs025"),
        ("icon_eu", "dwd_icon_eu"),
        ("harmonie", "dmi_harmonie_arome_europe"),
        ("ecmwf_ifs025", "ecmwf_ifs025"),  # domain passes through
        ("some_unknown_model", "some_unknown_model"),  # unknown passes through
    ],
)
def test_resolve_model_alias(name, expected):
    assert models.resolve_model_alias(name) == expected


def test_list_models_uses_probe_result():
    fake_desc = {
        "reachable": True,
        "models": [
            {"backend_name": "dwd_icon_eu", "coverage": {"start": "a", "end": "b"}},
            {"backend_name": "ecmwf_ifs025", "coverage": None},
        ],
        "error": None,
    }
    r = models.list_models(backend_desc=fake_desc)
    by_name = {m["backend_name"]: m for m in r["models"]}
    assert "icon_eu" in by_name["dwd_icon_eu"]["aliases"]
    assert "ecmwf" in by_name["ecmwf_ifs025"]["aliases"]
    assert by_name["dwd_icon_eu"]["coverage"] == {"start": "a", "end": "b"}


# --------------------------------------------------------------------------
# catalog
# --------------------------------------------------------------------------


def test_list_variables():
    r = catalog.list_variables()
    assert "temperature_2m" in r["all"]
    assert "wind_direction_10m" in r["circular"]
    assert r["units"]["wind_direction_10m"] == "degrees"
    assert "minimal" in r["sets"]
