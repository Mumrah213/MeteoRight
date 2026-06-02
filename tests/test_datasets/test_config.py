"""Tests for the dataset preparation config module.

Tests:
- Location presets resolve correctly
- Variable sets resolve correctly
- Dataset name generation works as expected
- Default models resolve correctly
"""

from datasets.config import (
    BUILT_IN_EVENTS,
    DEFAULT_MODELS,
    EVENT_DEFINITIONS,
    LOCATION_PRESETS,
    VARIABLE_SETS,
    generate_dataset_name,
)


class TestLocationPresets:
    def test_copenhagen(self):
        loc = LOCATION_PRESETS["copenhagen"]
        assert loc["name"] == "Copenhagen"
        assert loc["lat"] == 55.605
        assert loc["lon"] == 12.574

    def test_stockholm(self):
        loc = LOCATION_PRESETS["stockholm"]
        assert loc["name"] == "Stockholm"
        assert loc["lat"] == 59.329
        assert loc["lon"] == 18.069

    def test_berlin(self):
        loc = LOCATION_PRESETS["berlin"]
        assert loc["name"] == "Berlin"
        assert loc["lat"] == 52.520
        assert loc["lon"] == 13.405

    def test_all_locations_have_timezone(self):
        for name, loc in LOCATION_PRESETS.items():
            assert "timezone" in loc, f"Location '{name}' missing timezone"


class TestVariableSets:
    def test_minimal(self):
        assert VARIABLE_SETS["minimal"] == ("temperature_2m", "precipitation")

    def test_standard(self):
        assert len(VARIABLE_SETS["standard"]) == 4

    def test_all(self):
        assert len(VARIABLE_SETS["all"]) == 9
        assert "temperature_2m" in VARIABLE_SETS["all"]
        assert "precipitation" in VARIABLE_SETS["all"]
        assert "cloud_cover" in VARIABLE_SETS["all"]

    def test_all_superset(self):
        for v in VARIABLE_SETS["minimal"]:
            assert v in VARIABLE_SETS["all"]
        for v in VARIABLE_SETS["standard"]:
            assert v in VARIABLE_SETS["all"]


class TestDefaultModels:
    def test_copenhagen_models(self):
        models = DEFAULT_MODELS["copenhagen"]
        assert "icon_eu" in models
        assert "ecmwf_ifs_hres" in models

    def test_stockholm_models(self):
        models = DEFAULT_MODELS["stockholm"]
        assert "icon_eu" in models


class TestEventDefinitions:
    def test_all_events_defined(self):
        for event in BUILT_IN_EVENTS:
            assert event in EVENT_DEFINITIONS, f"Event '{event}' not defined"

    def test_event_structure(self):
        frost = EVENT_DEFINITIONS["frost"]
        assert "variable" in frost
        assert "threshold" in frost
        assert "operator" in frost
        assert "description" in frost
        assert frost["variable"] == "temperature_2m"
        assert frost["threshold"] == 0.0

    def test_frost_threshold(self):
        assert EVENT_DEFINITIONS["frost"]["threshold"] == 0.0

    def test_heavy_precip_threshold(self):
        assert EVENT_DEFINITIONS["heavy_precip"]["threshold"] == 5.0

    def test_hot_day_threshold(self):
        assert EVENT_DEFINITIONS["hot_day"]["threshold"] == 30.0


class TestGenerateDatasetName:
    def test_same_year(self):
        name = generate_dataset_name("Copenhagen", "2024-01-01", "2024-12-31")
        assert name == "copenhagen_2024"

    def test_multi_year(self):
        name = generate_dataset_name("Copenhagen", "2024-01-01", "2025-12-31")
        assert name == "copenhagen_2y"

    def test_three_years(self):
        name = generate_dataset_name("Stockholm", "2023-01-01", "2025-12-31")
        assert name == "stockholm_3y"

    def test_location_slug(self):
        name = generate_dataset_name("New York City", "2024-01-01", "2024-12-31")
        assert name == "new_york_city_2024"
