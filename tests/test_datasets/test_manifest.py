"""Tests for the dataset manifest module — essential cases only."""

import tempfile
from pathlib import Path

from datasets.manifest import (
    build_manifest,
    read_manifest,
    update_manifest_statistics,
    write_manifest,
)


class TestManifest:
    def test_basic_manifest(self):
        manifest = build_manifest(
            dataset_name="copenhagen_2y",
            location={"name": "Copenhagen", "lat": 55.605, "lon": 12.574},
            start_date="2024-01-01",
            end_date="2025-12-31",
            variables=["temperature_2m", "precipitation"],
            models=["icon_eu"],
        )

        assert manifest["dataset_name"] == "copenhagen_2y"
        assert manifest["version"] == "1.0.0"
        assert manifest["location"]["name"] == "Copenhagen"
        assert manifest["location"]["latitude"] == 55.605
        assert manifest["time_range"]["start_date"] == "2024-01-01"
        assert manifest["variables"] == ["precipitation", "temperature_2m"]
        assert manifest["models"] == ["icon_eu"]

    def test_write_and_read(self):
        manifest = build_manifest(
            dataset_name="test",
            location={"name": "Test", "lat": 55.6, "lon": 12.5},
            start_date="2024-01-01",
            end_date="2025-12-31",
            variables=["temp"],
            models=["model"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "manifests").mkdir(parents=True)
            path = tmp / "manifests" / "manifest.json"
            write_manifest(manifest, path)
            assert path.exists()

            read_back = read_manifest(tmp)
            assert read_back is not None
            assert read_back["dataset_name"] == "test"
            assert read_back["variables"] == ["temp"]

    def test_update_statistics(self):
        manifest = build_manifest(
            dataset_name="test",
            location={"name": "Test", "lat": 55.6, "lon": 12.5},
            start_date="2024-01-01",
            end_date="2025-12-31",
            variables=["temp"],
            models=["model"],
        )

        manifest = update_manifest_statistics(
            manifest,
            forecast_rows=1000,
            observation_rows=500,
            verification_rows=800,
            event_datasets=["frost", "heavy_precip"],
            metric_groups=["lead_hours", "season"],
        )

        assert manifest["statistics"]["forecast_rows"] == 1000
        assert manifest["statistics"]["observation_rows"] == 500
        assert manifest["statistics"]["verification_rows"] == 800
        assert sorted(manifest["statistics"]["event_datasets"]) == ["frost", "heavy_precip"]
