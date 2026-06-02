"""Curated research dataset preparation pipeline.

Generates reusable, precomputed, scientifically meaningful, analysis-ready
forecast verification datasets for specific locations and time ranges.

These are scientific artifacts — not temporary caches or throwaway intermediates.

Usage
-----
# Full preparation (CLI)
weather-analyzer prepare-dataset \\
    --location copenhagen \\
    --years 2 \\
    --variables all \\
    --build-all \\
    --output-dir ./datasets

# From code
from datasets.config import LOCATION_PRESETS
from datasets.prepare import run_pipeline

dataset_dir = run_pipeline(
    location=LOCATION_PRESETS["copenhagen"],
    start_date="2024-01-01",
    end_date="2025-12-31",
    variables=["temperature_2m", "precipitation"],
    models=["icon_eu"],
    build_metrics=True,
    build_events=True,
    output_dir="./datasets",
)
"""

__version__ = "2.0.0"
