"""Unified configuration models for the weather pipeline.

Combines v2's data architecture config (locations, variables, events)
with docker's ingestion config (HTTP transport, providers, validation)
into a single type-safe schema using Pydantic.

Design philosophy:
- v2 = data architecture + storage truth
- docker = forecast ingestion engine
- unified = configuration layer that works with both

Usage
-----
>>> from src.config.models import PipelineConfig, HTTPConfig, ProviderConfig

>>> config = PipelineConfig()
>>> print(config.output_dir)
data/
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ===========================================================================
# Transport / HTTP configuration (from docker)
# ===========================================================================


class HTTPConfig(BaseModel, frozen=True):
    """HTTP transport configuration shared by all providers.

    Defaults match Open-Meteo API conventions.
    """

    base_url: str = "https://api.open-meteo.com"
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    user_agent: str = "meteoright/1.0.0"


# ===========================================================================
# Provider configuration (from docker)
# ===========================================================================


class ProviderConfig(BaseModel, frozen=True):
    """Configuration for a single forecast provider."""

    name: str = "openmeteo"
    base_url: str = Field(default="https://api.open-meteo.com", description="Provider API base URL")
    raw_dir: str = Field(default="data/raw", description="Directory for raw API responses")
    http: HTTPConfig = Field(default_factory=HTTPConfig)


# ===========================================================================
# Storage configuration (from v2)
# ===========================================================================


class StorageConfig(BaseModel, frozen=True):
    """Parquet storage layout from v2."""

    output_dir: str = Field(default="data", description="Root data directory")
    observations_subdir: str = Field(default="raw/observations")
    forecasts_subdir: str = Field(default="raw/forecasts")
    intermediate_subdir: str = Field(default="intermediate")
    cache_subdir: str = Field(default="cache")
    parquet_compression: str = Field(default="snappy")


# ===========================================================================
# Forecast / ingestion configuration (from docker)
# ===========================================================================


class ForecastConfig(BaseModel, frozen=True):
    """Forecast ingestion configuration from docker."""

    models: list[str] = Field(
        default=["icon_eu", "gfs", "ecmwf_ifs_hres"],
        description="Default forecast models to ingest",
    )
    providers: dict[str, ProviderConfig] = Field(
        default_factory=lambda: {"openmeteo": ProviderConfig()},
        description="Provider configurations keyed by name",
    )
    raw_dir: str = Field(default="data/raw", description="Directory for raw API responses")


# ===========================================================================
# Historical / verification configuration (from v2)
# ===========================================================================


class VerificationConfig(BaseModel, frozen=True):
    """Verification pipeline configuration from v2."""

    min_sample_size: int = Field(default=100, description="Minimum sample size per group")
    max_bias: float = Field(default=5.0, description="Maximum acceptable bias (°C)")
    max_mae: float = Field(default=3.0, description="Maximum acceptable MAE (°C)")


# ===========================================================================
# Analysis configuration (from v2)
# ===========================================================================


class AnalysisConfig(BaseModel, frozen=True):
    """Analysis pipeline configuration from v2."""

    anomaly_std_threshold: float = Field(
        default=3.0, description="Standard deviations for anomaly detection"
    )
    extreme_percentile: float = Field(default=0.95, description="Percentile for extreme detection")


# ===========================================================================
# Location presets (from v2)
# ===========================================================================


class LocationConfig(BaseModel, frozen=True):
    """A single location preset from v2."""

    name: str
    lat: float
    lon: float
    timezone: str = "UTC"


# ===========================================================================
# Top-level unified configuration
# ===========================================================================


class PipelineConfig(BaseModel):
    """Unified pipeline configuration.

    Combines v2's data architecture config with docker's ingestion config.

    Usage:
        config = PipelineConfig()
        print(config.locations)           # v2 location presets
        print(config.http.base_url)       # docker HTTP settings
        print(config.storage.output_dir)  # v2 storage layout
    """

    # v2 locations
    locations: dict[str, LocationConfig] = Field(
        default={
            "copenhagen": LocationConfig(
                name="Copenhagen",
                lat=55.605,
                lon=12.574,
                timezone="Europe/Copenhagen",
            ),
            "stockholm": LocationConfig(
                name="Stockholm",
                lat=59.329,
                lon=18.069,
                timezone="Europe/Stockholm",
            ),
            "berlin": LocationConfig(
                name="Berlin",
                lat=52.520,
                lon=13.405,
                timezone="Europe/Berlin",
            ),
        }
    )

    # v2 variable sets
    variable_sets: dict[str, tuple[str, ...]] = Field(
        default={
            "minimal": ("temperature_2m", "precipitation"),
            "standard": (
                "temperature_2m",
                "precipitation",
                "wind_speed_10m",
                "pressure_msl",
            ),
            "all": (
                "temperature_2m",
                "dew_point_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "pressure_msl",
                "relative_humidity_2m",
                "cloud_cover",
            ),
        }
    )

    # v2 default models per location
    default_models: dict[str, tuple[str, ...]] = Field(
        default={
            "copenhagen": ("icon_eu", "ecmwf_ifs_hres"),
            "stockholm": ("icon_eu", "ecmwf_ifs_hres"),
            "berlin": ("icon_eu", "ecmwf_ifs_hres"),
        }
    )

    # docker HTTP transport
    http: HTTPConfig = Field(default_factory=HTTPConfig)

    # docker forecast ingestion
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)

    # v2 storage layout
    storage: StorageConfig = Field(default_factory=StorageConfig)

    # v2 verification
    verification: VerificationConfig = Field(default_factory=VerificationConfig)

    # v2 analysis
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    # location preset helper
    def get_location(self, key: str) -> LocationConfig:
        """Get a location preset by key."""
        return self.locations[key]

    def get_variables(self, preset: str) -> tuple[str, ...]:
        """Get variable set by preset name."""
        return self.variable_sets[preset]

    def get_models(self, location_key: str) -> tuple[str, ...]:
        """Get default models for a location."""
        return self.default_models[location_key]
