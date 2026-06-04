"""Pydantic models for weather forecast and observation records."""


from datetime import datetime

from pydantic import BaseModel, field_validator, model_validator


class ForecastRecord(BaseModel):
    """A single hourly weather forecast record.

    Each row represents one variable value for one target time from one
    forecast run. Multiple runs for the same target time are preserved
    independently (different forecast_issue_time).
    """

    forecast_issue_time: datetime
    forecast_target_time: datetime
    lead_hours: int | None = None
    model: str
    latitude: float
    longitude: float
    elevation: float
    location_id: str | None = None
    location_name: str | None = None
    requested_latitude: float | None = None
    requested_longitude: float | None = None
    location_distance_km: float | None = None

    # Variable fields — dynamically added based on download config
    # Stored as dict so we can handle any variable set
    variables: dict[str, float | None] = {}

    @model_validator(mode="after")
    def compute_and_validate(self):
        """Compute lead_hours and validate forecast constraints."""
        # Compute lead_hours from timestamps if not provided
        if self.lead_hours is None:
            delta = self.forecast_target_time - self.forecast_issue_time
            self.lead_hours = int(delta.total_seconds() // 3600)

        # Validate lead_hours is non-negative
        if self.lead_hours < 0:
            raise ValueError(f"lead_hours must be >= 0, got {self.lead_hours}")

        # Validate target time >= issue time
        if self.forecast_target_time < self.forecast_issue_time:
            raise ValueError(
                f"forecast_target_time ({self.forecast_target_time}) "
                f"must be >= forecast_issue_time ({self.forecast_issue_time})"
            )

        # Validate latitude
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"latitude must be in [-90, 90], got {self.latitude}")

        # Validate longitude
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"longitude must be in [-180, 180], got {self.longitude}")

        return self


class ObservationRecord(BaseModel):
    """A single hourly weather observation record.

    Observations come from the Open-Meteo Archive API (ERA5 reanalysis).
    """

    observation_time: datetime
    latitude: float
    longitude: float
    elevation: float
    location_id: str | None = None
    location_name: str | None = None
    requested_latitude: float | None = None
    requested_longitude: float | None = None
    location_distance_km: float | None = None
    variables: dict[str, float | None] = {}

    @field_validator("latitude")
    @classmethod
    def lat_range(cls, v):
        if not -90 <= v <= 90:
            raise ValueError(f"latitude must be in [-90, 90], got {v}")
        return v

    @field_validator("longitude")
    @classmethod
    def lon_range(cls, v):
        if not -180 <= v <= 180:
            raise ValueError(f"longitude must be in [-180, 180], got {v}")
        return v


def forecast_record_schema() -> dict:
    """Return the expected parquet schema for forecast records as a dict of column types."""
    return {
        "forecast_issue_time": "timestamp[us, tz=UTC]",
        "forecast_target_time": "timestamp[us, tz=UTC]",
        "lead_hours": "int32",
        "model": "string",
        "latitude": "float64",
        "longitude": "float64",
        "elevation": "float64",
        "location_id": "string",
        "location_name": "string",
        "requested_latitude": "float64",
        "requested_longitude": "float64",
        "location_distance_km": "float64",
    }


def observation_record_schema() -> dict:
    """Return the expected parquet schema for observation records as a dict of column types."""
    return {
        "observation_time": "timestamp[us, tz=UTC]",
        "latitude": "float64",
        "longitude": "float64",
        "elevation": "float64",
        "location_id": "string",
        "location_name": "string",
        "requested_latitude": "float64",
        "requested_longitude": "float64",
        "location_distance_km": "float64",
    }
