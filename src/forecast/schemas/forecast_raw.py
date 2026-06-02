"""Raw forecast response schemas.

These models capture the structure of raw API responses without
altering or normalizing the data. The goal is fidelity — the
schema mirrors what the provider returns.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

from pydantic import BaseModel, Field


class RawLocationData(BaseModel):
    """Response for a single location."""

    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None
    generationtime_ms: float | None = None
    hourly: dict[str, Any] | None = None
    hourly_units: dict[str, str] | None = None
    daily: dict[str, Any] | None = None
    daily_units: dict[str, str] | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        # Remove empty containers for cleaner serialization
        return {k: v for k, v in data.items() if v is not None}


class RawForecastResponse(BaseModel):
    """Complete raw response from a forecast API call.

    Wraps the response with metadata about the request that produced it.
    This enables reproducibility and debugging.
    """

    locations: list[RawLocationData] = Field(default_factory=list)

    # Request metadata for reproducibility
    request_params: dict[str, Any] = Field(default_factory=dict)
    request_timestamp: str | None = None  # ISO format
    provider: str = "unknown"
    endpoint: str = "unknown"
    raw_json: dict[str, Any] | None = None

    # Null classification (set by provider-specific validators)
    null_classification: str | None = None

    @property
    def is_valid(self) -> bool:
        """Quick check: does this response have data?"""
        return any(loc.hourly or loc.daily for loc in self.locations)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        # Ensure request_timestamp is ISO format
        if self.request_timestamp is None:
            from datetime import datetime

            self.request_timestamp = datetime.now(UTC).isoformat()
        return data


class RawVariableData(BaseModel):
    """Structured data for a single variable in a response.

    Separates timestamps from values for easier downstream processing.
    """

    name: str
    unit: str | None = None
    timestamps: list[str] | None = None
    values: list[float | None] | None = None

    @property
    def length(self) -> int:
        return len(self.values) if self.values else 0

    @property
    def is_all_null(self) -> bool:
        if not self.values:
            return True
        return all(v is None for v in self.values)

    @property
    def is_empty(self) -> bool:
        return len(self.values or []) == 0

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        return {k: v for k, v in data.items() if v is not None}
