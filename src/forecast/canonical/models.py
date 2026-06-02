"""Phase 2: Canonical forecasting dataset models.

This module defines the two core data models for the canonical
forecasting dataset:

1. ForecastRun — normalized per-request structure (intermediate)
2. CanonicalForecastRecord — one record per (time, variable) (final)

Both are pure, provider-agnostic, and deterministic. Every value in
a forecast becomes exactly one CanonicalForecastRecord.

Coordinate system:
  (provider, model, run_time, valid_time, variable) → value
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ── ForecastRun ────────────────────────────────────────────────────────────


class ForecastRun(BaseModel):
    """Normalized representation of a single forecast request.

    Contains all data from one provider response, stripped of provider
    quirks but preserving all raw values including nulls.

    A single run has exactly one (latitude, longitude) pair at this
    stage — multi-location responses are split into separate runs
    during flattening.
    """

    provider: str
    model: str | None
    run_time: datetime
    latitude: float
    longitude: float

    # All timestamps are ISO-8601 strings in UTC (as returned by provider)
    times: list[str]

    # Variable name → list of values (same length as times)
    variables: dict[str, list[float | None]]

    # Units per variable (optional; some providers omit them)
    units: dict[str, str] = Field(default_factory=dict)

    # Reference to persisted raw JSON (file path or URI)
    raw_ref: str | None = None

    # Arbitrary metadata carried through the pipeline
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_time", mode="before")
    @classmethod
    def enforce_utc(cls, value: datetime | str) -> datetime:
        """Ensure run_time is UTC-naive (conceptual UTC).

        If the input has a timezone, convert to UTC then drop tzinfo.
        """
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is not None:
                dt = dt.astimezone(UTC).replace(tzinfo=None)
            return dt
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value

    @property
    def time_count(self) -> int:
        return len(self.times)

    @property
    def variable_names(self) -> list[str]:
        return list(self.variables.keys())

    @property
    def is_valid(self) -> bool:
        """Basic sanity: have times and at least one variable."""
        return len(self.times) > 0 and len(self.variables) > 0

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        # Serialize run_time as ISO string
        if isinstance(data.get("run_time"), datetime):
            data["run_time"] = data["run_time"].isoformat()
        return data


# ── CanonicalForecastRecord ────────────────────────────────────────────────


class CanonicalForecastRecord(BaseModel):
    """One record per (variable, time step).

    Every forecast value expands into exactly one record. This creates
    a flat, queryable table where each row is uniquely identified by:
      (provider, model, run_time, valid_time, variable)

    lead_hours is derived from valid_time - run_time and must be
    a non-negative integer.
    """

    provider: str
    model: str | None
    run_time: datetime
    valid_time: datetime
    lead_hours: int
    latitude: float
    longitude: float
    variable: str
    value: float | None
    unit: str | None = None
    is_observation: bool = False
    is_reanalysis: bool = False
    raw_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("lead_hours", mode="before")
    @classmethod
    def must_be_integer_lead_hours(cls, value: int | float | None) -> int:
        """Lead hours must be a non-negative integer."""
        if value is None:
            raise ValueError("lead_hours cannot be None")
        if not isinstance(value, int):
            raise ValueError(f"lead_hours must be integer, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"lead_hours must be non-negative, got {value}")
        return value

    @field_validator("valid_time", mode="before")
    @classmethod
    def enforce_utc_valid_time(cls, value: datetime | str) -> datetime:
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is not None:
                dt = dt.astimezone(UTC).replace(tzinfo=None)
            return dt
        if value.tzinfo is not None:
            return value.astimezone(UTC).replace(tzinfo=None)
        return value

    @property
    def key(self) -> tuple[str, str, datetime, datetime, str, float, float]:
        """Global unique key: (provider, model, run_time, valid_time,
        variable, latitude, longitude).

        This is the ONLY valid identifier — it must be globally unique
        across all runs and locations.
        """
        return (
            self.provider,
            self.model or "",
            self.run_time,
            self.valid_time,
            self.variable,
            self.latitude,
            self.longitude,
        )

    def __hash__(self) -> int:
        """Hash based on the global key for set/dict lookups."""
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        """Equality based on the global key."""
        if not isinstance(other, CanonicalForecastRecord):
            return NotImplemented
        return self.key == other.key

    def __lt__(self, other: CanonicalForecastRecord) -> bool:
        """Comparison for sorting: by run_time, then valid_time, then variable."""
        return (self.run_time, self.valid_time, self.variable) < (
            other.run_time,
            other.valid_time,
            other.variable,
        )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        # Serialize datetimes as ISO strings
        for field_name in ("run_time", "valid_time"):
            val = data.get(field_name)
            if isinstance(val, datetime):
                data[field_name] = val.isoformat()
        return data
