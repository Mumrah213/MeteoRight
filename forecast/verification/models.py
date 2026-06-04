"""Phase 3: Data models for forecast verification.

Defines the core data structures used throughout the verification
pipeline: input schemas, alignment records, evaluation results,
and metric outputs.

All models are pure and immutable — they carry provenance (raw_ref)
and explicit null handling.
"""


from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def _month_to_season(month: int) -> str:
    """Convert month number to meteorological season.

    Meteorological seasons (not astronomical):
    - DJF: December, January, February (winter in NH)
    - MAM: March, April, May (spring in NH)
    - JJA: June, July, August (summer in NH)
    - SON: September, October, November (autumn in NH)

    Args:
        month: 1-12

    Returns:
        Season code: "DJF", "MAM", "JJA", "SON"
    """
    if month in (12, 1, 2):
        return "DJF"
    elif month in (3, 4, 5):
        return "MAM"
    elif month in (6, 7, 8):
        return "JJA"
    elif month in (9, 10, 11):
        return "SON"
    raise ValueError(f"Invalid month: {month}")


# ── Verification Input Schema ──────────────────────────────────────────────


class VerificationInput(BaseModel):
    """Input to a verification run.

    Contains all forecasts and observations to be compared.
    Alignment is performed on (lat, lon, valid_time, variable).
    """

    forecasts: list[Any]  # CanonicalForecastRecord or similar
    observations: list[Any]  # CanonicalForecastRecord with is_observation=True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("forecasts", mode="before")
    @classmethod
    def enforce_forecasts_non_null(cls, value: list[Any] | None) -> list[Any]:
        if value is None:
            raise ValueError("forecasts cannot be None")
        if not isinstance(value, list):
            raise ValueError("forecasts must be a list")
        return value

    @field_validator("observations", mode="before")
    @classmethod
    def enforce_observations_non_null(cls, value: list[Any] | None) -> list[Any]:
        if value is None:
            raise ValueError("observations cannot be None")
        if not isinstance(value, list):
            raise ValueError("observations must be a list")
        return value

    @property
    def location_keys(self) -> set[tuple[float, float]]:
        """Unique (latitude, longitude) pairs covered."""
        locations: set[tuple[float, float]] = set()
        for rec in self.forecasts:
            locations.add((rec.latitude, rec.longitude))
        return locations

    @property
    def variable_names(self) -> set[str]:
        """Unique variable names across forecasts."""
        return {rec.variable for rec in self.forecasts}

    @property
    def valid_time_range(self) -> tuple[datetime, datetime] | None:
        """(min_valid_time, max_valid_time) across all records."""
        if not self.forecasts and not self.observations:
            return None
        all_times = [rec.valid_time for rec in self.forecasts + self.observations]
        return (min(all_times), max(all_times))


# ── Alignment Record ───────────────────────────────────────────────────────


class AlignmentRecord(BaseModel):
    """One matched (forecast, observation) pair.

    Key: (latitude, longitude, valid_time, variable)

    Contains both forecast and observation values, or None if one is missing.
    Nulls are explicit — no silent interpolation.

    Temporal metadata:
    - month, season, year are derived from valid_time for temporal grouping
    """

    latitude: float
    longitude: float
    valid_time: datetime
    variable: str
    lead_hours: int
    # Temporal metadata (derived from valid_time)
    month: int = 0
    season: str = ""  # "DJF", "MAM", "JJA", "SON"
    year: int = 0

    # Forecast value (may be None if forecast is missing)
    forecast_value: float | None = None
    forecast_provider: str | None = None
    forecast_model: str | None = None
    forecast_run_time: datetime | None = None

    # Observation value (may be None if observation is missing)
    observation_value: float | None = None
    observation_model: str | None = None

    # Match status
    match_status: str = Field(  # "matched", "forecast_only", "observation_only"
        description="How well forecast and observation aligned"
    )

    @model_validator(mode="after")
    def derive_temporal_metadata(self) -> AlignmentRecord:
        """Derive month, season, year from valid_time."""
        self.month = self.valid_time.month
        self.year = self.valid_time.year
        self.season = _month_to_season(self.month)
        return self

    @property
    def is_fully_matched(self) -> bool:
        """Both forecast and observation present."""
        return (
            self.match_status == "matched"
            and self.forecast_value is not None
            and self.observation_value is not None
        )

    @property
    def key(self) -> tuple[float, float, datetime, str]:
        """Unique key for alignment."""
        return (self.latitude, self.longitude, self.valid_time, self.variable)

    def __hash__(self) -> int:
        return hash(self.key)


# ── Evaluation Result ──────────────────────────────────────────────────────


class ForecastEvaluation(BaseModel):
    """One metric for one grouping.

    Example:
      MAE(lead_time=72h, variable="temperature_2m", location=(55.6, 12.6))

    All metrics support grouping by lead_time, variable, location,
    run_time, and season.

    Coverage awareness:
    - sample_size: number of forecast-observation pairs contributing to metric
    - total_count: total records in this group (including null/missing)
    - coverage_fraction: sample_size / total_count (0..1)
    - is_stable: True if sample_size >= min_sample_size threshold
    - variable_type: semantic type ("temperature", "precipitation", etc.)
    """

    variable: str
    lead_time: int | None = None  # None means aggregated across all leads
    metric_type: str  # "mae", "rmse", "bias", "error_variance"
    value: float | None
    sample_size: int  # matched pairs contributing to this metric
    total_count: int = 0  # total records in group (incl. null/missing)
    coverage_fraction: float | None = None  # sample_size / total_count
    is_stable: bool = True  # True if sample_size >= min_sample_size
    min_sample_size: int = 1  # threshold for stability flag
    variable_type: str | None = None  # "temperature", "precipitation", "humidity", etc.
    location: tuple[float, float] | None = None
    time_range: tuple[datetime, datetime] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_coverage_and_stability(self) -> ForecastEvaluation:
        """Post-hoc: compute coverage_fraction and is_stability flags."""
        # Compute coverage_fraction
        if self.total_count > 0:
            self.coverage_fraction = self.sample_size / self.total_count
        elif self.sample_size == 0:
            self.coverage_fraction = 0.0
        else:
            self.coverage_fraction = 1.0

        # Compute stability flag
        if self.min_sample_size > 0:
            self.is_stable = self.sample_size >= self.min_sample_size
        else:
            self.is_stable = True

        return self

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        for field_name in ("time_range",):
            val = data.get(field_name)
            if isinstance(val, tuple) and len(val) == 2:
                if isinstance(val[0], datetime):
                    data[field_name] = [val[0].isoformat(), val[1].isoformat()]
        return data


# ── Degradation Profile ────────────────────────────────────────────────────


class DegradationPoint(BaseModel):
    """One point on the error vs lead_time curve.

    Coverage aware:
    - sample_size: matched pairs contributing to this metric
    - total_count: total records at this lead_time (incl. nulls)
    - coverage_fraction: sample_size / total_count
    - is_stable: True if sample_size >= min_sample_size
    """

    lead_hours: int
    metric_type: str
    variable: str
    value: float
    sample_size: int
    total_count: int = 0  # total records at this lead time (incl. nulls)
    coverage_fraction: float | None = None
    is_monotonic_with_prev: bool = False  # True if error >= prev lead's error
    is_stable: bool = True  # True if sample_size >= min_sample_size
    min_sample_size: int = 1

    @model_validator(mode="after")
    def compute_coverage_and_stability(self) -> DegradationPoint:
        """Post-hoc: compute coverage_fraction and is_stable flags."""
        if self.total_count > 0:
            self.coverage_fraction = self.sample_size / self.total_count
        elif self.sample_size == 0:
            self.coverage_fraction = 0.0
        else:
            self.coverage_fraction = 1.0

        self.is_stable = self.sample_size >= self.min_sample_size
        return self


class DegradationProfile(BaseModel):
    """Error growth analysis for one variable at one location."""

    variable: str
    location: tuple[float, float]
    metric_type: str  # "mae", "rmse", "bias"
    points: list[DegradationPoint] = Field(default_factory=list)
    is_monotonic: bool = False  # True if error never decreases
    instability_points: list[int] = Field(  # lead_hours where error decreases
        default_factory=list
    )
    summary: dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        return data


# ── EvaluationResult (orchestrator output) ─────────────────────────────────


class EvaluationResult(BaseModel):
    """Complete output from a verification run."""

    input_metadata: dict[str, Any] = Field(default_factory=dict)
    alignments: list[AlignmentRecord] = Field(default_factory=list)
    matched_count: int = 0
    forecast_only_count: int = 0
    observation_only_count: int = 0
    evaluations: list[ForecastEvaluation] = Field(default_factory=list)
    degradation_profiles: list[DegradationProfile] = Field(default_factory=list)

    @property
    def total_alignment_records(self) -> int:
        return len(self.alignments)

    def summary(self) -> dict[str, Any]:
        """Human-readable summary of the verification run."""
        return {
            "total_alignments": self.total_alignment_records,
            "matched": self.matched_count,
            "forecast_only": self.forecast_only_count,
            "observation_only": self.observation_only_count,
            "evaluations": len(self.evaluations),
            "degradation_profiles": len(self.degradation_profiles),
        }
