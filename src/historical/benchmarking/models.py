"""Phase 4: Data models for multi-model benchmarking.

Defines the core data structures for:
- Model capability tracking
- Benchmark results and comparisons
- Skill scores and rankings
- Regional and seasonal benchmarking
- Longitudinal performance tracking
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ═══════════════════════════════════════════════════════════════════════════
# Model Capabilities
# ═══════════════════════════════════════════════════════════════════════════


class ModelCapability(BaseModel):
    """Declared capabilities for one forecast model.

    These are discovered empirically and updated as new data arrives.
    No model is assumed to have full coverage of all variables.
    """

    model_name: str
    provider: str  # e.g., "openmeteo", "ecmwf_direct"
    version: str | None = None  # e.g., "ifs_12r1"

    # Variables this model supports (discovered from API)
    supported_variables: list[str] = Field(default_factory=list)

    # Forecast horizon in hours (max lead time)
    max_lead_hours: int | None = None

    # Update frequency in hours (e.g., 6 for 4x/day)
    update_frequency_hours: int | None = None

    # Spatial resolution in degrees (approximate)
    spatial_resolution_deg: float | None = None

    # Temporal resolution
    supports_hourly: bool = True
    supports_daily: bool = False

    # Null behavior: how does this model handle missing data?
    # "null", "zero", "interpolate" (future)
    null_behavior: str = "null"

    # Known unsupported variables (for explicit exclusion)
    unsupported_variables: list[str] = Field(default_factory=list)

    # Notes about the model's characteristics
    notes: dict[str, Any] = Field(default_factory=dict)

    # Discovery timestamp
    discovered_at: datetime | None = None

    @field_validator("supported_variables", mode="before")
    @classmethod
    def enforce_supported_not_none(cls, value: list[str] | None) -> list[str]:
        if value is None:
            return []
        return list(set(value))  # deduplicate

    def supports_variable(self, variable: str) -> bool:
        """Check if this model supports a given variable."""
        return variable in self.supported_variables

    def intersects_with(self, other: ModelCapability) -> list[str]:
        """Return variables supported by both models."""
        return list(set(self.supported_variables) & set(other.supported_variables))


class CapabilityMatrix(BaseModel):
    """Dynamic capability tracking for all forecast models.

    Tracks which models support which variables, horizons, etc.
    Used to determine fair comparison subsets.
    """

    models: dict[str, ModelCapability] = Field(default_factory=dict)

    def register(self, capability: ModelCapability) -> None:
        """Register or update capabilities for a model."""
        self.models[capability.model_name] = capability

    def get_common_variables(self, model_names: list[str]) -> list[str]:
        """Return variables supported by ALL listed models.

        CRITICAL: Only variables in the intersection are comparable.
        """
        if not model_names:
            return []
        common = set(self.models[model_names[0]].supported_variables)
        for name in model_names[1:]:
            common &= set(self.models[name].supported_variables)
        return sorted(common)

    def get_max_horizon(self, model_name: str) -> int | None:
        """Return the maximum forecast horizon for a model."""
        model = self.models.get(model_name)
        return model.max_lead_hours if model else None

    def get_intersecting_variables(
        self,
        model_a: str,
        model_b: str,
    ) -> list[str]:
        """Return variables supported by both models."""
        cap_a = self.models.get(model_a)
        cap_b = self.models.get(model_b)
        if cap_a and cap_b:
            return sorted(set(cap_a.supported_variables) & set(cap_b.supported_variables))
        return []

    def get_supported_models(self, variable: str) -> list[str]:
        """Return models that support a given variable."""
        return [name for name, cap in self.models.items() if variable in cap.supported_variables]


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark Results
# ═══════════════════════════════════════════════════════════════════════════


class ModelSkillScore(BaseModel):
    """One metric score for one model.

    Used for ranking and comparison.
    Always includes sample_size and coverage_fraction for fairness.
    """

    model_name: str
    variable: str
    lead_time: int | None = None  # None = aggregated
    metric_type: str  # "mae", "rmse", "bias", "error_variance"
    value: float | None  # Metric value (lower MAE/RMSE is better)
    sample_size: int  # matched pairs contributing
    total_count: int  # total records in this group
    coverage_fraction: float | None = None  # sample_size / total_count
    is_stable: bool = True  # sample_size >= min_sample_size
    min_sample_size: int = 1
    rank: int | None = None  # rank among models (1 = best for MAE/RMSE)

    @model_validator(mode="after")
    def compute_coverage(self) -> ModelSkillScore:
        if self.total_count > 0:
            self.coverage_fraction = self.sample_size / self.total_count
        elif self.sample_size == 0:
            self.coverage_fraction = 0.0
        else:
            self.coverage_fraction = 1.0
        # Stability: True if sample_size >= min_sample_size
        self.is_stable = self.sample_size >= self.min_sample_size
        return self


class BenchmarkResult(BaseModel):
    """Complete output from a cross-model benchmark.

    Answers: "Which model performs best for variable X at lead_time Y?"
    """

    # Comparison parameters
    models_compared: list[str]  # model names compared
    variable: str  # variable evaluated
    lead_time: int | None = None  # specific lead time or None (all)
    metric: str  # primary metric used for ranking
    evaluation_window: tuple[datetime, datetime] | None = None  # time range

    # Results
    best_model: str | None = None  # model with lowest MAE/RMSE
    rankings: list[ModelSkillScore] = Field(default_factory=list)

    # Coverage statistics across models
    coverage_statistics: dict[str, float] = Field(default_factory=dict)
    sample_sizes: dict[str, int] = Field(default_factory=dict)

    # Fairness warnings
    fairness_warnings: list[str] = Field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        if self.evaluation_window and isinstance(self.evaluation_window, tuple):
            data["evaluation_window"] = [
                self.evaluation_window[0].isoformat(),
                self.evaluation_window[1].isoformat(),
            ]
        return data


class BenchmarkComparison(BaseModel):
    """Direct comparison between two models for one variable.

    Answers: "How do Model A and Model B compare for variable X?"
    """

    model_a: str
    model_b: str
    variable: str
    lead_time: int | None = None
    metric: str

    # Metrics for each model
    model_a_score: float | None = None
    model_b_score: float | None = None
    model_a_sample_size: int = 0
    model_b_sample_size: int = 0
    model_a_coverage: float | None = None
    model_b_coverage: float | None = None

    # Comparison result
    better_model: str | None = None  # "model_a", "model_b", or None (tied)
    difference: float | None = None  # model_a_score - model_b_score
    model_b_improvement: bool = False  # True if model_b has lower MAE/RMSE

    # Fairness warnings
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def compute_coverage_fields(self) -> BenchmarkComparison:
        if self.model_a_sample_size > 0:
            pass  # coverage computed from context
        if self.model_b_sample_size > 0:
            pass
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Regional Benchmarking
# ═══════════════════════════════════════════════════════════════════════════


class RegionalBenchmarkResult(BaseModel):
    """Benchmark results grouped by geographic region."""

    variable: str
    metric: str
    regions: dict[str, list[ModelSkillScore]] = Field(default_factory=dict)
    coverage_by_region: dict[str, float] = Field(default_factory=dict)
    fairness_warnings: list[str] = Field(default_factory=list)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        return data


# ═══════════════════════════════════════════════════════════════════════════
# Seasonal Benchmarking
# ═══════════════════════════════════════════════════════════════════════════


class SeasonalSkillProfile(BaseModel):
    """Model skill profile across meteorological seasons."""

    model_name: str
    variable: str
    seasons: dict[str, ModelSkillScore] = Field(default_factory=dict)
    # DJF, MAM, JJA, SON

    @property
    def best_season(self) -> str | None:
        """Return the season with best (lowest) MAE/RMSE."""
        if not self.seasons:
            return None
        best_season = None
        best_value = None
        for season, score in self.seasons.items():
            val = score.value
            if val is not None and (best_value is None or val < best_value):
                best_value = val
                best_season = season
        return best_season


# ═══════════════════════════════════════════════════════════════════════════
# Longitudinal Tracking
# ═══════════════════════════════════════════════════════════════════════════


class LongitudinalSkillRecord(BaseModel):
    """Model skill at one point in time.

    Used for tracking how model performance changes over months/seasons.
    """

    model_name: str
    variable: str
    metric_type: str
    value: float | None
    timestamp: datetime  # When this benchmark was computed
    sample_size: int = 0
    coverage_fraction: float | None = None
    evaluation_window: tuple[datetime, datetime] | None = None

    @model_validator(mode="after")
    def compute_coverage(self) -> LongitudinalSkillRecord:
        if self.sample_size > 0 and self.evaluation_window:
            # Coverage is implicit from sample_size / total_count
            # which would come from the benchmark data
            pass
        return self


class LongitudinalTrend(BaseModel):
    """Trend analysis across longitudinal records.

    Uses simple linear regression to detect trends in model skill over time.
    """

    model_name: str
    variable: str
    metric_type: str
    records: list[LongitudinalSkillRecord] = Field(default_factory=list)

    # Trend direction
    improving: bool = False  # metric values decreasing over time (for MAE/RMSE)
    stable: bool = True
    declining: bool = False

    # Regression analysis
    slope: float = 0.0  # Rate of change per day
    r_squared: float = 0.0  # Goodness of fit

    # Statistical summary
    mean_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    trend_description: str = ""
