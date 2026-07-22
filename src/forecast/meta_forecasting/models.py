"""Data models for adaptive meta-forecasting.

Defines core structures for:
- Historical skill profiles (per model, variable, lead time, season, region)
- Model weights (dynamic, explainable, deterministic)
- Blended forecasts (with provenance, confidence, and attribution)
- Confidence scoring (with contributing factors)
- Ensemble agreement metrics
- Regime-aware selection

All models are pure, deterministic, and fully explainable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ── Historical Skill Profile ───────────────────────────────────────────────


class SkillProfileEntry(BaseModel):
    """One point in a model's historical skill profile.

    Captures performance of one model on one variable, at one lead time,
    in one season (optionally one region), averaged over a time window.

    This is NOT a single observation — it's an aggregated statistic from
    a specific evaluation window, so it can be used for future weighting
    without lookahead bias.
    """

    model_name: str
    variable: str
    lead_hours: int | None = None  # None = aggregated across all leads
    season: str = ""  # "DJF", "MAM", "JJA", "SON", "" = all seasons
    region: str = ""  # optional region label, "" = global
    metric_type: str  # "mae", "rmse", "bias"
    value: float | None  # Lower MAE/RMSE is better
    sample_size: int = 0
    coverage_fraction: float | None = None
    is_stable: bool = True
    min_sample_size: int = 1
    # When this profile was last computed
    updated_at: datetime | None = None
    # Evaluation window that produced this profile (for provenance)
    evaluation_window: tuple[datetime, datetime] | None = None

    @field_validator("season", mode="before")
    @classmethod
    def enforce_valid_season(cls, value: str | None) -> str:
        if value is None:
            return ""
        valid = ("DJF", "MAM", "JJA", "SON", "")
        if value not in valid:
            return ""  # silently normalize unknown seasons to ""
        return value

    @field_validator("metric_type", mode="before")
    @classmethod
    def enforce_valid_metric(cls, value: str | None) -> str:
        if value is None:
            return ""
        return value.lower()

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        if self.evaluation_window and isinstance(self.evaluation_window, tuple):
            data["evaluation_window"] = [
                self.evaluation_window[0].isoformat(),
                self.evaluation_window[1].isoformat(),
            ]
        return data


class SkillProfile(BaseModel):
    """Complete historical skill profile for one model.

    Organized by: variable → (lead_time, season, region) → metric → value.
    Supports temporal updates (new data extends the profile).
    """

    model_name: str
    # Key: (variable, lead_hours, season, region) → SkillProfileEntry
    entries: dict[tuple[str, int | None, str, str], SkillProfileEntry] = Field(default_factory=dict)

    def add_entry(
        self,
        variable: str,
        lead_hours: int | None,
        season: str,
        region: str,
        entry: SkillProfileEntry,
    ) -> None:
        """Add or update a skill profile entry.

        The key combines (variable, lead_time, season, region).
        If an entry with the same key exists, it is replaced.
        """
        key = (variable, lead_hours, season, region)
        self.entries[key] = entry

    def get_entry(
        self,
        variable: str,
        lead_hours: int | None,
        season: str = "",
        region: str = "",
    ) -> SkillProfileEntry | None:
        """Look up a specific profile entry by key dimensions."""
        return self.entries.get((variable, lead_hours, season, region))

    def get_entries_for_variable(self, variable: str) -> list[SkillProfileEntry]:
        """Return all entries for one variable (any lead_time, season, region)."""
        return [e for e in self.entries.values() if e.variable == variable]

    def get_entries_for_context(
        self,
        variable: str,
        lead_hours: int | None = None,
        season: str = "",
        region: str = "",
        fallback: bool = True,
    ) -> list[SkillProfileEntry]:
        """Return the best matching entries for a given context.

        Search order (most specific to least):
          1. (variable, lead_hours, season, region)
          2. (variable, lead_hours, season, "")
          3. (variable, lead_hours, "", region)
          4. (variable, lead_hours, "", "")
          5. (variable, None, season, region)
          6. (variable, None, season, "")
          7. (variable, None, "", region)
          8. (variable, None, "", "")  — universal fallback
        """
        if not fallback:
            return [
                e
                for e in self.entries.values()
                if e.variable == variable
                and (lead_hours is None or e.lead_hours == lead_hours)
                and (season == "" or e.season == season)
                and (region == "" or e.region == region)
            ]

        search_order = [
            (lead_hours, season, region),
            (lead_hours, season, ""),
            (lead_hours, "", region),
            (lead_hours, "", ""),
            (None, season, region),
            (None, season, ""),
            (None, "", region),
            (None, "", ""),
        ]
        for lh, s, r in search_order:
            entry = self.get_entries_for_variable(variable)
            matching = [
                e
                for e in entry
                if (lh is None or e.lead_hours == lh)
                and (s == "" or e.season == s)
                and (r == "" or e.region == r)
            ]
            if matching:
                return matching
        return []


# ── Model Weight ───────────────────────────────────────────────────────────


class ModelWeight(BaseModel):
    """One weight for one model in a blending context.

    Includes provenance: why this weight was assigned.
    All weights are non-negative and sum to 1.0 across models.
    """

    model_name: str
    weight: float  # 0.0 to 1.0
    # Contributing factors (for explainability)
    historical_skill_contribution: float = 0.0  # How much historical skill contributed
    recent_skill_contribution: float = 0.0  # How much recent performance contributed
    coverage_contribution: float = 0.0  # How much coverage affected the weight
    stability_contribution: float = 0.0  # How much forecast stability affected the weight
    # Rationale string (human-readable explanation)
    rationale: str = ""
    # Weight computation metadata
    computed_at: datetime | None = None
    weighting_method: str = "deterministic"  # method name for reproducibility


# ── Blended Forecast ───────────────────────────────────────────────────────


class BlendedForecastRecord(BaseModel):
    """One blended value from multiple contributing models.

    Preserves provenance: which models contributed, their weights,
    and the uncertainty from inter-model spread.

    This is NOT a silent average — every contribution is tracked.
    """

    # Blended value
    value: float | None = None

    # Context for this blend
    variable: str
    lead_hours: int
    latitude: float
    longitude: float
    valid_time: datetime
    season: str = ""

    # Contributing models
    contributors: list[dict[str, Any]] = Field(  # {model_name, weight, value}
        default_factory=list
    )

    # Inter-model spread (uncertainty signal)
    inter_model_std: float | None = None  # Std dev of contributing values
    inter_model_range: float | None = None  # max - min of contributing values
    model_agreement: str = "unknown"  # "high", "medium", "low", "insufficient"

    # Weighting provenance
    weighting_rationale: str = ""  # Why these weights were chosen
    weighting_method: str = "adaptive_historical"

    # Coverage
    coverage_fraction: float = 1.0  # fraction of models that contributed
    total_models: int = 0
    active_models: int = 0  # models that actually had values

    # Confidence score (if available)
    confidence_score: float | None = None  # 0.0 to 1.0
    confidence_factors: list[dict[str, Any]] = Field(default_factory=list)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        if isinstance(self.valid_time, datetime):
            data["valid_time"] = self.valid_time.isoformat()
        return data


# ── Forecast Confidence ────────────────────────────────────────────────────


class ForecastConfidence(BaseModel):
    """Confidence assessment for a blended forecast.

    Composed of multiple contributing factors, each with a score and
    rationale. The overall confidence is an interpretable composite.
    """

    confidence_score: float  # 0.0 to 1.0
    # Contributing factors
    historical_error_factor: float = 0.0  # Lower historical error → higher confidence
    model_agreement_factor: float = 0.0  # Higher agreement → higher confidence
    coverage_factor: float = 0.0  # Higher coverage → higher confidence
    lead_time_factor: float = 0.0  # Shorter lead → higher confidence
    stability_factor: float = 0.0  # More stable models → higher confidence
    variable_volatility_factor: float = 0.0  # Less volatile variable → higher confidence

    # Model agreement details
    agreement_details: dict[str, Any] = Field(default_factory=dict)
    inter_model_std: float | None = None
    disagreement_score: float | None = None  # 0..1, higher = more disagreement

    # Rationale
    rationale: str = ""
    factors_summary: list[dict[str, Any]] = Field(default_factory=list)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return super().model_dump(**kwargs)


# ── Regime Selection ───────────────────────────────────────────────────────


class RegimeCondition(BaseModel):
    """A weather regime condition for adaptive model selection.

    Currently supports seasonal and regional conditions.
    Extensible for future storm/precipitation/extreme-weather handling.
    """

    condition_type: str  # "season", "region", "storm", "precipitation_event", "extreme_weather"
    condition_value: str  # e.g., "JJA", "tropical", "hurricane"
    condition_metadata: dict[str, Any] = Field(default_factory=dict)

    def matches(self, context: dict[str, Any]) -> bool:
        """Check if this condition matches a given context dict.

        Args:
            context: Dict with keys like "season", "region", etc.

        Returns:
            True if the condition matches.
        """
        if self.condition_type == "season":
            return context.get("season", "") == self.condition_value
        if self.condition_type == "region":
            return context.get("region", "") == self.condition_value
        # Future types will add their own matching logic
        return False


class RegimeAwareSelection(BaseModel):
    """Adaptive model selection result for one context.

    Selects which models to trust most given the current regime/conditions.
    """

    context: dict[str, Any]  # season, region, etc.
    selected_models: list[dict[str, Any]] = Field(  # {model_name, reason, weight}
        default_factory=list
    )
    excluded_models: list[str] = Field(default_factory=list)
    rationale: str = ""
    regime_conditions_applied: list[RegimeCondition] = Field(default_factory=list)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return super().model_dump(**kwargs)


# ── Ensemble Agreement ─────────────────────────────────────────────────────


class EnsembleAgreement(BaseModel):
    """Analysis of agreement/disagreement among a set of model forecasts.

    Does NOT assume agreement implies correctness. Agreement is an
    uncertainty signal, not a correctness guarantee.
    """

    variable: str
    lead_hours: int
    latitude: float
    longitude: float
    valid_time: datetime

    # Number of models contributing
    total_models: int = 0
    models_with_values: int = 0

    # Agreement metrics
    inter_model_std: float | None = None  # Standard deviation across models
    inter_model_range: float | None = None  # Max - min
    disagreement_score: float = 0.0  # 0..1, higher = more disagreement
    consensus_score: float = 0.0  # 0..1, higher = more agreement

    # Model details
    model_values: list[dict[str, Any]] = Field(  # {model_name, value}
        default_factory=list
    )

    # Interpretation
    agreement_level: str = "unknown"  # "high", "medium", "low", "insufficient"
    interpretation: str = ""  # Human-readable interpretation

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        if isinstance(self.valid_time, datetime):
            data["valid_time"] = self.valid_time.isoformat()
        return data


# ── Meta-Evaluation Result ─────────────────────────────────────────────────


class MetaEvaluationResult(BaseModel):
    """Evaluation of the meta-forecasting system itself.

    Measures whether blending actually improves over the best single model.
    Never assumes blending is better — measures it empirically.
    """

    variable: str
    lead_hours: int | None = None
    metric: str  # "mae", "rmse"

    # Blended forecast performance
    blended_mae: float | None = None
    blended_rmse: float | None = None
    blended_coverage: float = 0.0

    # Best single model performance
    best_single_mae: float | None = None
    best_single_rmse: float | None = None
    best_single_model: str | None = None

    # Improvement metrics, as (blended - best_single) / best_single * 100.
    # These are error deltas, so NEGATIVE means the blend has lower error
    # (better) and positive means the blend is worse than the best model.
    mae_improvement_pct: float | None = None
    rmse_improvement_pct: float | None = None
    improvement_over_best: bool = False  # True if blended outperforms best single

    # Stability metrics
    weight_stability_over_time: float | None = None  # Correlation of weights over time
    blend_stability: float | None = None  # How stable are blends over time (0..1)

    # Degradation curve for blended forecasts
    blended_degradation: list[dict[str, Any]] = Field(default_factory=list)

    # Metadata
    evaluation_window: tuple[datetime, datetime] | None = None
    sample_size: int = 0
    rationale: str = ""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        if self.evaluation_window and isinstance(self.evaluation_window, tuple):
            data["evaluation_window"] = [
                self.evaluation_window[0].isoformat(),
                self.evaluation_window[1].isoformat(),
            ]
        return data
