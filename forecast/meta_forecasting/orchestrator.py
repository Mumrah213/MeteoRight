"""Meta-Forecasting Orchestrator.

High-level API that ties together all meta-forecasting components:
- Historical skill profiles
- Dynamic weighting
- Forecast blending
- Confidence scoring
- Ensemble agreement
- Regime-aware selection
- Meta-evaluation

Usage:
    orchestrator = MetaForecastingOrchestrator()

    # Update skill profiles from benchmark data
    orchestrator.update_skill_profiles(evaluations)

    # Get the best model for a context
    best = orchestrator.get_best_model(
        variable="temperature_2m",
        lead_hours=72,
        season="JJA",
    )

    # Blend forecasts from multiple models
    blended = orchestrator.blend_forecasts(
        forecasts_by_model={"IFS": [...], "GFS": [...]},
        variable="temperature_2m",
        lead_hours=72,
    )

    # Get confidence for the blend
    confidence = orchestrator.get_confidence(blended)

    # Analyze ensemble agreement
    agreement = orchestrator.analyze_agreement(
        forecasts_by_model={"IFS": [...], "GFS": [...]},
        variable="temperature_2m",
    )

    # Select models for a regime
    selection = orchestrator.select_for_regime(
        variable="temperature_2m",
        context={"season": "JJA", "region": "tropical"},
    )
"""


from typing import Any

from ..canonical.models import CanonicalForecastRecord
from ..verification.models import ForecastEvaluation
from .agreement import EnsembleAgreementAnalyzer
from .blending import ForecastBlender
from .confidence import ConfidenceScorer
from .evaluation import MetaEvaluator
from .models import (
    BlendedForecastRecord,
    ForecastConfidence,
    ModelWeight,
    RegimeAwareSelection,
)
from .regime_selection import RegimeAwareModelSelector
from .skill_profiles import SkillProfileStore
from .weighting import DynamicWeightingEngine

# ── Orchestrator ───────────────────────────────────────────────────────────


class MetaForecastingOrchestrator:
    """Orchestrate adaptive meta-forecasting operations.

    This is the main entry point for meta-forecasting. It provides a unified API
    for all meta-forecasting operations while keeping each component
    independently replaceable.
    """

    def __init__(
        self,
        skill_store: SkillProfileStore | None = None,
        weighting_engine: DynamicWeightingEngine | None = None,
        blending_engine: ForecastBlender | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        agreement_analyzer: EnsembleAgreementAnalyzer | None = None,
        regime_selector: RegimeAwareModelSelector | None = None,
        meta_evaluator: MetaEvaluator | None = None,
    ) -> None:
        """Initialize the orchestrator.

        All components are optional — defaults are created internally.
        This allows partial initialization (e.g., only skill profiles
        without blending).
        """
        self.skill_store = skill_store or SkillProfileStore()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.weighting_engine = weighting_engine or DynamicWeightingEngine(self.skill_store)
        self.blending_engine = blending_engine or ForecastBlender(
            self.weighting_engine, self.confidence_scorer
        )
        self.agreement_analyzer = agreement_analyzer or EnsembleAgreementAnalyzer()
        self.regime_selector = regime_selector or RegimeAwareModelSelector(self.skill_store)
        self.meta_evaluator = meta_evaluator or MetaEvaluator(self.skill_store)

    # ── Skill Profile Management ──────────────────────────────────────

    def update_skill_profiles(
        self,
        evaluations: list[ForecastEvaluation],
        model_name: str,
        region: str = "",
    ) -> int:
        """Update skill profiles from verification evaluations.

        Args:
            evaluations: ForecastEvaluation objects from benchmarking.
            model_name: Which model these evaluations belong to.
            region: Optional region label.

        Returns:
            Number of profile entries updated.
        """
        return self.skill_store.update_from_evaluations(evaluations, model_name, region)

    def get_best_model(
        self,
        variable: str,
        lead_hours: int | None = None,
        season: str = "",
        region: str = "",
        metric_type: str = "mae",
    ) -> tuple[str, float | None] | None:
        """Get the best model for a given context.

        Args:
            variable: Variable name.
            lead_hours: Optional lead time.
            season: Optional season.
            region: Optional region.
            metric_type: Metric type.

        Returns:
            (model_name, value) or None.
        """
        return self.skill_store.get_best_model(
            variable=variable,
            lead_hours=lead_hours,
            season=season,
            region=region,
            metric_type=metric_type,
        )

    def get_skill_ranking(
        self,
        variable: str,
        lead_hours: int | None = None,
        season: str = "",
        region: str = "",
        metric_type: str = "mae",
    ) -> list[tuple[str, float | None]]:
        """Get full model ranking for a context."""
        return self.skill_store.get_skill_ranking(
            variable=variable,
            lead_hours=lead_hours,
            season=season,
            region=region,
            metric_type=metric_type,
        )

    # ── Forecast Blending ─────────────────────────────────────────────

    def blend_forecasts(
        self,
        forecasts_by_model: dict[str, list[CanonicalForecastRecord]],
        variable: str | None = None,
        lead_hours: int | None = None,
        season: str = "",
        region: str = "",
    ) -> list[BlendedForecastRecord]:
        """Produce blended forecasts from multiple models.

        Args:
            forecasts_by_model: Dict mapping model_name → forecast records.
            variable: Optional variable filter.
            lead_hours: Optional lead time filter.
            season: Optional season context.
            region: Optional region context.

        Returns:
            List of blended forecast records.
        """
        return self.blending_engine.blend(
            forecasts_by_model=forecasts_by_model,
            weights=None,  # Compute dynamically
            variable=variable,
            lead_hours=lead_hours,
            season=season,
            region=region,
        )

    def get_model_weights(
        self,
        model_names: list[str],
        variable: str,
        lead_hours: int | None = None,
        season: str = "",
        region: str = "",
    ) -> list[ModelWeight]:
        """Get dynamically computed weights for models."""
        return self.weighting_engine.compute_weights(
            model_names=model_names,
            variable=variable,
            lead_hours=lead_hours,
            season=season,
            region=region,
        )

    # ── Confidence ────────────────────────────────────────────────────

    def get_confidence(
        self,
        blended: BlendedForecastRecord,
    ) -> ForecastConfidence:
        """Get confidence score for a blended forecast."""
        return self.confidence_scorer.compute_confidence(
            variable=blended.variable,
            lead_hours=blended.lead_hours,
            inter_model_std=blended.inter_model_std,
            coverage=blended.coverage_fraction,
            models_with_values=blended.active_models,
            total_models=blended.total_models,
            context={"season": blended.season},
        )

    # ── Ensemble Agreement ────────────────────────────────────────────

    def analyze_agreement(
        self,
        forecasts_by_model: dict[str, list[CanonicalForecastRecord]],
        variable: str | None = None,
        lead_hours: int | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[Any]:
        """Analyze ensemble agreement across models."""
        return self.agreement_analyzer.analyze(
            forecasts_by_model=forecasts_by_model,
            variable=variable,
            lead_hours=lead_hours,
            latitude=latitude,
            longitude=longitude,
        )

    # ── Regime Selection ──────────────────────────────────────────────

    def select_for_regime(
        self,
        model_names: list[str],
        variable: str,
        context: dict[str, Any],
    ) -> RegimeAwareSelection:
        """Select models for a specific regime/conditions."""
        return self.regime_selector.select(
            model_names=model_names,
            context=context,
            variable=variable,
        )

    # ── Meta-Evaluation ───────────────────────────────────────────────

    def evaluate_meta_forecasting(
        self,
        blended_forecasts: list[dict[str, Any]],
        single_model_forecasts: dict[str, list[dict[str, Any]]],
        observations: list[dict[str, Any]],
        variable: str,
        metric: str = "mae",
    ) -> Any:
        """Evaluate whether blending improves over best single model."""
        return self.meta_evaluator.evaluate(
            blended_forecasts=blended_forecasts,
            single_model_forecasts=single_model_forecasts,
            observations=observations,
            variable=variable,
            metric=metric,
        )

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the orchestrator state (mainly skill profiles)."""
        return {
            "skill_store": self.skill_store.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetaForecastingOrchestrator:
        """Deserialize the orchestrator state."""
        store = SkillProfileStore.from_dict(data.get("skill_store", {}))
        return cls(skill_store=store)
