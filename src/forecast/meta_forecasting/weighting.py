"""Dynamic Weighting Engine.

Computes adaptive model weights based on:
- Historical skill (from skill profiles)
- Recent model performance
- Coverage
- Forecast stability

Weights are deterministic, explainable, and causally correct
(only use data available before the forecast target time).

Usage:
    engine = DynamicWeightingEngine(skill_store)
    weights = engine.compute_weights(
        model_names=["IFS", "GFS", "ICON"],
        variable="temperature_2m",
        lead_hours=72,
        season="JJA",
        region="",
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import ModelWeight
from .skill_profiles import SkillProfileStore

# ── Weighting Engine ───────────────────────────────────────────────────────


class DynamicWeightingEngine:
    """Compute adaptive model weights for blending.

    Weight computation:
        weight(model) ∝ f(historical_skill, recent_skill, coverage, stability)

    Each component is normalized to [0, 1] and combined with fixed
    coefficients. All decisions are explainable.

    Coefficients (tunable, documented):
        - historical_skill_weight: 0.4 (skill matters most)
        - recent_skill_weight: 0.25 (recent performance matters)
        - coverage_weight: 0.2 (coverage matters)
        - stability_weight: 0.15 (stability matters)

    These can be overridden via set_coefficients().
    """

    def __init__(
        self,
        skill_store: SkillProfileStore | None = None,
        historical_skill_weight: float = 0.4,
        recent_skill_weight: float = 0.25,
        coverage_weight: float = 0.2,
        stability_weight: float = 0.15,
        min_weight: float = 0.0,  # Minimum weight per model (clamping)
    ) -> None:
        """Initialize the weighting engine.

        Args:
            skill_store: Pre-populated skill profile store.
            historical_skill_weight: Weight for historical skill component.
            recent_skill_weight: Weight for recent performance component.
            coverage_weight: Weight for coverage component.
            stability_weight: Weight for stability component.
            min_weight: Minimum weight per model after normalization.
        """
        self.skill_store = skill_store or SkillProfileStore()
        self._historical_skill_weight = historical_skill_weight
        self._recent_skill_weight = recent_skill_weight
        self._coverage_weight = coverage_weight
        self._stability_weight = stability_weight
        self._min_weight = min_weight

        # Validate coefficients sum to 1.0
        total = historical_skill_weight + recent_skill_weight + coverage_weight + stability_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weighting coefficients must sum to 1.0, got {total}")

    @property
    def coefficients(self) -> dict[str, float]:
        """Return current weighting coefficients."""
        return {
            "historical_skill": self._historical_skill_weight,
            "recent_skill": self._recent_skill_weight,
            "coverage": self._coverage_weight,
            "stability": self._stability_weight,
        }

    def set_coefficients(
        self,
        historical_skill: float | None = None,
        recent_skill: float | None = None,
        coverage: float | None = None,
        stability: float | None = None,
    ) -> None:
        """Update weighting coefficients.

        Unchanged coefficients keep their current values.
        All must be >= 0 and sum to 1.0 after update.
        """
        h = historical_skill if historical_skill is not None else self._historical_skill_weight
        r = recent_skill if recent_skill is not None else self._recent_skill_weight
        c = coverage if coverage is not None else self._coverage_weight
        s = stability if stability is not None else self._stability_weight

        total = h + r + c + s
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Coefficients must sum to 1.0, got {total}. "
                f"Current: historical={h}, recent={r}, coverage={c}, stability={s}"
            )

        self._historical_skill_weight = h
        self._recent_skill_weight = r
        self._coverage_weight = c
        self._stability_weight = s

    def compute_weights(
        self,
        model_names: list[str],
        variable: str,
        lead_hours: int | None = None,
        season: str = "",
        region: str = "",
        recent_evaluations: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[ModelWeight]:
        """Compute adaptive weights for all models in a context.

        Weights are normalized so they sum to 1.0 and each is >= 0.

        Args:
            model_names: Models to weight.
            variable: Variable context.
            lead_hours: Lead time context.
            season: Season context.
            region: Region context.
            recent_evaluations: Optional dict mapping model_name → list of
                recent evaluation results (for recent skill component).

        Returns:
            List of ModelWeight objects, one per model.
        """
        # Compute raw scores for each model
        raw_scores: dict[str, dict[str, float]] = {}

        for model_name in model_names:
            scores = {
                "historical_skill": self._score_historical_skill(
                    model_name, variable, lead_hours, season, region
                ),
                "recent_skill": self._score_recent_skill(model_name, recent_evaluations),
                "coverage": self._score_coverage(model_name, variable, lead_hours, season, region),
                "stability": self._score_stability(
                    model_name, variable, lead_hours, season, region
                ),
            }
            raw_scores[model_name] = scores

        # Combine weighted scores into composite score
        composite: dict[str, float] = {}
        for model_name in model_names:
            scores = raw_scores[model_name]
            composite[model_name] = (
                scores["historical_skill"] * self._historical_skill_weight
                + scores["recent_skill"] * self._recent_skill_weight
                + scores["coverage"] * self._coverage_weight
                + scores["stability"] * self._stability_weight
            )

        # Normalize composite scores to [0, 1] range first
        min_composite = min(composite.values()) if composite else 0.0
        max_composite = max(composite.values()) if composite else 1.0

        normalized: dict[str, float] = {}
        for model_name in model_names:
            if max_composite > min_composite:
                normalized[model_name] = (composite[model_name] - min_composite) / (
                    max_composite - min_composite
                )
            else:
                # All models have same composite — equal weights
                normalized[model_name] = 1.0 / len(model_names) if model_names else 0.0

        # Scale by coefficients and build ModelWeight objects
        weights: list[ModelWeight] = []
        total_weight = 0.0

        for model_name in model_names:
            base_weight = normalized[model_name]

            w = ModelWeight(
                model_name=model_name,
                weight=base_weight,  # Will be normalized
                historical_skill_contribution=scores["historical_skill"]
                * self._historical_skill_weight
                if model_name in raw_scores
                else 0.0,
                recent_skill_contribution=scores["recent_skill"] * self._recent_skill_weight
                if model_name in raw_scores
                else 0.0,
                coverage_contribution=scores["coverage"] * self._coverage_weight
                if model_name in raw_scores
                else 0.0,
                stability_contribution=scores["stability"] * self._stability_weight
                if model_name in raw_scores
                else 0.0,
                rationale=self._build_rationale(
                    model_name, raw_scores[model_name] if model_name in raw_scores else {}
                ),
                computed_at=datetime.now(),
                weighting_method="adaptive_historical",
            )
            weights.append(w)
            total_weight += base_weight

        # Clamp to minimum weight before final normalization
        if self._min_weight > 0:
            min_total = self._min_weight * len(weights)
            if min_total > 1.0:
                raise ValueError(f"min_weight={self._min_weight} * {len(weights)} models > 1.0")
            for w in weights:
                if w.weight < self._min_weight:
                    w.weight = self._min_weight

        # Normalize so weights sum to 1.0 (after clamping)
        total = sum(w.weight for w in weights)
        if total > 0:
            for w in weights:
                w.weight = w.weight / total

        return weights

    def _score_historical_skill(
        self,
        model_name: str,
        variable: str,
        lead_hours: int | None,
        season: str,
        region: str,
    ) -> float:
        """Score historical skill for a model.

        Returns 0.0 (worst) to 1.0 (best), based on relative error
        compared to the best model in this context.

        The best model gets 1.0, others are scaled proportionally.
        If no data exists, returns 0.0.
        """
        profile = self.skill_store.get_profile(model_name)
        if not profile:
            return 0.0

        entries = profile.get_entries_for_context(
            variable=variable,
            lead_hours=lead_hours,
            season=season,
            region=region,
            fallback=True,
        )

        # Find the best error in this context across all known models
        best_error = None
        for other_profile in self.skill_store.get_all_profiles().values():
            other_entries = other_profile.get_entries_for_context(
                variable=variable,
                lead_hours=lead_hours,
                season=season,
                region=region,
                fallback=True,
            )
            for entry in other_entries:
                if entry.metric_type == "mae" and entry.value is not None and entry.is_stable:
                    if best_error is None or entry.value < best_error:
                        best_error = entry.value

        if not entries or best_error is None:
            return 0.0

        # Find this model's best error
        model_best = None
        for entry in entries:
            if entry.metric_type == "mae" and entry.value is not None and entry.is_stable:
                if model_best is None or entry.value < model_best:
                    model_best = entry.value

        if model_best is None:
            return 0.0

        # Convert error to score: lower error → higher score
        # Use inverse scaling: score = best / model_value
        # This gives 1.0 for the best model and < 1.0 for others
        score = best_error / model_best if model_best > 0 else 0.0
        return min(score, 1.0)

    def _score_recent_skill(
        self,
        model_name: str,
        recent_evaluations: dict[str, list[dict[str, Any]]] | None,
    ) -> float:
        """Score recent performance for a model.

        Uses MAE from recent evaluations if provided.
        Returns 0.0 to 1.0 based on relative performance.
        """
        if not recent_evaluations:
            return 0.5  # Neutral score when no recent data

        evals = recent_evaluations.get(model_name, [])
        if not evals:
            return 0.0

        # Compute average recent MAE
        mae_values = [
            e["value"]
            for e in evals
            if e.get("metric_type") == "mae" and e.get("value") is not None
        ]
        if not mae_values:
            return 0.0

        avg_mae = sum(mae_values) / len(mae_values)

        # Score relative to best recent model (if available)
        all_maes = []
        for m_evals in recent_evaluations.values():
            for e in m_evals:
                if e.get("metric_type") == "mae" and e.get("value") is not None:
                    all_maes.append(e["value"])

        if not all_maes:
            return 0.5  # Neutral

        best_mae = min(all_maes)
        score = best_mae / avg_mae if avg_mae > 0 else 0.0
        return min(score, 1.0)

    def _score_coverage(
        self,
        model_name: str,
        variable: str,
        lead_hours: int | None,
        season: str,
        region: str,
    ) -> float:
        """Score model coverage for a given context.

        Returns 0.0 to 1.0 based on the coverage_fraction from skill entries.
        """
        profile = self.skill_store.get_profile(model_name)
        if not profile:
            return 0.0

        entries = profile.get_entries_for_context(
            variable=variable,
            lead_hours=lead_hours,
            season=season,
            region=region,
            fallback=True,
        )

        if not entries:
            return 0.0

        # Average coverage across relevant entries
        coverages = [e.coverage_fraction for e in entries if e.coverage_fraction is not None]
        if not coverages:
            return 0.0

        return min(sum(coverages) / len(coverages), 1.0)

    def _score_stability(
        self,
        model_name: str,
        variable: str,
        lead_hours: int | None,
        season: str,
        region: str,
    ) -> float:
        """Score forecast stability for a model.

        Returns 1.0 if all entries are stable, proportionally lower
        if some are unstable.
        """
        profile = self.skill_store.get_profile(model_name)
        if not profile:
            return 0.0

        entries = profile.get_entries_for_context(
            variable=variable,
            lead_hours=lead_hours,
            season=season,
            region=region,
            fallback=True,
        )

        if not entries:
            return 0.0

        stable_count = sum(1 for e in entries if e.is_stable)
        return stable_count / len(entries)

    def _build_rationale(
        self,
        model_name: str,
        scores: dict[str, float],
    ) -> str:
        """Build a human-readable rationale for model weights."""
        parts = []

        skill_score = scores.get("historical_skill", 0.0)
        if skill_score >= 0.8:
            parts.append(f"strong historical skill (score={skill_score:.2f})")
        elif skill_score >= 0.5:
            parts.append(f"moderate historical skill (score={skill_score:.2f})")
        elif skill_score > 0.0:
            parts.append(f"limited historical skill (score={skill_score:.2f})")
        else:
            parts.append("no historical skill data")

        recent = scores.get("recent_skill", 0.0)
        if recent >= 0.8:
            parts.append("consistent recent performance")
        elif recent > 0.0:
            parts.append(f"limited recent performance (score={recent:.2f})")
        else:
            parts.append("no recent data")

        cov = scores.get("coverage", 0.0)
        if cov >= 0.8:
            parts.append("full coverage")
        elif cov > 0.0:
            parts.append(f"partial coverage (score={cov:.2f})")
        else:
            parts.append("no coverage data")

        stab = scores.get("stability", 0.0)
        if stab >= 0.8:
            parts.append("stable forecasts")
        elif stab > 0.0:
            parts.append(f"partially stable (score={stab:.2f})")
        else:
            parts.append("unstable or no stability data")

        return "; ".join(parts)
