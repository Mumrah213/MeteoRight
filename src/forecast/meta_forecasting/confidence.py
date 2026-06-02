"""Forecast Confidence / Reliability Scoring.

Estimates confidence for blended forecasts based on:
- Historical model error (lower → higher confidence)
- Model agreement/disagreement (higher agreement → higher confidence)
- Coverage stability (more models → higher confidence)
- Lead time degradation (shorter → higher confidence)
- Variable volatility (less volatile → higher confidence)

Output: ForecastConfidence with score, uncertainty band hooks,
contributing factors, and model agreement metrics.

No black-box assumptions — all factors are explicitly scored
and explainable.
"""

from __future__ import annotations

from typing import Any

from .models import ForecastConfidence

# ── Confidence Scorer ─────────────────────────────────────────────────────


class ConfidenceScorer:
    """Score forecast confidence from multiple contributing factors.

    Each factor produces a score in [0, 1]. The composite is a weighted
    average of factors, with explicit coefficients.

    Factors:
        1. Historical error — how well models perform historically
        2. Model agreement — how much models agree on the value
        3. Coverage — how many models contribute
        4. Lead time — shorter leads are more confident
        5. Stability — how stable models are
        6. Variable volatility — some variables are inherently noisier
    """

    def __init__(
        self,
        historical_error_weight: float = 0.3,
        agreement_weight: float = 0.25,
        coverage_weight: float = 0.15,
        lead_time_weight: float = 0.15,
        stability_weight: float = 0.1,
        volatility_weight: float = 0.05,
    ) -> None:
        """Initialize the confidence scorer.

        Args:
            historical_error_weight: Weight for historical error factor.
            agreement_weight: Weight for model agreement factor.
            coverage_weight: Weight for coverage factor.
            lead_time_weight: Weight for lead time factor.
            stability_weight: Weight for stability factor.
            volatility_weight: Weight for variable volatility factor.
        """
        total = (
            historical_error_weight
            + agreement_weight
            + coverage_weight
            + lead_time_weight
            + stability_weight
            + volatility_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Confidence weights must sum to 1.0, got {total}")

        self._weights = {
            "historical_error": historical_error_weight,
            "agreement": agreement_weight,
            "coverage": coverage_weight,
            "lead_time": lead_time_weight,
            "stability": stability_weight,
            "volatility": volatility_weight,
        }

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def compute_confidence(
        self,
        variable: str,
        lead_hours: int | None = None,
        inter_model_std: float | None = None,
        coverage: float = 1.0,
        models_with_values: int = 0,
        total_models: int = 0,
        weights: list[Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ForecastConfidence:
        """Compute confidence for a forecast blend.

        All factors are scored independently and combined with weights.
        Each factor includes its own rationale for explainability.

        Args:
            variable: Variable being forecasted.
            lead_hours: Lead time of the forecast.
            inter_model_std: Standard deviation across contributing models.
            coverage: Fraction of available models that contributed (0..1).
            models_with_values: Number of models with non-null values.
            total_models: Total number of models in the blend pool.
            weights: Optional weighting objects for stability analysis.
            context: Optional context dict (season, region, etc.).

        Returns:
            ForecastConfidence with score and contributing factors.
        """
        # Score each factor
        historical_score = self._score_historical_error(variable)
        agreement_score = self._score_agreement(inter_model_std, lead_hours)
        coverage_score = self._score_coverage(coverage)
        lead_time_score = self._score_lead_time(lead_hours)
        stability_score = self._score_stability(weights)
        volatility_score = self._score_variable_volatility(variable)

        # Composite score (weighted average)
        composite = (
            historical_score * self._weights["historical_error"]
            + agreement_score * self._weights["agreement"]
            + coverage_score * self._weights["coverage"]
            + lead_time_score * self._weights["lead_time"]
            + stability_score * self._weights["stability"]
            + volatility_score * self._weights["volatility"]
        )

        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        # Compute disagreement score (1 - agreement)
        disagreement_score = max(0.0, 1.0 - agreement_score)

        # Build factors summary as list of dicts
        factors_summary = [
            {
                "factor": "historical_error",
                "score": historical_score,
                "rationale": self._historical_rationale(variable),
            },
            {
                "factor": "model_agreement",
                "score": agreement_score,
                "rationale": self._agreement_rationale(inter_model_std, lead_hours),
            },
            {
                "factor": "coverage",
                "score": coverage_score,
                "rationale": self._coverage_rationale(coverage, models_with_values, total_models),
            },
            {
                "factor": "lead_time",
                "score": lead_time_score,
                "rationale": self._lead_time_rationale(lead_hours),
            },
            {
                "factor": "stability",
                "score": stability_score,
                "rationale": self._stability_rationale(weights),
            },
            {
                "factor": "variable_volatility",
                "score": volatility_score,
                "rationale": self._volatility_rationale(variable),
            },
        ]

        # Build overall rationale
        rationale = self._build_rationale(composite, factors_summary)

        # Agreement details
        agreement_details = {
            "inter_model_std": inter_model_std,
            "disagreement_score": disagreement_score,
            "models_with_values": models_with_values,
            "total_models": total_models,
            "coverage": coverage,
        }

        return ForecastConfidence(
            confidence_score=composite,
            historical_error_factor=historical_score,
            model_agreement_factor=agreement_score,
            coverage_factor=coverage_score,
            lead_time_factor=lead_time_score,
            stability_factor=stability_score,
            variable_volatility_factor=volatility_score,
            agreement_details=agreement_details,
            inter_model_std=inter_model_std,
            disagreement_score=disagreement_score,
            rationale=rationale,
            factors_summary=factors_summary,
        )

    # ── Individual Factor Scorers ──────────────────────────────────────

    def _score_historical_error(self, variable: str) -> float:
        """Score based on historical error for a variable.

        Lower historical error → higher confidence.
        Uses variable-type-based heuristics (since we don't have
        real historical data in the scorer itself).
        """
        # Temperature is generally predictable (high confidence base)
        if "temp" in variable.lower():
            return 0.7
        # Precipitation is harder to predict
        if "precip" in variable.lower():
            return 0.4
        # Wind is moderate
        if "wind" in variable.lower():
            return 0.5
        # Humidity is moderate
        if "humid" in variable.lower():
            return 0.55
        # Pressure is very predictable
        if "pressure" in variable.lower() or "pres" in variable.lower():
            return 0.8
        return 0.5  # Unknown variable → neutral

    def _score_agreement(self, inter_model_std: float | None, lead_hours: int | None) -> float:
        """Score based on model agreement.

        Higher agreement (lower std) → higher confidence.
        But we do NOT assume agreement implies correctness.
        """
        if inter_model_std is None:
            return 0.5  # No agreement data → neutral

        # Use a heuristic: for temperature, std < 1°C = high agreement
        # Scale inversely with std
        if inter_model_std <= 0.5:
            return 0.95  # Very high agreement
        elif inter_model_std <= 1.0:
            return 0.8
        elif inter_model_std <= 2.0:
            return 0.6
        elif inter_model_std <= 5.0:
            return 0.4
        else:
            return 0.2  # Very low agreement

    def _score_coverage(self, coverage: float) -> float:
        """Score based on model coverage.

        More models contributing → higher confidence.
        """
        if coverage >= 1.0:
            return 1.0
        elif coverage >= 0.75:
            return 0.8
        elif coverage >= 0.5:
            return 0.6
        elif coverage >= 0.25:
            return 0.4
        else:
            return 0.2

    def _score_lead_time(self, lead_hours: int | None) -> float:
        """Score based on lead time degradation.

        Shorter leads → higher confidence.
        """
        if lead_hours is None:
            return 0.5  # No lead time info → neutral
        if lead_hours <= 24:
            return 0.9
        elif lead_hours <= 48:
            return 0.7
        elif lead_hours <= 72:
            return 0.5
        elif lead_hours <= 120:
            return 0.3
        else:
            return 0.1

    def _score_stability(self, weights: list[Any] | None) -> float:
        """Score based on forecast stability (weight consistency).

        More stable weights → higher confidence.
        """
        if not weights:
            return 0.5  # No weight data → neutral

        # Compute weight variance across models
        weight_values = [w.weight for w in weights if hasattr(w, "weight")]
        if not weight_values or len(weight_values) < 2:
            return 0.5

        mean_w = sum(weight_values) / len(weight_values)
        variance = sum((w - mean_w) ** 2 for w in weight_values) / len(weight_values)

        # Low variance = stable weights = high confidence
        if variance < 0.01:
            return 0.9
        elif variance < 0.05:
            return 0.7
        elif variance < 0.1:
            return 0.5
        else:
            return 0.3

    def _score_variable_volatility(self, variable: str) -> float:
        """Score based on variable volatility.

        Less volatile variables → higher confidence.
        """
        # Temperature is relatively stable
        if "temp" in variable.lower():
            return 0.8
        # Precipitation is highly volatile
        if "precip" in variable.lower():
            return 0.3
        # Wind is moderately volatile
        if "wind" in variable.lower():
            return 0.5
        # Humidity is moderate
        if "humid" in variable.lower():
            return 0.6
        # Pressure is stable
        if "pressure" in variable.lower() or "pres" in variable.lower():
            return 0.9
        return 0.5

    # ── Rationale Builders ────────────────────────────────────────────

    @staticmethod
    def _historical_rationale(variable: str) -> str:
        name = variable.lower()
        if "temp" in name:
            return "Temperature forecasts have moderate historical error"
        if "precip" in name:
            return "Precipitation forecasts have higher historical error"
        if "wind" in name:
            return "Wind forecasts have moderate historical error"
        return "Unknown variable; historical error score is neutral"

    @staticmethod
    def _agreement_rationale(inter_model_std: float | None, lead_hours: int | None) -> str:
        parts = []
        if inter_model_std is not None:
            if inter_model_std < 1.0:
                parts.append("High model agreement (low spread)")
            elif inter_model_std < 2.0:
                parts.append("Moderate model agreement")
            else:
                parts.append("Low model agreement (high spread)")
        if lead_hours is not None and lead_hours > 72:
            parts.append("(note: agreement does not guarantee correctness at long leads)")
        return "; ".join(parts) if parts else "No agreement data available"

    @staticmethod
    def _coverage_rationale(coverage: float, models_with: int, total: int) -> str:
        if coverage >= 0.75:
            return f"Good coverage: {models_with}/{total} models contributed"
        elif coverage >= 0.5:
            return f"Partial coverage: {models_with}/{total} models contributed"
        elif coverage > 0:
            return f"Poor coverage: {models_with}/{total} models contributed"
        return "No coverage data"

    @staticmethod
    def _lead_time_rationale(lead_hours: int | None) -> str:
        if lead_hours is None:
            return "No lead time information"
        if lead_hours <= 24:
            return f"Short lead time ({lead_hours}h) — higher reliability expected"
        elif lead_hours <= 72:
            return f"Moderate lead time ({lead_hours}h)"
        else:
            return f"Long lead time ({lead_hours}h) — degradation expected"

    @staticmethod
    def _stability_rationale(weights: list[Any] | None) -> str:
        if not weights:
            return "No weight data for stability assessment"
        return "Stable weighting across models"

    @staticmethod
    def _volatility_rationale(variable: str) -> str:
        name = variable.lower()
        if "temp" in name:
            return "Temperature is relatively stable"
        if "precip" in name:
            return "Precipitation is inherently volatile"
        return "Unknown volatility profile"

    @staticmethod
    def _build_rationale(composite: float, factors: list[tuple[str, float, str] | dict]) -> str:
        """Build overall confidence rationale from factors."""
        parts = [f"Overall confidence: {composite:.1%}"]
        for factor in factors:
            if isinstance(factor, dict):
                name = factor.get("factor", "unknown")
                score = factor.get("score", 0.0)
                factor_rationale = factor.get("rationale", "")
            else:
                name, score, factor_rationale = factor
            if score > 0.7:
                status = "strong"
            elif score > 0.4:
                status = "moderate"
            else:
                status = "weak"
            parts.append(f"{name}: {status} ({factor_rationale})")
        return ". ".join(parts) + "."
