"""Forecast Blending Engine.

Implements weighted forecast blending across multiple models.

Input:
    - Multiple model forecasts (CanonicalForecastRecord)
    - Dynamic model weights (from DynamicWeightingEngine)

Output:
    - Blended canonical forecast (BlendedForecastRecord)

Requirements:
    - Preserve provenance (contributing models, weights)
    - Preserve uncertainty metadata (inter-model spread)
    - No silent averaging (all contributions tracked)
    - Deterministic and reproducible

Usage:
    blender = ForecastBlender(weighting_engine)
    blended = blender.blend(
        forecasts_by_model={"IFS": [...], "GFS": [...]},
        weights=[ModelWeight(...), ...],
        variable="temperature_2m",
        lead_hours=72,
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..canonical.models import CanonicalForecastRecord
from .confidence import ConfidenceScorer
from .models import (
    BlendedForecastRecord,
)
from .weighting import DynamicWeightingEngine

# ── Forecast Blender ───────────────────────────────────────────────────────


class ForecastBlender:
    """Blend multiple model forecasts using adaptive weights.

    For each (variable, lead_time, location, valid_time), finds all
    contributing model forecasts and produces a weighted blend.

    Inter-model spread is preserved as uncertainty metadata.
    Every contribution is tracked for explainability.
    """

    def __init__(
        self,
        weighting_engine: DynamicWeightingEngine | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        min_models_for_blend: int = 1,
        blend_method: str = "weighted_average",
    ) -> None:
        """Initialize the blending engine.

        Args:
            weighting_engine: Optional pre-configured weighting engine.
            confidence_scorer: Optional confidence scorer.
            min_models_for_blend: Minimum models required for a blend.
                                  If 1, single-model forecasts are passed through.
            blend_method: Currently only "weighted_average" is supported.
        """
        self.weighting_engine = weighting_engine or DynamicWeightingEngine()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self._min_models = min_models_for_blend
        self._blend_method = blend_method

    @property
    def blend_method(self) -> str:
        return self._blend_method

    def blend(
        self,
        forecasts_by_model: dict[str, list[CanonicalForecastRecord]],
        weights: list[Any] | None = None,
        variable: str | None = None,
        lead_hours: int | None = None,
        season: str = "",
        region: str = "",
    ) -> list[BlendedForecastRecord]:
        """Produce blended forecasts from multiple models.

        For each unique (variable, lead_time, latitude, longitude, valid_time),
        finds all model forecasts and blends them using adaptive weights.

        If weights are not provided, they are computed dynamically using the
        weighting engine.

        Args:
            forecasts_by_model: Dict mapping model_name → forecast records.
            weights: Optional pre-computed weights (ModelWeight objects).
            variable: Optional variable filter.
            lead_hours: Optional lead time filter.
            season: Optional season context.
            region: Optional region context.

        Returns:
            List of BlendedForecastRecord objects.
        """
        if self._blend_method != "weighted_average":
            raise ValueError(f"blend_method '{self._blend_method}' not yet supported")

        # Determine variable and lead time filters
        filtered = {}
        for model_name, records in forecasts_by_model.items():
            filtered[model_name] = [
                r
                for r in records
                if (variable is None or r.variable == variable)
                and (lead_hours is None or r.lead_hours == lead_hours)
            ]

        # Group by blend key: (variable, lead_hours, lat, lon, valid_time)
        grouped: dict[
            tuple[str, int | None, float, float, datetime],
            list[tuple[str, CanonicalForecastRecord]],
        ] = {}

        for model_name, records in filtered.items():
            for rec in records:
                key = (
                    rec.variable,
                    rec.lead_hours,
                    rec.latitude,
                    rec.longitude,
                    rec.valid_time,
                )
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append((model_name, rec))

        # Compute weights if not provided
        if weights is None and forecasts_by_model:
            variable_filter = (
                variable
                or list(set(r.variable for recs in forecasts_by_model.values() for r in recs))[0]
                if forecasts_by_model
                else None
            )
            if variable_filter:
                weights = self.weighting_engine.compute_weights(
                    model_names=list(forecasts_by_model.keys()),
                    variable=variable_filter,
                    lead_hours=lead_hours,
                    season=season,
                    region=region,
                )

        # Produce blended records
        blends: list[BlendedForecastRecord] = []

        for key, model_records in grouped.items():
            var, lh, lat, lon, vt = key

            # Filter to models with actual values
            contributors = []
            for model_name, rec in model_records:
                if rec.value is not None:
                    contributors.append((model_name, rec))

            if len(contributors) < 1:
                continue  # No values to blend

            # Get weights for contributing models
            model_weights: dict[str, float] = {}
            if weights:
                for w in weights:
                    model_weights[w.model_name] = w.weight

            # Compute blend
            blended_value = self._weighted_average(contributors, model_weights)

            # Compute inter-model spread
            values = [rec.value for _, rec in contributors]
            inter_std = self._compute_std(values)
            inter_range = max(values) - min(values) if values else 0.0

            # Determine agreement level
            agreement_level = self._determine_agreement(inter_std, values)

            # Build confidence
            confidence = self.confidence_scorer.compute_confidence(
                variable=var,
                lead_hours=lh,
                inter_model_std=inter_std,
                coverage=len(contributors) / max(len(forecasts_by_model), 1),
                models_with_values=len(contributors),
                total_models=len(forecasts_by_model),
                weights=weights,
                context={"season": season, "region": region},
            )

            # Build contributing model info
            contrib_info = []
            for model_name, rec in contributors:
                w = model_weights.get(model_name, 1.0 / len(contributors))
                contrib_info.append(
                    {
                        "model_name": model_name,
                        "weight": w,
                        "value": rec.value,
                    }
                )

            # Build rationale
            rationale = self._build_blend_rationale(contributors, model_weights, var, lh)

            blend = BlendedForecastRecord(
                value=blended_value,
                variable=var,
                lead_hours=lh or 0,
                latitude=lat,
                longitude=lon,
                valid_time=vt,
                season=season,
                contributors=contrib_info,
                inter_model_std=inter_std,
                inter_model_range=inter_range,
                model_agreement=agreement_level,
                weighting_rationale=rationale,
                weighting_method="adaptive_historical",
                coverage_fraction=len(contributors) / max(len(forecasts_by_model), 1),
                total_models=len(forecasts_by_model),
                active_models=len(contributors),
                confidence_score=confidence.confidence_score if confidence else None,
                confidence_factors=confidence.factors_summary if confidence else [],
            )
            blends.append(blend)

        return blends

    def blend_single(
        self,
        forecasts_by_model: dict[str, list[CanonicalForecastRecord]],
        weights: list[Any] | None = None,
        variable: str | None = None,
        lead_hours: int | None = None,
        season: str = "",
    ) -> BlendedForecastRecord | None:
        """Produce one blended forecast for a specific key.

        Convenience method that returns a single blend result instead of
        a list. Useful for single-location forecasts.

        Args:
            forecasts_by_model: Dict mapping model_name → forecast records.
            weights: Optional pre-computed weights.
            variable: Variable filter.
            lead_hours: Lead time filter.
            season: Season context.

        Returns:
            Single BlendedForecastRecord, or None if no data.
        """
        blends = self.blend(
            forecasts_by_model=forecasts_by_model,
            weights=weights,
            variable=variable,
            lead_hours=lead_hours,
            season=season,
        )
        return blends[0] if blends else None

    def _weighted_average(
        self,
        contributors: list[tuple[str, CanonicalForecastRecord]],
        model_weights: dict[str, float],
    ) -> float | None:
        """Compute weighted average of contributing model values.

        Only includes models with actual (non-null) values.
        Weights are renormalized if not all models contributed.

        Args:
            contributors: List of (model_name, record) with non-null values.
            model_weights: Dict mapping model_name → weight.

        Returns:
            Weighted average, or None if no data.
        """
        if not contributors:
            return None

        total_weight = 0.0
        weighted_sum = 0.0

        for model_name, rec in contributors:
            w = model_weights.get(model_name, 1.0)
            weighted_sum += w * rec.value
            total_weight += w

        if total_weight == 0:
            # Fall back to simple average
            values = [rec.value for _, rec in contributors]
            return sum(values) / len(values) if values else None

        return weighted_sum / total_weight

    @staticmethod
    def _compute_std(values: list[float]) -> float | None:
        """Compute standard deviation of a list of values."""
        n = len(values)
        if n < 2:
            return None

        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        return variance**0.5

    @staticmethod
    def _determine_agreement(
        inter_std: float | None,
        values: list[float],
    ) -> str:
        """Determine agreement level from inter-model spread.

        Uses coefficient of variation (CV = std/mean) for relative spread.
        """
        if not values or len(values) < 2:
            return "insufficient"

        mean_val = sum(values) / len(values)
        if mean_val == 0:
            # Use range instead for zero-mean variables
            if inter_std is not None and inter_std > 10.0:
                return "low"
            elif inter_std is not None and inter_std > 1.0:
                return "medium"
            return "high"

        cv = (inter_std or 0) / abs(mean_val)

        if cv < 0.05:
            return "high"
        elif cv < 0.2:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _build_blend_rationale(
        contributors: list[tuple[str, CanonicalForecastRecord]],
        model_weights: dict[str, float],
        variable: str,
        lead_hours: int | None,
    ) -> str:
        """Build human-readable rationale for a blend decision."""
        parts = [f"Blended {len(contributors)} models for '{variable}'"]
        if lead_hours is not None:
            parts.append(f"at {lead_hours}h lead time")

        weight_parts = []
        for model_name, weight in sorted(model_weights.items(), key=lambda x: -x[1]):
            weight_parts.append(f"{model_name}={weight:.2f}")
        parts.append("weights: " + ", ".join(weight_parts))

        return "; ".join(parts)
