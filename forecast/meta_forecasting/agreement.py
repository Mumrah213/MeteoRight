"""Ensemble Agreement Analysis.

Analyzes inter-model spread, disagreement, and consensus among
multiple model forecasts.

Key principles:
- Agreement does NOT imply correctness
- Disagreement is an uncertainty signal, not a quality signal
- All metrics are computed from forecast values only (no observation leakage)

Metrics:
- Inter-model standard deviation
- Inter-model range
- Disagreement score (0..1)
- Consensus score (0..1)
- Agreement level classification

Usage:
    analyzer = EnsembleAgreementAnalyzer()
    result = analyzer.analyze(
        forecasts_by_model={"IFS": [...], "GFS": [...]},
        variable="temperature_2m",
        lead_hours=72,
    )
"""


from datetime import datetime
from typing import Any

from ..canonical.models import CanonicalForecastRecord
from .models import EnsembleAgreement

# ── Ensemble Agreement Analyzer ────────────────────────────────────────────


class EnsembleAgreementAnalyzer:
    """Analyze agreement/disagreement among model forecasts.

    Does NOT assume agreement implies correctness. Agreement is an
    uncertainty signal only.

    For each blend key (variable, lead_time, location, valid_time),
    computes inter-model statistics from the forecast values only.
    """

    def __init__(
        self,
        agreement_thresholds: dict[str, float] | None = None,
    ) -> None:
        """Initialize the agreement analyzer.

        Args:
            agreement_thresholds: Dict mapping agreement_level → std threshold.
                Default: {"high": 1.0, "medium": 3.0, "low": float("inf")}
                Adjust per variable type (temperature vs precipitation).
        """
        self._thresholds = agreement_thresholds or {
            "high": 1.0,
            "medium": 3.0,
            "low": float("inf"),
        }

    def analyze(
        self,
        forecasts_by_model: dict[str, list[CanonicalForecastRecord]],
        variable: str | None = None,
        lead_hours: int | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> list[EnsembleAgreement]:
        """Analyze agreement across all models for given filters.

        Groups forecasts by (variable, lead_hours, latitude, longitude,
        valid_time) and computes inter-model statistics.

        Args:
            forecasts_by_model: Dict mapping model_name → forecast records.
            variable: Optional variable filter.
            lead_hours: Optional lead time filter.
            latitude: Optional latitude filter.
            longitude: Optional longitude filter.

        Returns:
            List of EnsembleAgreement objects.
        """
        # Group by blend key
        groups: dict[tuple[str, int | None, float, float, datetime], list[tuple[str, float]]] = {}

        for model_name, records in forecasts_by_model.items():
            for rec in records:
                # Apply filters
                if variable is not None and rec.variable != variable:
                    continue
                if lead_hours is not None and rec.lead_hours != lead_hours:
                    continue
                if latitude is not None and rec.latitude != latitude:
                    continue
                if longitude is not None and rec.longitude != longitude:
                    continue

                key = (
                    rec.variable,
                    rec.lead_hours,
                    rec.latitude,
                    rec.longitude,
                    rec.valid_time,
                )
                if rec.value is not None:
                    if key not in groups:
                        groups[key] = []
                    groups[key].append((model_name, rec.value))

        results: list[EnsembleAgreement] = []

        for key, model_values in groups.items():
            var, lh, lat, lon, vt = key

            # Separate models with values
            models_with_values: list[dict[str, Any]] = [
                {"model_name": mn, "value": v} for mn, v in model_values
            ]

            # Extract numeric values
            values = [v for _, v in model_values]
            n_models = len(models_with_values)

            if n_models < 2:
                # Need at least 2 models for agreement analysis
                result = EnsembleAgreement(
                    variable=var,
                    lead_hours=lh or 0,
                    latitude=lat,
                    longitude=lon,
                    valid_time=vt,
                    total_models=len(forecasts_by_model),
                    models_with_values=n_models,
                    agreement_level="insufficient",
                    interpretation="Only one model has values — cannot assess agreement.",
                    model_values=models_with_values,
                )
                results.append(result)
                continue

            # Compute statistics
            inter_std = self._compute_std(values)
            inter_range = max(values) - min(values)
            disagreement = self._compute_disagreement(values, inter_std)
            consensus = 1.0 - disagreement

            # Determine agreement level
            agreement_level = self._classify_agreement(inter_std)

            # Build interpretation
            interpretation = self._build_interpretation(
                agreement_level, inter_std, n_models, var, lh
            )

            result = EnsembleAgreement(
                variable=var,
                lead_hours=lh or 0,
                latitude=lat,
                longitude=lon,
                valid_time=vt,
                total_models=len(forecasts_by_model),
                models_with_values=n_models,
                inter_model_std=inter_std,
                inter_model_range=inter_range,
                disagreement_score=disagreement,
                consensus_score=consensus,
                model_values=models_with_values,
                agreement_level=agreement_level,
                interpretation=interpretation,
            )
            results.append(result)

        return results

    def analyze_single(
        self,
        forecasts_by_model: dict[str, list[CanonicalForecastRecord]],
        variable: str,
        lead_hours: int,
        latitude: float,
        longitude: float,
        valid_time: datetime,
    ) -> EnsembleAgreement | None:
        """Analyze agreement for a specific key.

        Convenience method for single-location, single-variable analysis.

        Args:
            forecasts_by_model: Dict mapping model_name → forecast records.
            variable: Variable name.
            lead_hours: Lead time.
            latitude: Latitude.
            longitude: Longitude.
            valid_time: Valid time.

        Returns:
            EnsembleAgreement or None if insufficient data.
        """
        results = self.analyze(
            forecasts_by_model=forecasts_by_model,
            variable=variable,
            lead_hours=lead_hours,
            latitude=latitude,
            longitude=longitude,
        )

        for r in results:
            if r.latitude == latitude and r.longitude == longitude and r.valid_time == valid_time:
                return r
        return None

    # ── Statistical Methods ───────────────────────────────────────────

    @staticmethod
    def _compute_std(values: list[float]) -> float | None:
        """Compute standard deviation of values."""
        n = len(values)
        if n < 2:
            return None

        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        return variance**0.5

    @staticmethod
    def _compute_disagreement(
        values: list[float],
        inter_std: float | None,
    ) -> float:
        """Compute disagreement score from 0 (full agreement) to 1 (max disagreement).

        Uses a normalized spread metric. For temperature-like variables,
        we normalize by the typical range.

        Does NOT assume disagreement implies incorrectness.
        """
        if len(values) < 2:
            return 0.0

        # Use coefficient of variation as the disagreement measure
        mean_val = sum(values) / len(values)
        if mean_val == 0:
            # For zero-mean variables, use range relative to a typical scale
            # Temperature range scale ~10°C
            typical_range = 10.0
            range_val = max(values) - min(values)
            norm_spread = min(range_val / typical_range, 1.0)
        else:
            cv = (inter_std or 0) / abs(mean_val)
            # Map CV to [0, 1]
            # CV of 0 = full agreement, CV > 0.5 = max disagreement
            norm_spread = min(cv / 0.5, 1.0)

        return norm_spread

    def _classify_agreement(self, inter_std: float | None) -> str:
        """Classify agreement level based on thresholds."""
        if inter_std is None:
            return "insufficient"

        for level in ("high", "medium", "low"):
            threshold = self._thresholds.get(level, float("inf"))
            if inter_std <= threshold:
                return level

        return "low"

    @staticmethod
    def _build_interpretation(
        level: str,
        inter_std: float | None,
        n_models: int,
        variable: str,
        lead_hours: int | None,
    ) -> str:
        """Build human-readable interpretation of agreement analysis."""
        parts = [f"Agreement level: {level}."]

        if level == "high":
            parts.append(
                f"All {n_models} models agree within ±{inter_std:.1f}°C. "
                f"This suggests consistent signal, but does not guarantee correctness."
            )
        elif level == "medium":
            parts.append(
                f"Models show moderate spread (±{inter_std:.1f}°C) across {n_models} models. "
                f"There is partial consensus with some divergence."
            )
        elif level == "low":
            parts.append(
                f"Strong disagreement among {n_models} models (spread ±{inter_std:.1f}°C). "
                f"Forecast confidence should be reduced."
            )
        else:
            parts.append("Insufficient models for meaningful agreement analysis.")

        if lead_hours is not None and lead_hours > 72:
            parts.append(f"Note: at {lead_hours}h lead time, lower agreement is expected.")

        return " ".join(parts)
