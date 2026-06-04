"""Phase 4: Seasonal benchmarking for multi-model comparison.

Enables meteorological seasonal stratification of model skill:
- Group by DJF (Dec-Feb), MAM (Mar-May), JJA (Jun-Aug), SON (Sep-Nov)
- Compute per-season metrics for each model
- Report seasonal skill profiles
- Track which season a model performs best/worst

Seasonal analysis is critical because:
- Model skill can vary significantly by season
- Blindly aggregating across seasons hides seasonal biases
- Comparisons should be season-aware, not season-blinded
"""


from collections import defaultdict

from ...forecast.canonical.models import CanonicalForecastRecord
from ...forecast.verification.alignment import AlignmentEngine
from ...forecast.verification.metrics import MetricsEngine
from ...forecast.verification.models import AlignmentRecord, _month_to_season
from .models import (
    ModelSkillScore,
    SeasonalSkillProfile,
)

# Meteorological season mappings
SEASON_MONTHS: dict[str, list[int]] = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}


class SeasonalBenchmark:
    """Seasonal benchmarking for multi-model comparison.

    Supports:
    - Meteorological season grouping (DJF, MAM, JJA, SON)
    - Per-season metrics computation per model
    - Seasonal skill profiles
    - Best/worst season identification

    Usage:
        seasonal = SeasonalBenchmark()
        profile = seasonal.benchmark(
            models_data={...},
            observations=obs_records,
            variable="temperature_2m",
            metric="mae",
        )
    """

    def __init__(
        self,
        metrics_engine: MetricsEngine | None = None,
        alignment_engine: AlignmentEngine | None = None,
    ) -> None:
        """Initialize seasonal benchmark.

        Args:
            metrics_engine: Optional MetricsEngine
            alignment_engine: Optional AlignmentEngine
        """
        self.metrics_engine = metrics_engine or MetricsEngine()
        self.alignment_engine = alignment_engine or AlignmentEngine()

    def _assign_season(self, record: CanonicalForecastRecord) -> str:
        """Derive season from a record's valid_time."""
        return _month_to_season(record.valid_time.month)

    def benchmark(
        self,
        models_data: dict[str, list[CanonicalForecastRecord]],
        observations: list[CanonicalForecastRecord],
        variable: str,
        metric: str = "mae",
        min_sample_size: int = 1,
    ) -> dict[str, SeasonalSkillProfile]:
        """Run seasonal benchmark for one variable.

        For each model:
        1. Align forecasts to observations
        2. Group alignments by season
        3. Compute metrics per season
        4. Generate SeasonalSkillProfile

        Args:
            models_data: Dict mapping model_name -> forecast records
            observations: Observation records
            variable: Variable to benchmark
            metric: Metric for ranking ("mae", "rmse", "bias")
            min_sample_size: Minimum samples for stability

        Returns:
            Dict mapping model_name -> SeasonalSkillProfile
        """
        profiles: dict[str, SeasonalSkillProfile] = {}

        for model_name, forecasts in models_data.items():
            # Filter to variable
            var_forecasts = [r for r in forecasts if r.variable == variable]
            var_observations = [r for r in observations if r.variable == variable]

            if not var_forecasts or not var_observations:
                continue

            # Align
            alignments = self.alignment_engine.align(var_forecasts, var_observations)
            matched = [a for a in alignments if a.is_fully_matched]

            if not matched:
                continue

            # Assign season to each alignment
            for a in matched:
                a.season = _month_to_season(a.valid_time.month)

            # Group by season
            seasonal_groups: dict[str, list[AlignmentRecord]] = defaultdict(list)
            for a in matched:
                seasonal_groups[a.season].append(a)

            # Compute metrics per season
            scores_by_season: dict[str, ModelSkillScore] = {}
            for season, season_alignments in seasonal_groups.items():
                if metric == "mae":
                    evals = self.metrics_engine.compute_mae(season_alignments, group_by=None)
                elif metric == "rmse":
                    evals = self.metrics_engine.compute_rmse(season_alignments, group_by=None)
                elif metric == "bias":
                    evals = self.metrics_engine.compute_bias(season_alignments, group_by=None)
                else:
                    continue

                if evals:
                    ev = evals[0]
                    score = ModelSkillScore(
                        model_name=model_name,
                        variable=variable,
                        lead_time=None,
                        metric_type=metric,
                        value=ev.value,
                        sample_size=ev.sample_size,
                        total_count=ev.total_count,
                        coverage_fraction=ev.coverage_fraction,
                        is_stable=ev.is_stable,
                        min_sample_size=ev.min_sample_size,
                    )
                    scores_by_season[season] = score

            # Build profile
            if scores_by_season:
                profile = SeasonalSkillProfile(
                    model_name=model_name,
                    variable=variable,
                    seasons=scores_by_season,
                )
                profiles[model_name] = profile

        return profiles

    def get_best_season_for_model(
        self,
        models_data: dict[str, list[CanonicalForecastRecord]],
        observations: list[CanonicalForecastRecord],
        model_name: str,
        variable: str,
        metric: str = "mae",
    ) -> tuple[str | None, float | None]:
        """Return the best season for a specific model.

        Best = lowest MAE/RMSE. For bias, closest to zero.

        Args:
            models_data: Dict mapping model_name -> forecast records
            observations: Observation records
            model_name: Model to analyze
            variable: Variable to benchmark
            metric: Metric for ranking

        Returns:
            (best_season, best_metric_value) or (None, None) if no data
        """
        profiles = self.benchmark(models_data, observations, variable, metric)
        profile = profiles.get(model_name)
        if not profile or not profile.seasons:
            return None, None

        best_season = None
        best_value = None

        for season, score in profile.seasons.items():
            val = score.value
            if val is None:
                continue

            if best_value is None:
                best_season = season
                best_value = val
            else:
                # Lower is better for MAE/RMSE
                if metric in ("mae", "rmse", "error_variance"):
                    if val < best_value:
                        best_season = season
                        best_value = val
                elif metric == "bias":
                    if abs(val) < abs(best_value):
                        best_season = season
                        best_value = val

        return best_season, best_value

    def compare_seasonal_profiles(
        self,
        models_data: dict[str, list[CanonicalForecastRecord]],
        observations: list[CanonicalForecastRecord],
        variable: str,
        metric: str = "mae",
    ) -> dict[str, dict[str, float]]:
        """Compare seasonal skill profiles across all models.

        Returns a dict of dicts:
            {season: {model_name: metric_value}}

        Args:
            models_data: Dict mapping model_name -> forecast records
            observations: Observation records
            variable: Variable to benchmark
            metric: Metric for ranking

        Returns:
            Season x Model metric values
        """
        profiles = self.benchmark(models_data, observations, variable, metric)

        # Build season x model table
        seasons = sorted(set(SEASON_MONTHS.keys()))
        table: dict[str, dict[str, float]] = {season: {} for season in seasons}

        for model_name, profile in profiles.items():
            for season, score in profile.seasons.items():
                if score.value is not None:
                    table.setdefault(season, {})[model_name] = score.value

        return table
