"""Phase 4: Regional benchmarking for multi-model comparison.

Enables geographic stratification of model skill:
- Group forecasts by location/region
- Compute per-region metrics for each model
- Report coverage differences across regions
- Support custom region definitions (bounding boxes, named locations)

All regional comparisons are coverage-aware and fairness-checked.
"""


from pydantic import BaseModel

from ...forecast.canonical.models import CanonicalForecastRecord
from ...forecast.verification.alignment import AlignmentEngine
from ...forecast.verification.metrics import MetricsEngine
from .fairness import FairnessConstraintChecker
from .models import ModelSkillScore, RegionalBenchmarkResult


class RegionDefinition(BaseModel):
    """Definition of a geographic region for benchmarking."""

    name: str
    description: str = ""

    # Bounding box (degrees)
    min_lat: float | None = None
    max_lat: float | None = None
    min_lon: float | None = None
    max_lon: float | None = None

    # Named locations within this region
    locations: list[tuple[float, float]] = []  # (lat, lon) tuples

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a coordinate is within this region."""
        if self.locations:
            return (lat, lon) in self.locations
        if self.min_lat is not None and self.max_lat is not None:
            if self.min_lon is not None and self.max_lon is not None:
                return self.min_lat <= lat <= self.max_lat and self.min_lon <= lon <= self.max_lon
        return False


class RegionalBenchmark:
    """Regional benchmarking for multi-model comparison.

    Supports:
    - Custom region definitions (bounding boxes, named locations)
    - Per-region metrics computation per model
    - Coverage reporting across regions
    - Fairness checking per region

    Usage:
        regional = RegionalBenchmark()
        regional.add_region(RegionDefinition(
            name="Europe",
            min_lat=35.0, max_lat=60.0,
            min_lon=-10.0, max_lon=40.0,
        ))
        result = regional.benchmark(
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
        fairness_checker: FairnessConstraintChecker | None = None,
    ) -> None:
        """Initialize regional benchmark.

        Args:
            metrics_engine: Optional MetricsEngine
            alignment_engine: Optional AlignmentEngine
            fairness_checker: Optional FairnessConstraintChecker
        """
        self.metrics_engine = metrics_engine or MetricsEngine()
        self.alignment_engine = alignment_engine or AlignmentEngine()
        self.fairness_checker = fairness_checker or FairnessConstraintChecker()
        self.regions: dict[str, RegionDefinition] = {}

    def add_region(self, region: RegionDefinition) -> None:
        """Add a geographic region for benchmarking."""
        self.regions[region.name] = region

    def remove_region(self, name: str) -> bool:
        """Remove a region. Returns True if found and removed."""
        return self.regions.pop(name, None) is not None

    def get_region_for_location(self, lat: float, lon: float) -> str | None:
        """Return the region name for a coordinate, or None."""
        for name, region in self.regions.items():
            if region.contains(lat, lon):
                return name
        return None

    def benchmark(
        self,
        models_data: dict[str, list[CanonicalForecastRecord]],
        observations: list[CanonicalForecastRecord],
        variable: str,
        metric: str = "mae",
        min_sample_size: int = 1,
    ) -> RegionalBenchmarkResult:
        """Run regional benchmark for one variable.

        For each region:
        1. Extract forecasts and observations within the region
        2. Align forecasts to observations
        3. Compute metrics
        4. Generate ModelSkillScore per model

        Args:
            models_data: Dict mapping model_name -> forecast records
            observations: Observation records
            variable: Variable to benchmark
            metric: Metric for ranking ("mae", "rmse", "bias")
            min_sample_size: Minimum samples for stability

        Returns:
            RegionalBenchmarkResult with per-region ModelSkillScore lists.
            If no regions are defined or all regions have insufficient data,
            returns a result with empty region scores and no rankings.
        """
        results_by_region: dict[str, list[ModelSkillScore]] = {}
        coverage_by_region: dict[str, float] = {}
        fairness_warnings: list[str] = []

        # Determine which models have data
        models = [
            name for name in models_data if any(r.variable == variable for r in models_data[name])
        ]

        for region_name, region_def in self.regions.items():
            region_scores: list[ModelSkillScore] = []
            region_total_records = 0
            region_matched_records = 0

            for model_name in models:
                # Filter to this region
                model_records = [
                    r
                    for r in models_data[model_name]
                    if region_def.contains(r.latitude, r.longitude)
                ]

                obs_records = [
                    o for o in observations if region_def.contains(o.latitude, o.longitude)
                ]

                if not model_records or not obs_records:
                    continue

                # Align
                alignments = self.alignment_engine.align(model_records, obs_records)
                var_alignments = [
                    a for a in alignments if a.variable == variable and a.is_fully_matched
                ]

                region_total_records += len(alignments)
                region_matched_records += len(var_alignments)

                # Compute metrics
                if metric == "mae":
                    evals = self.metrics_engine.compute_mae(var_alignments, group_by=None)
                elif metric == "rmse":
                    evals = self.metrics_engine.compute_rmse(var_alignments, group_by=None)
                elif metric == "bias":
                    evals = self.metrics_engine.compute_bias(var_alignments, group_by=None)
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
                    region_scores.append(score)

            # Sort scores by metric value
            if metric in ("mae", "rmse", "error_variance"):
                region_scores.sort(key=lambda s: s.value if s.value is not None else float("inf"))
            elif metric == "bias":
                region_scores.sort(
                    key=lambda s: abs(s.value) if s.value is not None else float("inf")
                )

            # Assign ranks
            for rank, score in enumerate(region_scores, start=1):
                score.rank = rank

            results_by_region[region_name] = region_scores

            if region_total_records > 0:
                coverage_by_region[region_name] = region_matched_records / region_total_records

        # Check fairness across regions
        for region_name, scores in results_by_region.items():
            if len(scores) < 2:
                fairness_warnings.append(
                    f"Region '{region_name}' has only {len(scores)} model(s). "
                    "Cannot perform fair comparison."
                )

        return RegionalBenchmarkResult(
            variable=variable,
            metric=metric,
            regions=results_by_region,
            coverage_by_region=coverage_by_region,
            fairness_warnings=fairness_warnings,
        )
