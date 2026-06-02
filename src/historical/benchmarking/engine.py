"""Phase 4: Benchmark Engine — orchestrates cross-model comparison.

Provides the core benchmarking pipeline:
1. Discover model capabilities
2. Find common variables and horizons
3. Align forecasts to observations per model
4. Compute metrics per model
5. Rank models by metric
6. Produce BenchmarkResult with fairness warnings

All comparisons are coverage-aware, fairness-checked, and reproducible.
"""

from __future__ import annotations

from typing import Any

from ...forecast.canonical.models import CanonicalForecastRecord
from ...forecast.verification.alignment import AlignmentEngine
from ...forecast.verification.metrics import MetricsEngine
from .fairness import FairnessConstraintChecker
from .models import (
    BenchmarkResult,
    CapabilityMatrix,
    ModelCapability,
    ModelSkillScore,
)


class BenchmarkEngineError(Exception):
    """Raised when benchmarking fails."""


class BenchmarkEngine:
    """Orchestrate multi-model benchmarking.

    Core workflow:
    1. Register models and discover capabilities
    2. Find common variables between models
    3. Align each model's forecasts to observations
    4. Compute metrics per model
    5. Rank models by metric (lowest MAE/RMSE = best)
    6. Return BenchmarkResult with rankings and fairness warnings

    Fairness constraints are always enforced:
    - Only common variables compared
    - Coverage differences reported
    - Unequal sample sizes flagged
    - No silent filtering or interpolation
    """

    def __init__(
        self,
        capability_matrix: CapabilityMatrix | None = None,
        alignment_engine: AlignmentEngine | None = None,
        metrics_engine: MetricsEngine | None = None,
        fairness_checker: FairnessConstraintChecker | None = None,
        min_sample_size: int = 1,
        min_coverage: float = 0.0,
    ) -> None:
        """Initialize benchmark engine.

        Args:
            capability_matrix: Optional CapabilityMatrix with model capabilities
            alignment_engine: Optional AlignmentEngine for forecast-obs alignment
            metrics_engine: Optional MetricsEngine for metric computation
            fairness_checker: Optional FairnessConstraintChecker
            min_sample_size: Minimum samples for stability flag
            min_coverage: Minimum coverage_fraction (0..1)
        """
        self.capability_matrix = capability_matrix or CapabilityMatrix()
        self.alignment_engine = alignment_engine or AlignmentEngine()
        self.metrics_engine = metrics_engine or MetricsEngine(min_sample_size)
        self.fairness_checker = fairness_checker or FairnessConstraintChecker(
            min_coverage=min_coverage, min_sample_size=min_sample_size
        )

    def benchmark_single_variable(
        self,
        models_data: dict[str, list[CanonicalForecastRecord]],
        observations: list[CanonicalForecastRecord],
        variable: str,
        metric: str = "mae",
        lead_time: int | None = None,
        min_sample_size: int = 1,
    ) -> BenchmarkResult:
        """Benchmark multiple models on one variable.

        Aligns each model's forecasts to observations independently,
        computes metrics, and ranks models.

        Fairness:
        - Only variables supported by all models are compared
        - Coverage differences reported per model
        - Unequal sample sizes flagged

        Args:
            models_data: Dict mapping model_name -> forecast records
            observations: Observation records (is_observation=True)
            variable: Variable to benchmark
            metric: Metric for ranking ("mae", "rmse", "bias")
            lead_time: Specific lead time to evaluate, or None (all)
            min_sample_size: Minimum samples for stability

        Returns:
            BenchmarkResult with rankings and fairness warnings
        """
        if lead_time is not None:
            # Filter each model's data to the specific lead time
            filtered_data: dict[str, list[CanonicalForecastRecord]] = {}
            for name, records in models_data.items():
                filtered_data[name] = [
                    r for r in records if r.lead_hours == lead_time and r.variable == variable
                ]
        else:
            filtered_data = {
                name: [r for r in records if r.variable == variable]
                for name, records in models_data.items()
            }

        # Align each model independently
        alignments_by_model: dict[str, list[Any]] = {}
        scores: list[ModelSkillScore] = []
        coverage_stats: dict[str, float] = {}
        sample_sizes: dict[str, int] = {}
        fairness_warnings: list[str] = []

        for model_name, forecasts in filtered_data.items():
            if not forecasts:
                continue

            # Align this model's forecasts to observations
            alignments = self.alignment_engine.align(forecasts, observations)
            alignments_by_model[model_name] = alignments

            # Filter alignments to this variable (should be all)
            var_alignments = [
                a for a in alignments if a.variable == variable and a.is_fully_matched
            ]

            if not var_alignments:
                continue

            # Compute metrics
            if metric == "mae":
                evals = self.metrics_engine.compute_mae(var_alignments, group_by=None)
            elif metric == "rmse":
                evals = self.metrics_engine.compute_rmse(var_alignments, group_by=None)
            elif metric == "bias":
                evals = self.metrics_engine.compute_bias(var_alignments, group_by=None)
            else:
                raise BenchmarkEngineError(f"Unknown metric: {metric}")

            if evals:
                ev = evals[0]
                score = ModelSkillScore(
                    model_name=model_name,
                    variable=variable,
                    lead_time=lead_time,
                    metric_type=metric,
                    value=ev.value,
                    sample_size=ev.sample_size,
                    total_count=ev.total_count,
                    coverage_fraction=ev.coverage_fraction,
                    is_stable=ev.is_stable,
                    min_sample_size=ev.min_sample_size,
                )
                scores.append(score)
                coverage_stats[model_name] = ev.coverage_fraction
                sample_sizes[model_name] = ev.sample_size

        # Rank models (lowest MAE/RMSE = best)
        if metric in ("mae", "rmse", "error_variance"):
            scores.sort(key=lambda s: s.value if s.value is not None else float("inf"))
        elif metric == "bias":
            # For bias, best is closest to zero
            scores.sort(key=lambda s: abs(s.value) if s.value is not None else float("inf"))

        for rank, score in enumerate(scores, start=1):
            score.rank = rank

        best_model = scores[0].model_name if scores else None

        # Check fairness
        for name in filtered_data.keys():
            coverage = coverage_stats.get(name, 0.0)
            if coverage < self.fairness_checker.min_coverage:
                fairness_warnings.append(
                    f"Model '{name}' has low coverage ({coverage:.2%}) "
                    f"for '{variable}' (below threshold {self.fairness_checker.min_coverage:.2%})."
                )

        return BenchmarkResult(
            models_compared=[m for m in filtered_data.keys() if m in sample_sizes],
            variable=variable,
            lead_time=lead_time,
            metric=metric,
            evaluation_window=None,
            best_model=best_model,
            rankings=scores,
            coverage_statistics=coverage_stats,
            sample_sizes=sample_sizes,
            fairness_warnings=fairness_warnings,
        )

    def benchmark_multiple_variables(
        self,
        models_data: dict[str, list[CanonicalForecastRecord]],
        observations: list[CanonicalForecastRecord],
        variables: list[str] | None = None,
        metric: str = "mae",
    ) -> dict[str, BenchmarkResult]:
        """Benchmark multiple models on multiple variables.

        Args:
            models_data: Dict mapping model_name -> forecast records
            observations: Observation records
            variables: List of variables to benchmark, or None (all)
            metric: Metric for ranking

        Returns:
            Dict mapping variable name -> BenchmarkResult
        """
        # Determine which variables to benchmark
        if variables is None:
            all_vars: set[str] = set()
            for records in models_data.values():
                all_vars.update(r.variable for r in records)
            variables = sorted(all_vars)

        results: dict[str, BenchmarkResult] = {}
        for var in variables:
            results[var] = self.benchmark_single_variable(
                models_data, observations, var, metric=metric
            )
        return results

    def register_model(self, capability: ModelCapability) -> None:
        """Register a model's capabilities with the capability matrix.

        Args:
            capability: ModelCapability for this model
        """
        self.capability_matrix.register(capability)
